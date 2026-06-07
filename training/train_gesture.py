"""
手势识别微调训练脚本
====================

微调 EfficientNet-B0 用于 10 类手势识别。

使用方法：
    python training/train_gesture.py --demo
    python training/train_gesture.py --epochs 30
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import yaml
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.classification.gesture_classifier import GestureClassifier
from training.engine import Trainer
from training.data_utils import create_dataloaders


def load_config(config_path: str = 'configs/training_config.yaml') -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_demo_data():
    from tools.generate_demo_data import generate_dataset
    generate_dataset(output_dir='data/demo', task='gesture',
                     train_per_class=50, val_per_class=10, test_per_class=10)


def main():
    parser = argparse.ArgumentParser(description='手势识别微调训练')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default='models/weights')
    args = parser.parse_args()

    print("=" * 60)
    print("手势识别微调训练")
    print("=" * 60)

    config = load_config(args.config)
    task_config = config.get('training', {}).get('gesture', {})
    augmentation_config = config.get('augmentation', {})

    if args.epochs: task_config['epochs'] = args.epochs
    if args.batch_size: task_config['batch_size'] = args.batch_size
    if args.lr: task_config['learning_rate'] = args.lr

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")

    # 数据集
    print("\n准备数据集...")
    if args.demo:
        generate_demo_data()
        data_dir = 'data/demo/gesture'
    elif args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = task_config.get('data_dir', 'data/raw/gesture')

    task_config['train_dir'] = os.path.join(data_dir, 'train')
    task_config['val_dir'] = os.path.join(data_dir, 'val')
    task_config['test_dir'] = os.path.join(data_dir, 'test')

    dataloaders = create_dataloaders(task_config, augmentation_config,
                                     task_config.get('batch_size', 64), device=device)

    if 'train' not in dataloaders or 'val' not in dataloaders:
        print("错误: 数据集不存在！"); return

    # 模型
    print("\n创建模型...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GestureClassifier(num_classes=10, pretrained=True)

    # 训练
    trainer = Trainer(model, dataloaders['train'], dataloaders['val'],
                      task_config, device, args.save_dir, 'gesture_efficientnet')

    # 阶段 1：只训练头部
    print("\n" + "=" * 60)
    print("阶段 1：训练分类头部")
    trainer.freeze_backbone()
    p1 = min(5, task_config.get('epochs', 80) // 4)
    h1 = trainer.train(p1)

    # 阶段 2：解冻部分层
    print("\n" + "=" * 60)
    print("阶段 2：解冻最后部分层")
    trainer.unfreeze_backbone()
    p2 = min(10, task_config.get('epochs', 80) // 3)
    h2 = trainer.train(p2)

    # 阶段 3：全模型微调
    print("\n" + "=" * 60)
    print("阶段 3：全模型微调")
    p3 = task_config.get('epochs', 80) - p1 - p2
    if p3 > 0:
        h3 = trainer.train(p3)
    else:
        h3 = {}

    # 保存
    full = {}
    for k in set(list(h1.keys()) + list(h2.keys()) + list(h3.keys())):
        full[k] = h1.get(k, []) + h2.get(k, []) + h3.get(k, [])
    trainer.save_training_artifacts(full)

    print("\n训练完成！")
    print(f"最佳模型: {args.save_dir}/gesture_efficientnet_best.pth")


if __name__ == '__main__':
    main()
