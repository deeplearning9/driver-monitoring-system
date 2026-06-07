"""
嘴巴状态微调训练脚本
====================

微调 ResNet50 用于分类嘴巴状态（张嘴/闭嘴），用于打哈欠检测。

使用方法：
    python training/train_mouth_state.py --demo
    python training/train_mouth_state.py --epochs 20
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import yaml
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.classification.mouth_state import MouthStateClassifier
from training.engine import Trainer
from training.data_utils import create_dataloaders


def load_config(config_path: str = 'configs/training_config.yaml') -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_demo_data():
    from tools.generate_demo_data import generate_dataset
    generate_dataset(output_dir='data/demo', task='mouth_state',
                     train_per_class=150, val_per_class=30, test_per_class=20)


def main():
    parser = argparse.ArgumentParser(description='嘴巴状态微调训练')
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
    print("嘴巴状态微调训练")
    print("=" * 60)

    config = load_config(args.config)
    task_config = config.get('training', {}).get('mouth_state', {})
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
        data_dir = 'data/demo/mouth_state'
    elif args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = task_config.get('data_dir', 'data/raw/mouth_state')

    task_config['train_dir'] = os.path.join(data_dir, 'train')
    task_config['val_dir'] = os.path.join(data_dir, 'val')
    task_config['test_dir'] = os.path.join(data_dir, 'test')

    dataloaders = create_dataloaders(task_config, augmentation_config,
                                     task_config.get('batch_size', 32), device=device)

    if 'train' not in dataloaders or 'val' not in dataloaders:
        print("错误: 数据集不存在！"); return

    # 模型
    print("\n创建模型...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MouthStateClassifier(num_classes=2, pretrained=True)

    # 训练
    trainer = Trainer(model, dataloaders['train'], dataloaders['val'],
                      task_config, device, args.save_dir, 'mouth_state_resnet')

    # 阶段 1
    print("\n" + "=" * 60)
    print("阶段 1：训练分类头部")
    trainer.freeze_backbone()
    p1 = min(5, task_config.get('epochs', 50) // 3)
    h1 = trainer.train(p1)

    # 阶段 2
    print("\n" + "=" * 60)
    print("阶段 2：全模型微调")
    trainer.unfreeze_backbone()
    h2 = trainer.train(task_config.get('epochs', 50) - p1)

    # 保存
    full = {k: h1.get(k, []) + h2.get(k, []) for k in h1}
    trainer.save_training_artifacts(full)

    print("\n训练完成！")
    print(f"最佳模型: {args.save_dir}/mouth_state_resnet_best.pth")


if __name__ == '__main__':
    main()
