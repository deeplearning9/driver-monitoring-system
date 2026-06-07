"""
眼睛状态微调训练脚本
====================

这个脚本演示如何微调一个 ImageNet 预训练的 ResNet50 模型，
用于分类驾驶员的眼睛状态（睁眼/闭眼）。

微调概念：
1. 迁移学习（Transfer Learning）：利用在大数据集（ImageNet）上学到的通用特征
2. 层冻结（Layer Freezing）：固定骨干网络参数，只训练分类头部
3. 渐进式解冻（Progressive Unfreezing）：逐步解冻更多层进行微调
4. 学习率调度（Learning Rate Scheduling）：随训练进展降低学习率
5. 早停（Early Stopping）：监控验证集损失，防止过拟合

使用方法：
    # 使用合成数据快速验证流程
    python training/train_eye_state.py --demo

    # 指定参数训练
    python training/train_eye_state.py --epochs 20 --batch-size 32

    # 使用真实数据集
    python training/train_eye_state.py --data-dir data/raw/eye_state
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import yaml
import torch

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.classification.eye_state import EyeStateClassifier
from training.engine import Trainer
from training.data_utils import create_dataloaders


def load_config(config_path: str = 'configs/training_config.yaml') -> dict:
    """加载训练配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_demo_data(output_dir: str = 'data/demo/eye_state',
                       train_per_class: int = 150,
                       val_per_class: int = 30,
                       test_per_class: int = 20):
    """生成演示数据集"""
    from tools.generate_demo_data import generate_dataset
    generate_dataset(
        output_dir='data/demo',
        task='eye_state',
        train_per_class=train_per_class,
        val_per_class=val_per_class,
        test_per_class=test_per_class
    )


def main():
    # ============================================================
    # 1. 解析命令行参数
    # ============================================================
    parser = argparse.ArgumentParser(
        description='眼睛状态微调训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python training/train_eye_state.py --demo              # 使用合成数据
  python training/train_eye_state.py --epochs 10         # 训练 10 轮
  python training/train_eye_state.py --data-dir data/raw/eye_state
        """
    )
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='训练配置文件路径')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数（覆盖配置文件）')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='批次大小（覆盖配置文件）')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率（覆盖配置文件）')
    parser.add_argument('--demo', action='store_true',
                        help='使用合成演示数据')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='数据集目录（覆盖配置文件）')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda', 'auto'],
                        help='训练设备')
    parser.add_argument('--save-dir', type=str, default='models/weights',
                        help='模型保存目录')

    args = parser.parse_args()

    # ============================================================
    # 2. 加载配置
    # ============================================================
    print("=" * 60)
    print("眼睛状态微调训练")
    print("=" * 60)

    config = load_config(args.config)
    task_config = config.get('training', {}).get('eye_state', {})
    augmentation_config = config.get('augmentation', {})

    # 命令行参数覆盖配置文件
    if args.epochs is not None:
        task_config['epochs'] = args.epochs
    if args.batch_size is not None:
        task_config['batch_size'] = args.batch_size
    if args.lr is not None:
        task_config['learning_rate'] = args.lr

    # 确定设备
    if args.device == 'auto' or args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"\n配置信息:")
    print(f"  设备: {device}")
    print(f"  轮数: {task_config.get('epochs', 50)}")
    print(f"  批次大小: {task_config.get('batch_size', 32)}")
    print(f"  学习率: {task_config.get('learning_rate', 0.0001)}")
    print(f"  优化器: {task_config.get('optimizer', 'Adam')}")

    # ============================================================
    # 3. 准备数据集
    # ============================================================
    print("\n" + "-" * 60)
    print("准备数据集...")

    if args.demo:
        # 使用合成数据
        print("  模式: 合成演示数据")
        generate_demo_data()
        data_dir = 'data/demo/eye_state'
    elif args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = task_config.get('data_dir', 'data/raw/eye_state')

    # 更新配置中的数据路径
    task_config['train_dir'] = os.path.join(data_dir, 'train')
    task_config['val_dir'] = os.path.join(data_dir, 'val')
    task_config['test_dir'] = os.path.join(data_dir, 'test')

    # 合并顶层配置（loss 等）到 task_config
    if 'loss' in config:
        task_config['loss'] = config['loss']

    # 创建数据加载器
    batch_size = task_config.get('batch_size', 32)
    dataloaders = create_dataloaders(
        task_config=task_config,
        augmentation_config=augmentation_config,
        batch_size=batch_size,
        device=device
    )

    if 'train' not in dataloaders or 'val' not in dataloaders:
        print("错误: 训练集或验证集不存在！")
        return

    # ============================================================
    # 4. 创建模型
    # ============================================================
    print("\n" + "-" * 60)
    print("创建模型...")

    # 抑制 pretrained 参数的弃用警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = EyeStateClassifier(num_classes=2, pretrained=True)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型: EyeStateClassifier (ResNet50)")
    print(f"  总参数量: {total_params:,}")
    print(f"  输入尺寸: 224x224")
    print(f"  输出类别: 2 (closed, open)")

    # ============================================================
    # 5. 创建训练器
    # ============================================================
    print("\n" + "-" * 60)
    print("初始化训练器...")

    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        config=task_config,
        device=device,
        save_dir=args.save_dir,
        task_name='eye_state_resnet'
    )

    # ============================================================
    # 6. 第一阶段：冻结骨干，只训练分类头部
    # ============================================================
    print("\n" + "=" * 60)
    print("第一阶段：训练分类头部（骨干网络冻结）")
    print("=" * 60)
    print("目的：让新的分类头部适应预训练特征，不破坏骨干网络的通用特征")

    trainer.freeze_backbone()
    phase1_epochs = min(5, task_config.get('epochs', 50) // 3)
    history1 = trainer.train(num_epochs=phase1_epochs)

    # ============================================================
    # 7. 第二阶段：解冻骨干网络，全模型微调
    # ============================================================
    print("\n" + "=" * 60)
    print("第二阶段：全模型微调（解冻骨干网络）")
    print("=" * 60)
    print("目的：让骨干网络适应新任务的数据分布，进一步提升精度")

    trainer.unfreeze_backbone()
    phase2_epochs = task_config.get('epochs', 50) - phase1_epochs
    history2 = trainer.train(num_epochs=phase2_epochs)

    # ============================================================
    # 8. 保存训练产物
    # ============================================================
    print("\n" + "-" * 60)
    print("保存训练产物...")

    # 合并两阶段的历史
    full_history = {}
    for key in history1:
        full_history[key] = history1.get(key, []) + history2.get(key, [])

    trainer.save_training_artifacts(full_history)

    # ============================================================
    # 9. 完成
    # ============================================================
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"最佳模型: {args.save_dir}/eye_state_resnet_best.pth")
    print(f"最终权重: {args.save_dir}/eye_state_resnet_finetuned.pth")
    print(f"\n使用方法:")
    print(f"  在 configs/model_config.yaml 中设置:")
    print(f"    eye_state:")
    print(f"      model_path: \"{args.save_dir}/eye_state_resnet_best.pth\"")
    print(f"  然后运行: python inference/demo.py --camera")


if __name__ == '__main__':
    main()
