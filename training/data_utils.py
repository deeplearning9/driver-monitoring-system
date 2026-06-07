"""
数据加载工具模块
提供数据集加载、数据增强、DataLoader 创建等功能

学习笔记：
- 数据增强（Data Augmentation）：通过对训练图片做随机变换来增加数据多样性
  ，防止模型过拟合。常用变换包括翻转、旋转、颜色抖动等。
- ImageFolder：PyTorch 提供的数据集类，自动从文件夹结构读取图片并分配标签。
  目录结构要求：
    data/train/
      ├── class_a/    ← 标签 0
      │   ├── img1.jpg
      │   └── img2.jpg
      └── class_b/    ← 标签 1
          ├── img3.jpg
          └── img4.jpg
"""

import os
import platform
from typing import Dict, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(augmentation_config: Dict, split: str) -> transforms.Compose:
    """
    根据配置构建数据变换管道

    学习笔记：
    - 训练集需要数据增强（增加多样性，防止过拟合）
    - 验证集/测试集只需要基础预处理（保证评估一致性）
    - 所有图片都需要 Normalize（使用 ImageNet 的均值和标准差）

    Args:
        augmentation_config: 数据增强配置字典
        split: 数据集划分 ('train', 'val', 'test')

    Returns:
        transforms.Compose 对象
    """
    # ImageNet 标准化参数（所有预训练模型都使用这组值）
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if split == 'train':
        # 训练集：应用数据增强
        transform_list = [
            transforms.Resize((256, 256)),  # 先放大到 256x256
            transforms.RandomCrop(224),      # 再随机裁剪到 224x224
            transforms.RandomHorizontalFlip(p=0.5),  # 50% 概率水平翻转
            transforms.RandomRotation(15),   # 随机旋转 ±15°
            transforms.ColorJitter(          # 颜色抖动
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),           # 转为张量 [0, 1]
            transforms.Normalize(imagenet_mean, imagenet_std),  # 标准化
        ]
    else:
        # 验证集/测试集：只做基础预处理
        transform_list = [
            transforms.Resize((224, 224)),   # 缩放到 224x224
            transforms.ToTensor(),           # 转为张量 [0, 1]
            transforms.Normalize(imagenet_mean, imagenet_std),  # 标准化
        ]

    return transforms.Compose(transform_list)


def create_dataloaders(
    task_config: Dict,
    augmentation_config: Dict,
    batch_size: int,
    num_workers: int = 0,
    device: str = 'cpu'
) -> Dict[str, DataLoader]:
    """
    创建训练、验证、测试数据加载器

    学习笔记：
    - DataLoader 是 PyTorch 的数据加载工具，支持批量加载、打乱、多进程等
    - batch_size：每次训练送入多少张图片
    - shuffle=True：训练时打乱数据顺序（增加随机性）
    - num_workers：加载数据的进程数（Windows 下建议设为 0）

    Args:
        task_config: 任务配置（包含数据集路径）
        augmentation_config: 数据增强配置
        batch_size: 批次大小
        num_workers: 数据加载进程数
        device: 设备类型

    Returns:
        包含 'train', 'val', 'test' DataLoader 的字典
    """
    # Windows 不支持 fork 多进程，必须设为 0
    if platform.system() == 'Windows':
        num_workers = 0
        print("  [Windows] num_workers 已设为 0（避免多进程错误）")

    dataloaders = {}

    for split in ['train', 'val', 'test']:
        # 获取数据集路径
        data_dir = task_config.get(f'{split}_dir')
        if data_dir is None:
            data_dir = task_config.get('data_dir', '')
            data_dir = os.path.join(data_dir, split)

        if not os.path.exists(data_dir):
            print(f"  警告: {split} 数据集目录不存在: {data_dir}")
            continue

        # 构建数据变换
        transform = build_transforms(augmentation_config, split)

        # 使用 ImageFolder 加载数据集
        dataset = datasets.ImageFolder(data_dir, transform=transform)

        # 打印数据集信息
        print(f"  {split} 数据集:")
        print(f"    路径: {data_dir}")
        print(f"    样本数: {len(dataset)}")
        print(f"    类别: {dataset.class_to_idx}")

        # 创建 DataLoader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),  # 只有训练集打乱
            num_workers=num_workers,
            pin_memory=(device == 'cuda'),  # GPU 时启用
            drop_last=(split == 'train'),  # 训练集丢弃最后不完整的 batch
        )

        dataloaders[split] = loader

    return dataloaders
