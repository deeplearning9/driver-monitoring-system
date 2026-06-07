"""
模型评估工具
评估训练好的模型在测试集上的表现

使用方法：
    python tools/evaluate_model.py --task eye_state --weights models/weights/eye_state_resnet_best.pth
    python tools/evaluate_model.py --task eye_state --demo
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def evaluate(model, test_loader, class_names, device='cpu'):
    """
    评估模型

    Args:
        model: 模型
        test_loader: 测试数据加载器
        class_names: 类别名称列表
        device: 设备
    """
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算准确率
    accuracy = np.mean(all_preds == all_labels)
    print(f"\n测试集准确率: {accuracy:.4f} ({accuracy*100:.1f}%)")

    # 分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names, digits=4))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("混淆矩阵:")
    print(cm)

    return accuracy


def main():
    parser = argparse.ArgumentParser(description='模型评估')
    parser.add_argument('--task', type=str, required=True,
                        choices=['eye_state', 'mouth_state', 'gesture'])
    parser.add_argument('--weights', type=str, default=None,
                        help='模型权重路径')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='测试数据目录')
    parser.add_argument('--demo', action='store_true',
                        help='使用演示数据')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    print("=" * 50)
    print(f"模型评估: {args.task}")
    print("=" * 50)

    # 加载模型
    if args.task == 'eye_state':
        from models.classification.eye_state import EyeStateClassifier
        model = EyeStateClassifier(num_classes=2, pretrained=False)
        class_names = ['closed', 'open']
        default_weights = 'models/weights/eye_state_resnet_best.pth'
    elif args.task == 'mouth_state':
        from models.classification.mouth_state import MouthStateClassifier
        model = MouthStateClassifier(num_classes=2, pretrained=False)
        class_names = ['closed', 'open']
        default_weights = 'models/weights/mouth_state_resnet_best.pth'
    elif args.task == 'gesture':
        from models.classification.gesture_classifier import GestureClassifier
        model = GestureClassifier(num_classes=10, pretrained=False)
        class_names = ['fist', 'open_palm', 'thumbs_up', 'thumbs_down',
                       'peace', 'ok', 'pointing', 'wave', 'pinch', 'none']
        default_weights = 'models/weights/gesture_efficientnet_best.pth'

    weights_path = args.weights or default_weights
    if not os.path.exists(weights_path):
        print(f"错误: 权重文件不存在: {weights_path}")
        print("请先运行训练脚本")
        return

    print(f"加载权重: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=args.device))

    # 加载测试数据
    if args.demo:
        data_dir = f'data/demo/{args.task}'
    elif args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = f'data/raw/{args.task}'

    test_dir = os.path.join(data_dir, 'test')
    if not os.path.exists(test_dir):
        print(f"错误: 测试数据目录不存在: {test_dir}")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"测试样本数: {len(test_dataset)}")
    print(f"类别: {test_dataset.class_to_idx}")

    # 评估
    evaluate(model, test_loader, class_names, args.device)


if __name__ == '__main__':
    main()
