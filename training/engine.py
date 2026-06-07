"""
通用训练引擎模块
提供 Trainer 类，封装完整的训练循环逻辑

学习笔记：
- 训练循环的核心步骤：
  1. 前向传播：输入图片 → 模型 → 得到预测
  2. 计算损失：预测 vs 真实标签 → loss
  3. 反向传播：loss.backward() 计算梯度
  4. 更新参数：optimizer.step() 根据梯度调整权重
  5. 清零梯度：optimizer.zero_grad() 准备下一轮

- 微调（Fine-tuning）的关键技巧：
  1. 冻结骨干网络：只训练分类头部，让新学到的分类器适配预训练特征
  2. 渐进式解冻：逐步解冻更多层，让模型慢慢适应新任务
  3. 差异化学习率：底层用小学习率，顶层用大学习率
"""

import os
import json
import time
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .callbacks import EarlyStopping, CheckpointSaver, MetricLogger


class Trainer:
    """
    通用训练引擎

    封装了完整的训练流程：训练循环、验证循环、早停、检查点保存等。
    支持层冻结/解冻，适用于微调场景。

    使用示例：
        model = EyeStateClassifier(num_classes=2, pretrained=True)
        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.freeze_backbone()  # 冻结骨干，只训练头部
        trainer.train(epochs=5)
        trainer.unfreeze_backbone()  # 解冻骨干
        trainer.train(epochs=10)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cpu',
        save_dir: str = 'models/weights',
        task_name: str = 'model'
    ):
        """
        Args:
            model: PyTorch 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 训练配置字典
            device: 设备 ('cpu' 或 'cuda')
            save_dir: 模型保存目录
            task_name: 任务名称
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.save_dir = save_dir
        self.task_name = task_name

        # 将模型移到指定设备
        self.model = self.model.to(self.device)

        # 初始化训练组件
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stopping = None
        self.checkpoint_saver = None
        self.metric_logger = None

        self._setup_training()

    def _setup_training(self):
        """初始化优化器、损失函数、调度器等"""
        # 损失函数
        label_smoothing = self.config.get('loss', {}).get('label_smoothing', 0.0)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"  损失函数: CrossEntropyLoss (label_smoothing={label_smoothing})")

        # 优化器
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.0001)
        optimizer_name = self.config.get('optimizer', 'Adam')

        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
        print(f"  优化器: {optimizer_name} (lr={lr}, weight_decay={weight_decay})")

        # 学习率调度器
        scheduler_name = self.config.get('scheduler', 'ReduceLROnPlateau')
        if scheduler_name == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
            )
        elif scheduler_name == 'CosineAnnealingLR':
            T_max = self.config.get('epochs', 50)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max
            )
        elif scheduler_name == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=0.1
            )
        print(f"  调度器: {scheduler_name}")

        # 早停
        early_stopping_config = self.config.get('early_stopping', {})
        # 兼容：如果 early_stopping 是整数（直接指定 patience）
        if isinstance(early_stopping_config, int):
            patience = early_stopping_config
            min_delta = 0.001
        else:
            patience = early_stopping_config.get('patience', 10)
            min_delta = early_stopping_config.get('min_delta', 0.001)
        self.early_stopping = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            mode='min'
        )

        # 检查点保存器
        self.checkpoint_saver = CheckpointSaver(
            save_dir=self.save_dir,
            task_name=self.task_name,
            mode='min'
        )

        # 指标记录器
        self.metric_logger = MetricLogger()

    def freeze_backbone(self):
        """
        冻结骨干网络参数

        学习笔记：
        冻结骨干网络是微调的第一步。
        预训练模型的骨干（如 ResNet 的卷积层）已经学会了提取通用特征，
        我们只需要训练新的分类头部来适配我们的任务。
        这样可以：
        1. 减少需要训练的参数量（从 ~25M 降到 ~几十K）
        2. 防止小数据集上过拟合
        3. 加快训练速度
        """
        print("\n正在冻结骨干网络参数...")

        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 解冻分类头部
        # 根据模型结构找到分类头
        if hasattr(self.model, 'backbone'):
            backbone = self.model.backbone
            # ResNet 系列：解冻 fc 层
            if hasattr(backbone, 'fc'):
                for param in backbone.fc.parameters():
                    param.requires_grad = True
            # EfficientNet 系列：解冻 classifier 层
            elif hasattr(backbone, 'classifier'):
                for param in backbone.classifier.parameters():
                    param.requires_grad = True

        # 打印参数统计
        self._print_param_stats()

    def unfreeze_backbone(self, num_layers: int = None):
        """
        解冻骨干网络参数

        学习笔记：
        解冻骨干网络是微调的第二步。
        当分类头部已经训练得差不多了，可以解冻骨干网络的部分层，
        让模型进一步适应新任务的数据分布。
        通常解冻最后几层（高层特征更任务相关），
        并使用较小的学习率（避免破坏已学到的特征）。

        Args:
            num_layers: 解冻最后几层。None 表示全部解冻。
        """
        print("\n正在解冻骨干网络参数...")

        # 解冻所有参数
        for param in self.model.parameters():
            param.requires_grad = True

        # 如果指定了解冻层数，冻结前面的层
        if num_layers is not None:
            children = list(self.model.named_parameters())
            total = len(children)
            freeze_count = total - num_layers
            for i, (name, param) in enumerate(children):
                if i < freeze_count:
                    param.requires_grad = False
            print(f"  解冻最后 {num_layers} 层（共 {total} 层）")

        # 解冻后降低学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
        print(f"  学习率已降低到 {self.optimizer.param_groups[0]['lr']:.6f}")

        # 打印参数统计
        self._print_param_stats()

    def _print_param_stats(self):
        """打印模型参数统计"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters()
                        if p.requires_grad)
        frozen = total - trainable
        print(f"  总参数: {total:,}")
        print(f"  可训练: {trainable:,}")
        print(f"  已冻结: {frozen:,}")
        print(f"  可训练比例: {100 * trainable / total:.1f}%")

    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        训练一个 epoch

        学习笔记：
        一个 epoch = 所有训练样本都过一遍。
        每个 epoch 的步骤：
        1. 设置模型为训练模式 (model.train())
        2. 遍历所有 batch：
           a. 前向传播 → 得到预测
           b. 计算损失
           c. 反向传播 → 计算梯度
           d. 更新参数
           e. 清零梯度
        3. 返回平均损失和准确率

        Args:
            epoch: 当前 epoch 编号（从 0 开始）

        Returns:
            (平均损失, 准确率)
        """
        self.model.train()  # 训练模式（启用 Dropout、BatchNorm 更新）

        total_loss = 0.0
        correct = 0
        total = 0

        # 进度条
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [训练]")

        for batch_idx, (images, labels) in enumerate(pbar):
            # 移到设备
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 1. 前向传播
            outputs = self.model(images)

            # 2. 计算损失
            loss = self.criterion(outputs, labels)

            # 3. 反向传播
            self.optimizer.zero_grad()  # 清零梯度（PyTorch 默认累加梯度）
            loss.backward()             # 计算梯度

            # 4. 更新参数
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # 取最大值的索引作为预测
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.1f}%"
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()  # 验证时不需要计算梯度
    def validate(self) -> Tuple[float, float]:
        """
        验证模型

        学习笔记：
        验证步骤和训练类似，但：
        1. 不计算梯度（torch.no_grad()）
        2. 不更新参数
        3. 使用 model.eval() 设置模型为评估模式

        Args:
            无

        Returns:
            (平均损失, 准确率)
        """
        self.model.eval()  # 评估模式（关闭 Dropout，固定 BatchNorm）

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(self, num_epochs: int, start_epoch: int = 0) -> Dict:
        """
        执行完整的训练循环

        学习笔记：
        主训练循环的结构：
        for each epoch:
            train_loss, train_acc = train_one_epoch()
            val_loss, val_acc = validate()
            check_early_stopping()
            save_checkpoint_if_best()
            log_metrics()

        Args:
            num_epochs: 训练轮数
            start_epoch: 起始 epoch 编号（用于恢复训练）

        Returns:
            训练历史字典
        """
        print(f"\n开始训练: {num_epochs} 个 epoch")
        print(f"设备: {self.device}")
        print("-" * 60)

        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.metric_logger.start_epoch()

            # 训练
            train_loss, train_acc = self.train_one_epoch(epoch)

            # 验证
            val_loss, val_acc = self.validate()

            # 更新学习率调度器
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # 记录指标
            self.metric_logger.log(
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc
            )

            # 打印 epoch 摘要
            summary = self.metric_logger.format_epoch_summary(epoch, start_epoch + num_epochs)
            print(f"\n{summary}")
            print(f"  当前学习率: {self.optimizer.param_groups[0]['lr']:.6f}")

            # 保存最佳模型
            saved = self.checkpoint_saver(self.model, val_loss, epoch)
            if saved:
                print(f"  ✓ 新的最佳模型已保存")

            # 早停检查
            if self.early_stopping(val_loss):
                print(f"\n早停触发，训练在第 {epoch + 1} 轮终止")
                break

        # 打印最终摘要
        self.metric_logger.print_final_summary(
            best_epoch=self.checkpoint_saver.best_epoch
        )

        return self.metric_logger.get_history()

    def save_training_artifacts(self, history: Dict):
        """
        保存训练产物

        Args:
            history: 训练历史字典
        """
        # 保存最终模型权重（兼容推理代码）
        final_path = os.path.join(self.save_dir, f"{self.task_name}_finetuned.pth")
        torch.save(self.model.state_dict(), final_path)
        print(f"\n最终模型权重已保存到: {final_path}")

        # 保存训练历史
        history_path = os.path.join(self.save_dir, f"{self.task_name}_history.json")
        self.metric_logger.save_history(history_path)

        # 保存训练配置
        config_path = os.path.join(self.save_dir, f"{self.task_name}_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            # 只保存可序列化的配置
            serializable_config = {}
            for k, v in self.config.items():
                try:
                    json.dumps(v)
                    serializable_config[k] = v
                except (TypeError, ValueError):
                    serializable_config[k] = str(v)
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)
        print(f"训练配置已保存到: {config_path}")

        # 保存最佳模型路径信息
        best_path = os.path.join(self.save_dir, f"{self.task_name}_best.pth")
        if os.path.exists(best_path):
            print(f"最佳模型权重: {best_path}")
