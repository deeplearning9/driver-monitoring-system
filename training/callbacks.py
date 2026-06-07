"""
训练回调模块
包含早停、检查点保存、指标记录等功能

学习笔记：
- EarlyStopping：当验证集指标不再提升时提前终止训练，防止过拟合
- CheckpointSaver：保存最佳模型权重，用于后续推理或恢复训练
- MetricLogger：记录每个 epoch 的指标，用于可视化训练过程
"""

import os
import json
import time
from typing import Dict, List, Optional
from collections import defaultdict

import torch
import torch.nn as nn


class EarlyStopping:
    """
    早停回调

    原理：监控验证集指标，如果连续 patience 个 epoch 没有提升，
    则认为模型已经过拟合，停止训练。

    使用示例：
        early_stopping = EarlyStopping(patience=10, mode='min')
        for epoch in range(100):
            val_loss = validate(...)
            if early_stopping(val_loss):
                print("Early stopping triggered!")
                break
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001,
                 mode: str = 'min', verbose: bool = True):
        """
        Args:
            patience: 容忍多少个 epoch 没有提升
            min_delta: 最小改善幅度（低于此值不算提升）
            mode: 'min' 表示指标越小越好（如 loss），'max' 表示越大越好（如 accuracy）
            verbose: 是否打印早停信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0  # 连续没有提升的 epoch 计数
        self.best_score = None  # 历史最佳指标值
        self.early_stop = False  # 是否触发早停

    def __call__(self, metric: float) -> bool:
        """
        检查是否应该早停

        Args:
            metric: 当前 epoch 的验证指标

        Returns:
            True 表示应该停止训练
        """
        if self.best_score is None:
            # 第一个 epoch，记录初始值
            self.best_score = metric
            return False

        # 判断是否有提升
        if self.mode == 'min':
            improved = metric < self.best_score - self.min_delta
        else:
            improved = metric > self.best_score + self.min_delta

        if improved:
            # 有提升，更新最佳值，重置计数器
            self.best_score = metric
            self.counter = 0
        else:
            # 没有提升，计数器加 1
            self.counter += 1
            if self.verbose:
                print(f"  早停计数器: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  早停触发！连续 {self.patience} 个 epoch 没有改善")
                return True

        return False


class CheckpointSaver:
    """
    检查点保存器

    原理：当验证指标改善时保存模型权重。
    保存的是 state_dict（模型参数），不是整个模型对象，
    这样更节省空间，也更灵活。

    使用示例：
        saver = CheckpointSaver(save_dir='models/weights', task_name='eye_state')
        for epoch in range(100):
            val_loss = validate(...)
            saver(model, val_loss, epoch)
    """

    def __init__(self, save_dir: str, task_name: str,
                 mode: str = 'min', verbose: bool = True):
        """
        Args:
            save_dir: 保存目录
            task_name: 任务名称（用于文件名前缀）
            mode: 'min' 表示指标越小越好，'max' 表示越大越好
            verbose: 是否打印保存信息
        """
        self.save_dir = save_dir
        self.task_name = task_name
        self.mode = mode
        self.verbose = verbose

        self.best_score = None
        self.best_epoch = 0

        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

    def __call__(self, model: nn.Module, metric: float, epoch: int) -> bool:
        """
        检查并保存最佳模型

        Args:
            model: 模型
            metric: 验证指标
            epoch: 当前 epoch

        Returns:
            True 表示保存了新的最佳模型
        """
        if self.best_score is None:
            self.best_score = metric
            self.best_epoch = epoch
            self._save(model, metric, epoch)
            return True

        # 判断是否有提升
        if self.mode == 'min':
            improved = metric < self.best_score
        else:
            improved = metric > self.best_score

        if improved:
            self.best_score = metric
            self.best_epoch = epoch
            self._save(model, metric, epoch)
            return True

        return False

    def _save(self, model: nn.Module, metric: float, epoch: int):
        """保存模型权重"""
        # 保存路径
        save_path = os.path.join(self.save_dir, f"{self.task_name}_best.pth")

        # 保存 state_dict（推理代码期望的格式）
        torch.save(model.state_dict(), save_path)

        if self.verbose:
            print(f"  保存最佳模型: {save_path}")
            print(f"  Epoch: {epoch + 1}, 指标: {metric:.4f}")


class MetricLogger:
    """
    指标记录器

    原理：记录每个 epoch 的训练和验证指标，
    用于生成训练曲线和最终报告。

    使用示例：
        logger = MetricLogger()
        for epoch in range(100):
            train_loss, train_acc = train_one_epoch(...)
            val_loss, val_acc = validate(...)
            logger.log(train_loss=train_loss, train_acc=train_acc,
                       val_loss=val_loss, val_acc=val_acc)
        history = logger.get_history()
    """

    def __init__(self):
        self.history = defaultdict(list)
        self.epoch_times = []
        self._epoch_start = None

    def start_epoch(self):
        """标记 epoch 开始时间"""
        self._epoch_start = time.time()

    def log(self, **metrics):
        """
        记录一个 epoch 的指标

        Args:
            **metrics: 关键字参数，如 train_loss=0.5, val_acc=0.9
        """
        for key, value in metrics.items():
            self.history[key].append(value)

        # 记录 epoch 耗时
        if self._epoch_start is not None:
            elapsed = time.time() - self._epoch_start
            self.epoch_times.append(elapsed)

    def get_history(self) -> Dict[str, List]:
        """获取训练历史"""
        return dict(self.history)

    def get_epoch_time(self) -> float:
        """获取平均 epoch 耗时"""
        if not self.epoch_times:
            return 0.0
        return sum(self.epoch_times) / len(self.epoch_times)

    def format_epoch_summary(self, epoch: int, total_epochs: int) -> str:
        """
        格式化 epoch 摘要

        Args:
            epoch: 当前 epoch（从 0 开始）
            total_epochs: 总 epoch 数

        Returns:
            格式化的摘要字符串
        """
        parts = [f"Epoch {epoch + 1}/{total_epochs}"]

        for key in ['train_loss', 'val_loss', 'train_acc', 'val_acc']:
            if key in self.history and self.history[key]:
                value = self.history[key][-1]
                parts.append(f"{key}: {value:.4f}")

        if self.epoch_times:
            parts.append(f"耗时: {self.epoch_times[-1]:.1f}s")

        return " | ".join(parts)

    def save_history(self, save_path: str):
        """保存训练历史到 JSON 文件"""
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(dict(self.history), f, indent=2, ensure_ascii=False)
        print(f"训练历史已保存到: {save_path}")

    def print_final_summary(self, best_epoch: int = None):
        """打印最终训练摘要"""
        print("\n" + "=" * 50)
        print("训练完成！")
        print("=" * 50)

        if self.epoch_times:
            total_time = sum(self.epoch_times)
            print(f"总训练时间: {total_time:.1f}s ({total_time/60:.1f}min)")
            print(f"平均每 epoch: {self.get_epoch_time():.1f}s")

        if 'val_loss' in self.history:
            best_idx = best_epoch if best_epoch is not None else \
                min(range(len(self.history['val_loss'])),
                    key=lambda i: self.history['val_loss'][i])
            print(f"最佳 epoch: {best_idx + 1}")
            for key in ['train_loss', 'val_loss', 'train_acc', 'val_acc']:
                if key in self.history and len(self.history[key]) > best_idx:
                    print(f"  {key}: {self.history[key][best_idx]:.4f}")

        print("=" * 50)
