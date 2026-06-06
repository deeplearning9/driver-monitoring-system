"""
可视化工具模块
提供各种可视化功能
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import time


class Visualizer:
    """可视化工具类"""

    def __init__(self, config: Dict = None):
        """
        初始化可视化工具

        Args:
            config: 配置字典
        """
        self.config = config or {}

        # 颜色配置
        self.colors = {
            'face': (0, 255, 0),       # 绿色 - 人脸
            'hand': (255, 0, 0),       # 蓝色 - 手部
            'eye_open': (0, 255, 0),   # 绿色 - 睁眼
            'eye_closed': (0, 0, 255), # 红色 - 闭眼
            'mouth_open': (0, 0, 255), # 红色 - 张嘴
            'mouth_closed': (0, 255, 0), # 绿色 - 闭嘴
            'alert': (0, 0, 255),      # 红色 - 警告
            'normal': (0, 255, 0),     # 绿色 - 正常
            'text': (255, 255, 255),   # 白色 - 文字
            'background': (0, 0, 0),   # 黑色 - 背景
        }

        # 字体配置
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2

        # 历史记录
        self.fps_history = deque(maxlen=30)
        self.state_history = deque(maxlen=100)

    def draw_face_detection(self, image: np.ndarray,
                           detections: List[Dict],
                           show_landmarks: bool = True) -> np.ndarray:
        """
        绘制人脸检测结果

        Args:
            image: 输入图像
            detections: 检测结果列表
            show_landmarks: 是否显示关键点

        Returns:
            绘制后的图像
        """
        output = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']

            # 绘制边界框
            cv2.rectangle(output, (x1, y1), (x2, y2), self.colors['face'], 2)

            # 绘制置信度
            label = f"Face: {confidence:.2f}"
            self._draw_label(output, label, (x1, y1 - 10), self.colors['face'])

            # 绘制关键点
            if show_landmarks and det.get('landmarks'):
                for point in det['landmarks']:
                    cv2.circle(output, (point['x'], point['y']), 3, (0, 0, 255), -1)

        return output

    def draw_eye_state(self, image: np.ndarray,
                      eye_regions: List[Tuple[List[int], Dict]]) -> np.ndarray:
        """
        绘制眼睛状态

        Args:
            image: 输入图像
            eye_regions: 眼睛区域列表 [(bbox, state), ...]

        Returns:
            绘制后的图像
        """
        output = image.copy()

        for bbox, state in eye_regions:
            x1, y1, x2, y2 = bbox

            # 根据状态选择颜色
            if state['state'] == 'open':
                color = self.colors['eye_open']
            else:
                color = self.colors['eye_closed']

            # 绘制边界框
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # 绘制状态标签
            label = f"{state['state']}: {state['confidence']:.2f}"
            self._draw_label(output, label, (x1, y1 - 5), color)

        return output

    def draw_mouth_state(self, image: np.ndarray,
                        mouth_regions: List[Tuple[List[int], Dict]]) -> np.ndarray:
        """
        绘制嘴巴状态

        Args:
            image: 输入图像
            mouth_regions: 嘴巴区域列表 [(bbox, state), ...]

        Returns:
            绘制后的图像
        """
        output = image.copy()

        for bbox, state in mouth_regions:
            x1, y1, x2, y2 = bbox

            # 根据状态选择颜色
            if state['state'] == 'closed':
                color = self.colors['mouth_closed']
            else:
                color = self.colors['mouth_open']

            # 绘制边界框
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # 绘制状态标签
            label = f"{state['state']}: {state['confidence']:.2f}"
            self._draw_label(output, label, (x1, y1 - 5), color)

        return output

    def draw_gesture(self, image: np.ndarray,
                    hand_bbox: List[int],
                    gesture_result: Dict) -> np.ndarray:
        """
        绘制手势识别结果

        Args:
            image: 输入图像
            hand_bbox: 手部边界框
            gesture_result: 手势识别结果

        Returns:
            绘制后的图像
        """
        output = image.copy()
        x1, y1, x2, y2 = hand_bbox

        # 绘制边界框
        cv2.rectangle(output, (x1, y1), (x2, y2), self.colors['hand'], 2)

        # 绘制手势名称
        gesture_cn = gesture_result.get('gesture_cn', gesture_result['gesture'])
        label = f"{gesture_cn}: {gesture_result['confidence']:.2f}"
        self._draw_label(output, label, (x1, y1 - 10), self.colors['hand'])

        # 绘制控制命令
        command = gesture_result.get('command')
        if command:
            cv2.putText(output, f"CMD: {command}", (x1, y2 + 25),
                       self.font, 0.6, (255, 255, 0), 2)

        return output

    def draw_driver_state(self, image: np.ndarray,
                         state: Dict,
                         position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        绘制驾驶员状态信息

        Args:
            image: 输入图像
            state: 驾驶员状态
            position: 显示位置

        Returns:
            绘制后的图像
        """
        output = image.copy()
        x, y = position

        # 状态信息
        fatigue_score = state.get('fatigue_score', 0)
        distraction_score = state.get('distraction_score', 0)
        overall_state = state.get('overall_state', 'normal')

        # 根据整体状态选择颜色
        if overall_state == 'normal':
            state_color = self.colors['normal']
            state_text = "正常"
        elif overall_state == 'fatigued':
            state_color = self.colors['alert']
            state_text = "疲劳"
        elif overall_state == 'distracted':
            state_color = self.colors['alert']
            state_text = "分心"
        else:
            state_color = self.colors['alert']
            state_text = "警告"

        # 绘制背景
        cv2.rectangle(output, (x - 5, y - 25), (x + 300, y + 120),
                     self.colors['background'], -1)

        # 绘制标题
        cv2.putText(output, "Driver Status", (x, y),
                   self.font, 0.8, self.colors['text'], 2)

        # 绘制疲劳分数
        fatigue_color = self.colors['alert'] if fatigue_score > 0.6 else self.colors['normal']
        cv2.putText(output, f"Fatigue: {fatigue_score:.2f}", (x, y + 30),
                   self.font, 0.6, fatigue_color, 2)

        # 绘制分心分数
        distraction_color = self.colors['alert'] if distraction_score > 0.5 else self.colors['normal']
        cv2.putText(output, f"Distraction: {distraction_score:.2f}", (x, y + 60),
                   self.font, 0.6, distraction_color, 2)

        # 绘制整体状态
        cv2.putText(output, f"State: {state_text}", (x, y + 90),
                   self.font, 0.7, state_color, 2)

        return output

    def draw_fps(self, image: np.ndarray, fps: float,
                position: Tuple[int, int] = None) -> np.ndarray:
        """
        绘制 FPS 信息

        Args:
            image: 输入图像
            fps: FPS 值
            position: 显示位置

        Returns:
            绘制后的图像
        """
        output = image.copy()

        if position is None:
            h, w = image.shape[:2]
            position = (w - 150, 30)

        x, y = position

        # 绘制 FPS
        cv2.putText(output, f"FPS: {fps:.1f}", (x, y),
                   self.font, 0.7, self.colors['text'], 2)

        return output

    def draw_alert(self, image: np.ndarray,
                  message: str,
                  severity: str = 'warning') -> np.ndarray:
        """
        绘制警告信息

        Args:
            image: 输入图像
            message: 警告消息
            severity: 严重程度 ('warning', 'danger', 'info')

        Returns:
            绘制后的图像
        """
        output = image.copy()
        h, w = image.shape[:2]

        # 根据严重程度选择颜色
        if severity == 'danger':
            color = (0, 0, 255)  # 红色
            bg_color = (0, 0, 200)
        elif severity == 'warning':
            color = (0, 165, 255)  # 橙色
            bg_color = (0, 130, 200)
        else:
            color = (255, 255, 0)  # 黄色
            bg_color = (200, 200, 0)

        # 绘制背景
        text_size = cv2.getTextSize(message, self.font, 0.8, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h - 50

        cv2.rectangle(output, (text_x - 10, text_y - 30),
                     (text_x + text_size[0] + 10, text_y + 10),
                     bg_color, -1)

        # 绘制文字
        cv2.putText(output, message, (text_x, text_y),
                   self.font, 0.8, color, 2)

        return output

    def create_dashboard(self, image: np.ndarray,
                        detections: Dict,
                        states: Dict) -> np.ndarray:
        """
        创建仪表盘显示

        Args:
            image: 输入图像
            detections: 检测结果
            states: 状态信息

        Returns:
            仪表盘图像
        """
        output = image.copy()

        # 绘制人脸检测
        if 'faces' in detections:
            output = self.draw_face_detection(output, detections['faces'])

        # 绘制眼睛状态
        if 'eyes' in detections:
            output = self.draw_eye_state(output, detections['eyes'])

        # 绘制嘴巴状态
        if 'mouths' in detections:
            output = self.draw_mouth_state(output, detections['mouths'])

        # 绘制手势
        if 'hands' in detections:
            for hand in detections['hands']:
                if 'gesture' in hand:
                    output = self.draw_gesture(output, hand['bbox'], hand['gesture'])

        # 绘制驾驶员状态
        if 'driver_state' in states:
            output = self.draw_driver_state(output, states['driver_state'])

        # 绘制 FPS
        if 'fps' in states:
            output = self.draw_fps(output, states['fps'])

        # 绘制警告
        if 'alert' in states:
            output = self.draw_alert(output, states['alert']['message'],
                                    states['alert']['severity'])

        return output

    def _draw_label(self, image: np.ndarray, text: str,
                   position: Tuple[int, int],
                   color: Tuple[int, int, int]) -> None:
        """绘制文字标签"""
        x, y = position

        # 绘制背景
        text_size = cv2.getTextSize(text, self.font, 0.5, 1)[0]
        cv2.rectangle(image, (x, y - text_size[1] - 5),
                     (x + text_size[0], y + 5), color, -1)

        # 绘制文字
        cv2.putText(image, text, (x, y),
                   self.font, 0.5, self.colors['text'], 1)

    def plot_training_history(self, history: Dict,
                            save_path: Optional[str] = None) -> None:
        """
        绘制训练历史曲线

        Args:
            history: 训练历史字典
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 绘制损失曲线
        if 'train_loss' in history and 'val_loss' in history:
            axes[0].plot(history['train_loss'], label='Train Loss')
            axes[0].plot(history['val_loss'], label='Val Loss')
            axes[0].set_title('Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True)

        # 绘制准确率曲线
        if 'train_acc' in history and 'val_acc' in history:
            axes[1].plot(history['train_acc'], label='Train Acc')
            axes[1].plot(history['val_acc'], label='Val Acc')
            axes[1].set_title('Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存到: {save_path}")

        plt.show()

    def plot_confusion_matrix(self, confusion_matrix: np.ndarray,
                            class_names: List[str],
                            save_path: Optional[str] = None) -> None:
        """
        绘制混淆矩阵

        Args:
            confusion_matrix: 混淆矩阵
            class_names: 类别名称
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='d',
                   cmap='Blues', xticklabels=class_names,
                   yticklabels=class_names)

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到: {save_path}")

        plt.show()


# 便捷函数
def create_visualizer(**kwargs) -> Visualizer:
    """创建可视化工具实例"""
    return Visualizer(kwargs)


if __name__ == "__main__":
    # 测试代码
    visualizer = create_visualizer()

    # 创建测试图像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # 模拟检测结果
    detections = {
        'faces': [{'bbox': [100, 100, 200, 200], 'confidence': 0.95}],
        'eyes': [([120, 120, 160, 140], {'state': 'open', 'confidence': 0.98})],
        'mouths': [([140, 160, 180, 180], {'state': 'closed', 'confidence': 0.95})]
    }

    states = {
        'driver_state': {
            'fatigue_score': 0.3,
            'distraction_score': 0.1,
            'overall_state': 'normal'
        },
        'fps': 30.0
    }

    # 创建仪表盘
    dashboard = visualizer.create_dashboard(test_image, detections, states)

    # 显示结果
    cv2.imshow("Dashboard", dashboard)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
