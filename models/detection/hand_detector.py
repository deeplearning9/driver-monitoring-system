"""
手部检测模块
支持 MediaPipe 和 YOLOv8 手部检测
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import torch
import mediapipe as mp
from ultralytics import YOLO


class HandDetector:
    """手部检测器"""

    def __init__(self, config: Dict):
        """
        初始化手部检测器

        Args:
            config: 配置字典
        """
        self.config = config
        self.model_type = config.get('model_type', 'mediapipe')
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.max_hands = config.get('max_hands', 2)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化模型
        self.model = self._load_model()

    def _load_model(self):
        """加载模型"""
        if self.model_type == 'mediapipe':
            return self._load_mediapipe()
        elif self.model_type == 'yolov8':
            return self._load_yolov8()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def _load_mediapipe(self):
        """加载 MediaPipe 手部检测模型"""
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.confidence_threshold,
            min_tracking_confidence=self.config.get('tracking_confidence', 0.5)
        )
        return hands

    def _load_yolov8(self):
        """加载 YOLOv8 手部检测模型"""
        model_path = self.config.get('model_path', 'models/weights/yolov8-hand.pt')
        try:
            model = YOLO(model_path)
            model.to(self.device)
            return model
        except Exception as e:
            print(f"加载 YOLOv8 手部检测模型失败: {e}")
            print("使用 MediaPipe 替代...")
            return self._load_mediapipe()

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的手部

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            检测结果列表，每个结果包含:
            - bbox: 边界框 [x1, y1, x2, y2]
            - confidence: 置信度
            - landmarks: 手部关键点
            - handedness: 左手/右手
        """
        if self.model_type == 'mediapipe':
            return self._detect_mediapipe(image)
        elif self.model_type == 'yolov8':
            return self._detect_yolov8(image)
        else:
            return []

    def _detect_mediapipe(self, image: np.ndarray) -> List[Dict]:
        """使用 MediaPipe 检测手部"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model.process(rgb_image)

        detections = []
        if results.multi_hand_landmarks:
            h, w = image.shape[:2]

            for hand_idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # 提取关键点
                landmarks = []
                x_coords = []
                y_coords = []

                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append({'x': x, 'y': y, 'z': landmark.z})
                    x_coords.append(x)
                    y_coords.append(y)

                # 计算边界框
                x1 = max(0, min(x_coords) - 20)
                y1 = max(0, min(y_coords) - 20)
                x2 = min(w, max(x_coords) + 20)
                y2 = min(h, max(y_coords) + 20)

                # 获取手的朝向
                handedness_label = handedness.classification[0].label
                handedness_score = handedness.classification[0].score

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(handedness_score),
                    'landmarks': landmarks,
                    'handedness': handedness_label,  # 'Left' 或 'Right'
                    'hand_idx': hand_idx
                })

        return detections

    def _detect_yolov8(self, image: np.ndarray) -> List[Dict]:
        """使用 YOLOv8 检测手部"""
        results = self.model(image, conf=self.confidence_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()

                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'landmarks': None,
                        'handedness': None,
                        'hand_idx': len(detections)
                    })

        return detections

    def extract_hand_roi(self, image: np.ndarray, detection: Dict,
                        target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
        """
        提取手部区域图像

        Args:
            image: 输入图像
            detection: 检测结果
            target_size: 目标尺寸

        Returns:
            手部区域图像
        """
        x1, y1, x2, y2 = detection['bbox']
        h, w = image.shape[:2]

        # 确保边界框在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # 裁剪手部区域
        hand_image = image[y1:y2, x1:x2]

        if hand_image.size == 0:
            return None

        # 调整大小
        hand_image = cv2.resize(hand_image, target_size)

        return hand_image

    def get_finger_states(self, landmarks: List[Dict]) -> Dict[str, bool]:
        """
        根据关键点判断手指状态

        Args:
            landmarks: 手部关键点列表

        Returns:
            手指状态字典
        """
        if not landmarks or len(landmarks) < 21:
            return {}

        # 手指关键点索引
        # 拇指: 1-4, 食指: 5-8, 中指: 9-12, 无名指: 13-16, 小指: 17-20
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]

        finger_states = {
            'thumb': False,
            'index': False,
            'middle': False,
            'ring': False,
            'pinky': False
        }

        # 判断手指是否伸直
        for i, (tip, pip) in enumerate(zip(finger_tips, finger_pips)):
            tip_point = landmarks[tip]
            pip_point = landmarks[pip]

            # 对于拇指，比较 x 坐标
            if i == 0:
                # 根据手的朝向判断
                wrist = landmarks[0]
                if wrist['x'] < tip_point['x']:  # 右手
                    finger_states['thumb'] = tip_point['x'] > pip_point['x']
                else:  # 左手
                    finger_states['thumb'] = tip_point['x'] < pip_point['x']
            else:
                # 对于其他手指，比较 y 坐标
                finger_states[list(finger_states.keys())[i]] = tip_point['y'] < pip_point['y']

        return finger_states

    def draw_detections(self, image: np.ndarray, detections: List[Dict],
                       draw_landmarks: bool = True,
                       draw_bbox: bool = True,
                       color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        在图像上绘制检测结果

        Args:
            image: 输入图像
            detections: 检测结果列表
            draw_landmarks: 是否绘制关键点
            draw_bbox: 是否绘制边界框
            color: 颜色 (BGR)

        Returns:
            绘制了检测结果的图像
        """
        output_image = image.copy()

        for det in detections:
            if draw_bbox:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

                # 绘制标签
                label = f"Hand: {det['confidence']:.2f}"
                if det.get('handedness'):
                    label = f"{det['handedness']}: {det['confidence']:.2f}"

                cv2.putText(output_image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if draw_landmarks and det.get('landmarks'):
                landmarks = det['landmarks']

                # 绘制关键点
                for point in landmarks:
                    cv2.circle(output_image, (point['x'], point['y']), 3, (0, 0, 255), -1)

                # 绘制连接线
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
                    (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
                    (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
                    (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
                    (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
                    (5, 9), (9, 13), (13, 17)  # 手掌
                ]

                for start_idx, end_idx in connections:
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        start_point = (landmarks[start_idx]['x'], landmarks[start_idx]['y'])
                        end_point = (landmarks[end_idx]['x'], landmarks[end_idx]['y'])
                        cv2.line(output_image, start_point, end_point, (255, 0, 0), 2)

        return output_image


class HandDetectorFactory:
    """手部检测器工厂类"""

    @staticmethod
    def create_detector(config: Dict) -> HandDetector:
        """创建手部检测器"""
        return HandDetector(config)


# 便捷函数
def create_hand_detector(model_type: str = 'mediapipe', **kwargs) -> HandDetector:
    """
    创建手部检测器的便捷函数

    Args:
        model_type: 模型类型 ('mediapipe', 'yolov8')
        **kwargs: 其他配置参数

    Returns:
        手部检测器实例
    """
    config = {
        'model_type': model_type,
        'confidence_threshold': kwargs.get('confidence_threshold', 0.7),
        'max_hands': kwargs.get('max_hands', 2),
        'tracking_confidence': kwargs.get('tracking_confidence', 0.5),
        'device': kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        'model_path': kwargs.get('model_path', None)
    }

    return HandDetector(config)


if __name__ == "__main__":
    # 测试代码
    import sys

    # 创建检测器
    detector = create_hand_detector('mediapipe')

    # 读取图像
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "data/samples/test_hand.jpg"

    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        sys.exit(1)

    # 检测手部
    detections = detector.detect(image)
    print(f"检测到 {len(detections)} 只手")

    # 打印手指状态
    for det in detections:
        if det.get('landmarks'):
            finger_states = detector.get_finger_states(det['landmarks'])
            print(f"手指状态: {finger_states}")

    # 绘制结果
    result_image = detector.draw_detections(image, detections)

    # 显示结果
    cv2.imshow("Hand Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    output_path = "data/samples/hand_detection_result.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"结果已保存到: {output_path}")
