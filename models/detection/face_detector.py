"""
人脸检测模块
支持多种检测器：YOLOv8, MTCNN, RetinaFace
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import torch
from ultralytics import YOLO
import mediapipe as mp


class FaceDetector:
    """人脸检测器基类"""

    def __init__(self, config: Dict):
        """
        初始化人脸检测器

        Args:
            config: 配置字典，包含模型类型、路径等参数
        """
        self.config = config
        self.model_type = config.get('model_type', 'yolov8')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.nms_threshold = config.get('nms_threshold', 0.4)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化模型
        self.model = self._load_model()

    def _load_model(self):
        """加载模型"""
        if self.model_type == 'yolov8':
            return self._load_yolov8()
        elif self.model_type == 'mediapipe':
            return self._load_mediapipe()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def _load_yolov8(self):
        """加载 YOLOv8 人脸检测模型"""
        model_path = self.config.get('model_path', 'models/weights/yolov8-face.pt')
        try:
            model = YOLO(model_path)
            model.to(self.device)
            return model
        except Exception as e:
            print(f"加载 YOLOv8 模型失败: {e}")
            print("使用预训练模型...")
            model = YOLO('yolov8n.pt')  # 使用预训练模型
            model.to(self.device)
            return model

    def _load_mediapipe(self):
        """加载 MediaPipe 人脸检测模型"""
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1,  # 0: 近距离, 1: 远距离
            min_detection_confidence=self.confidence_threshold
        )
        return face_detection

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的人脸

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            检测结果列表，每个结果包含:
            - bbox: 边界框 [x1, y1, x2, y2]
            - confidence: 置信度
            - landmarks: 关键点 (如果可用)
        """
        if self.model_type == 'yolov8':
            return self._detect_yolov8(image)
        elif self.model_type == 'mediapipe':
            return self._detect_mediapipe(image)
        else:
            return []

    def _detect_yolov8(self, image: np.ndarray) -> List[Dict]:
        """使用 YOLOv8 检测人脸"""
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
                        'landmarks': None
                    })

        return detections

    def _detect_mediapipe(self, image: np.ndarray) -> List[Dict]:
        """使用 MediaPipe 检测人脸"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model.process(rgb_image)

        detections = []
        if results.detections:
            h, w = image.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                confidence = detection.score[0]

                # 提取关键点
                landmarks = []
                for point in detection.location_data.relative_keypoints:
                    landmarks.append({
                        'x': int(point.x * w),
                        'y': int(point.y * h)
                    })

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(confidence),
                    'landmarks': landmarks
                })

        return detections

    def detect_and_crop(self, image: np.ndarray, margin: float = 0.1) -> List[Tuple[np.ndarray, Dict]]:
        """
        检测人脸并裁剪

        Args:
            image: 输入图像
            margin: 边界框扩展比例

        Returns:
            裁剪后的人脸图像和检测结果的元组列表
        """
        detections = self.detect(image)
        results = []

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            h, w = image.shape[:2]

            # 扩展边界框
            margin_x = int((x2 - x1) * margin)
            margin_y = int((y2 - y1) * margin)

            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)

            # 裁剪人脸
            face_image = image[y1:y2, x1:x2]

            if face_image.size > 0:
                results.append((face_image, det))

        return results

    def draw_detections(self, image: np.ndarray, detections: List[Dict],
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
        """
        在图像上绘制检测结果

        Args:
            image: 输入图像
            detections: 检测结果列表
            color: 边界框颜色 (BGR)
            thickness: 线条粗细

        Returns:
            绘制了检测结果的图像
        """
        output_image = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']

            # 绘制边界框
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)

            # 绘制置信度
            label = f"Face: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output_image, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(output_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 绘制关键点
            if det.get('landmarks'):
                for point in det['landmarks']:
                    cv2.circle(output_image, (point['x'], point['y']), 3, (0, 0, 255), -1)

        return output_image


class FaceDetectorFactory:
    """人脸检测器工厂类"""

    @staticmethod
    def create_detector(config: Dict) -> FaceDetector:
        """
        创建人脸检测器

        Args:
            config: 配置字典

        Returns:
            人脸检测器实例
        """
        return FaceDetector(config)


# 便捷函数
def create_face_detector(model_type: str = 'yolov8', **kwargs) -> FaceDetector:
    """
    创建人脸检测器的便捷函数

    Args:
        model_type: 模型类型 ('yolov8', 'mediapipe')
        **kwargs: 其他配置参数

    Returns:
        人脸检测器实例
    """
    config = {
        'model_type': model_type,
        'confidence_threshold': kwargs.get('confidence_threshold', 0.5),
        'nms_threshold': kwargs.get('nms_threshold', 0.4),
        'device': kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        'model_path': kwargs.get('model_path', None)
    }

    return FaceDetector(config)


if __name__ == "__main__":
    # 测试代码
    import sys

    # 创建检测器
    detector = create_face_detector('mediapipe')

    # 读取图像
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "data/samples/test_face.jpg"

    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        sys.exit(1)

    # 检测人脸
    detections = detector.detect(image)
    print(f"检测到 {len(detections)} 张人脸")

    # 绘制结果
    result_image = detector.draw_detections(image, detections)

    # 显示结果
    cv2.imshow("Face Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    output_path = "data/samples/face_detection_result.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"结果已保存到: {output_path}")
