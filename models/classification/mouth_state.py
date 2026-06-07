"""
嘴巴状态分类模块
检测嘴巴是张开还是闭合，用于打哈欠检测
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from typing import Dict, Tuple, Optional, List
from PIL import Image


class MouthStateClassifier(nn.Module):
    """嘴巴状态分类网络"""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """
        初始化嘴巴状态分类网络

        Args:
            num_classes: 类别数 (2: 张嘴/闭嘴)
            pretrained: 是否使用预训练权重
        """
        super(MouthStateClassifier, self).__init__()

        # 使用 ResNet50 作为骨干网络
        self.backbone = models.resnet50(pretrained=pretrained)

        # 修改最后一层全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            分类结果
        """
        return self.backbone(x)


class MouthStateDetector:
    """嘴巴状态检测器"""

    def __init__(self, config: Dict):
        """
        初始化嘴巴状态检测器

        Args:
            config: 配置字典
        """
        self.config = config
        self.num_classes = config.get('num_classes', 2)
        self.input_size = config.get('input_size', [224, 224])
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.mar_threshold = config.get('mar_threshold', 0.35)

        # 类别标签
        self.classes = ['closed', 'open']

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 加载模型
        self.use_mar = False
        self.model = self._load_model()

        # MAR 模式下使用 MediaPipe face mesh
        if self.use_mar:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            # MediaPipe 嘴巴关键点索引（上下嘴唇 + 嘴角）
            self.MOUTH_EAR_INDICES = [13, 14, 78, 308, 82, 312]
            print("使用 MAR 模式检测嘴巴状态（基于面部关键点）")

    def _load_model(self) -> nn.Module:
        """加载模型"""
        model_path = self.config.get('model_path')

        if not model_path:
            print("未指定模型路径，将使用 MAR 模式")
            self.use_mar = True
            return None

        # 创建模型
        model = MouthStateClassifier(
            num_classes=self.num_classes,
            pretrained=False
        )

        # 加载预训练权重
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"加载嘴巴状态模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("将使用 MAR 模式")
            self.use_mar = True
            return None

        model = model.to(self.device)
        model.eval()

        return model

    def _calculate_mar_from_landmarks(self, landmarks, image_w, image_h):
        """从 MediaPipe 关键点计算 MAR"""
        # 上唇中心 (13), 下唇中心 (14)
        top_lip = (landmarks[13].x * image_w, landmarks[13].y * image_h)
        bottom_lip = (landmarks[14].x * image_w, landmarks[14].y * image_h)

        # 左嘴角 (78), 右嘴角 (308)
        left_corner = (landmarks[78].x * image_w, landmarks[78].y * image_h)
        right_corner = (landmarks[308].x * image_w, landmarks[308].y * image_h)

        # 垂直距离（上下唇）
        v = np.linalg.norm(np.array(top_lip) - np.array(bottom_lip))

        # 水平距离（嘴角）
        h = np.linalg.norm(np.array(left_corner) - np.array(right_corner))

        if h == 0:
            return 0.0

        mar = v / h
        return mar

    def _predict_mar(self, mouth_image: np.ndarray) -> Dict:
        """使用 MAR 方法预测嘴巴状态"""
        rgb_image = cv2.cvtColor(mouth_image, cv2.COLOR_BGR2RGB)
        h, w = mouth_image.shape[:2]

        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return {
                'state': 'closed',
                'confidence': 0.5,
                'probabilities': {'closed': 0.5, 'open': 0.5}
            }

        face_landmarks = results.multi_face_landmarks[0].landmark
        mar = self._calculate_mar_from_landmarks(face_landmarks, w, h)

        # 判断状态
        if mar > self.mar_threshold:
            state = 'open'
            confidence = min(1.0, mar / (self.mar_threshold * 2))
        else:
            state = 'closed'
            confidence = min(1.0, (self.mar_threshold - mar) / self.mar_threshold)

        # 转换为概率
        if state == 'open':
            prob_open = 0.5 + confidence * 0.5
            prob_closed = 1.0 - prob_open
        else:
            prob_closed = 0.5 + confidence * 0.5
            prob_open = 1.0 - prob_closed

        return {
            'state': state,
            'confidence': confidence,
            'probabilities': {'closed': prob_closed, 'open': prob_open}
        }

    def preprocess(self, mouth_image: np.ndarray) -> torch.Tensor:
        """
        预处理嘴巴图像

        Args:
            mouth_image: 嘴巴区域图像 (BGR格式)

        Returns:
            预处理后的张量
        """
        # 转换为 RGB
        rgb_image = cv2.cvtColor(mouth_image, cv2.COLOR_BGR2RGB)

        # 转换为 PIL 图像
        pil_image = Image.fromarray(rgb_image)

        # 应用变换
        tensor = self.transform(pil_image)

        # 添加批次维度
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def predict(self, mouth_image: np.ndarray) -> Dict:
        """
        预测嘴巴状态

        Args:
            mouth_image: 嘴巴区域图像

        Returns:
            预测结果字典:
            - state: 状态 ('open' 或 'closed')
            - confidence: 置信度
            - probabilities: 各类别概率
        """
        # 使用 MAR 模式
        if self.use_mar:
            return self._predict_mar(mouth_image)

        # 使用 CNN 模型
        # 预处理
        input_tensor = self.preprocess(mouth_image)

        # 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # 获取结果
        state = self.classes[predicted.item()]
        confidence = confidence.item()
        probs = probabilities[0].cpu().numpy()

        return {
            'state': state,
            'confidence': confidence,
            'probabilities': {
                'closed': float(probs[0]),
                'open': float(probs[1])
            }
        }

    def predict_batch(self, mouth_images: List[np.ndarray]) -> List[Dict]:
        """
        批量预测嘴巴状态

        Args:
            mouth_images: 嘴巴区域图像列表

        Returns:
            预测结果列表
        """
        results = []

        for mouth_image in mouth_images:
            result = self.predict(mouth_image)
            results.append(result)

        return results

    def calculate_mar(self, mouth_landmarks: List[Dict]) -> float:
        """
        计算嘴巴纵横比 (Mouth Aspect Ratio, MAR)

        MAR 用于判断嘴巴是否张开（打哈欠）

        Args:
            mouth_landmarks: 嘴巴关键点列表

        Returns:
            MAR 值
        """
        if len(mouth_landmarks) < 8:
            return 0.0

        # 计算垂直距离
        # 上唇到下唇的距离
        v1 = np.linalg.norm(np.array([
            mouth_landmarks[2]['x'] - mouth_landmarks[6]['x'],
            mouth_landmarks[2]['y'] - mouth_landmarks[6]['y']
        ]))

        v2 = np.linalg.norm(np.array([
            mouth_landmarks[3]['x'] - mouth_landmarks[5]['x'],
            mouth_landmarks[3]['y'] - mouth_landmarks[5]['y']
        ]))

        # 计算水平距离
        # 嘴角之间的距离
        h = np.linalg.norm(np.array([
            mouth_landmarks[0]['x'] - mouth_landmarks[4]['x'],
            mouth_landmarks[0]['y'] - mouth_landmarks[4]['y']
        ]))

        # 计算 MAR
        if h == 0:
            return 0.0

        mar = (v1 + v2) / (2.0 * h)

        return mar

    def detect_yawning(self, mouth_states: List[Dict],
                      mar_threshold: float = 0.6,
                      consecutive_frames: int = 5) -> Dict:
        """
        检测打哈欠

        Args:
            mouth_states: 嘴巴状态列表 (历史记录)
            mar_threshold: MAR 阈值
            consecutive_frames: 连续帧数阈值

        Returns:
            打哈欠检测结果
        """
        if len(mouth_states) < consecutive_frames:
            return {
                'is_yawning': False,
                'yawn_score': 0.0,
                'open_mouth_count': 0
            }

        # 统计张嘴帧数
        open_count = sum(1 for state in mouth_states[-consecutive_frames:]
                        if state['state'] == 'open')

        # 判断是否打哈欠
        is_yawning = open_count >= consecutive_frames

        # 计算打哈欠分数
        yawn_score = open_count / consecutive_frames

        return {
            'is_yawning': is_yawning,
            'yawn_score': yawn_score,
            'open_mouth_count': open_count
        }

    def draw_result(self, image: np.ndarray, bbox: List[int],
                   result: Dict) -> np.ndarray:
        """
        在图像上绘制检测结果

        Args:
            image: 输入图像
            bbox: 嘴巴边界框 [x1, y1, x2, y2]
            result: 检测结果

        Returns:
            绘制了结果的图像
        """
        output_image = image.copy()
        x1, y1, x2, y2 = bbox

        # 根据状态选择颜色
        if result['state'] == 'closed':
            color = (0, 255, 0)  # 绿色 - 闭嘴
        else:
            color = (0, 0, 255)  # 红色 - 张嘴

        # 绘制边界框
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

        # 绘制标签
        label = f"{result['state']}: {result['confidence']:.2f}"
        cv2.putText(output_image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return output_image


class MouthStateDetectorFactory:
    """嘴巴状态检测器工厂类"""

    @staticmethod
    def create_detector(config: Dict) -> MouthStateDetector:
        """创建嘴巴状态检测器"""
        return MouthStateDetector(config)


# 便捷函数
def create_mouth_state_detector(**kwargs) -> MouthStateDetector:
    """
    创建嘴巴状态检测器的便捷函数

    Args:
        **kwargs: 配置参数

    Returns:
        嘴巴状态检测器实例
    """
    config = {
        'model_path': kwargs.get('model_path', None),
        'num_classes': kwargs.get('num_classes', 2),
        'input_size': kwargs.get('input_size', [224, 224]),
        'confidence_threshold': kwargs.get('confidence_threshold', 0.8),
        'device': kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    }

    return MouthStateDetector(config)


if __name__ == "__main__":
    # 测试代码
    import sys

    # 创建检测器
    detector = create_mouth_state_detector()

    # 读取图像
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "data/samples/test_mouth.jpg"

    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        sys.exit(1)

    # 预测嘴巴状态
    result = detector.predict(image)
    print(f"嘴巴状态: {result['state']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"概率分布: {result['probabilities']}")

    # 绘制结果
    bbox = [50, 50, 150, 100]  # 示例边界框
    result_image = detector.draw_result(image, bbox, result)

    # 显示结果
    cv2.imshow("Mouth State Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
