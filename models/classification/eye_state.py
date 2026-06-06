"""
眼睛状态分类模块
检测眼睛是睁开还是闭合，用于疲劳检测
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from typing import Dict, Tuple, Optional, List
from PIL import Image


class EyeStateClassifier(nn.Module):
    """眼睛状态分类网络"""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """
        初始化眼睛状态分类网络

        Args:
            num_classes: 类别数 (2: 睁眼/闭眼)
            pretrained: 是否使用预训练权重
        """
        super(EyeStateClassifier, self).__init__()

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


class EyeStateDetector:
    """眼睛状态检测器"""

    def __init__(self, config: Dict):
        """
        初始化眼睛状态检测器

        Args:
            config: 配置字典
        """
        self.config = config
        self.num_classes = config.get('num_classes', 2)
        self.input_size = config.get('input_size', [224, 224])
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

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
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        """加载模型"""
        model_path = self.config.get('model_path')

        # 创建模型
        model = EyeStateClassifier(
            num_classes=self.num_classes,
            pretrained=False
        )

        # 加载预训练权重
        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                print(f"加载眼睛状态模型: {model_path}")
            except Exception as e:
                print(f"加载模型失败: {e}")
                print("使用随机初始化的模型")
        else:
            print("未指定模型路径，使用随机初始化的模型")

        model = model.to(self.device)
        model.eval()

        return model

    def preprocess(self, eye_image: np.ndarray) -> torch.Tensor:
        """
        预处理眼睛图像

        Args:
            eye_image: 眼睛区域图像 (BGR格式)

        Returns:
            预处理后的张量
        """
        # 转换为 RGB
        rgb_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)

        # 转换为 PIL 图像
        pil_image = Image.fromarray(rgb_image)

        # 应用变换
        tensor = self.transform(pil_image)

        # 添加批次维度
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def predict(self, eye_image: np.ndarray) -> Dict:
        """
        预测眼睛状态

        Args:
            eye_image: 眼睛区域图像

        Returns:
            预测结果字典:
            - state: 状态 ('open' 或 'closed')
            - confidence: 置信度
            - probabilities: 各类别概率
        """
        # 预处理
        input_tensor = self.preprocess(eye_image)

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

    def predict_batch(self, eye_images: List[np.ndarray]) -> List[Dict]:
        """
        批量预测眼睛状态

        Args:
            eye_images: 眼睛区域图像列表

        Returns:
            预测结果列表
        """
        results = []

        for eye_image in eye_images:
            result = self.predict(eye_image)
            results.append(result)

        return results

    def calculate_ear(self, eye_landmarks: List[Dict]) -> float:
        """
        计算眼睛纵横比 (Eye Aspect Ratio, EAR)

        EAR 用于判断眼睛是否闭合

        Args:
            eye_landmarks: 眼睛关键点列表

        Returns:
            EAR 值
        """
        if len(eye_landmarks) < 6:
            return 0.0

        # 计算垂直距离
        # 上眼睑到下眼睑的距离
        v1 = np.linalg.norm(np.array([
            eye_landmarks[1]['x'] - eye_landmarks[5]['x'],
            eye_landmarks[1]['y'] - eye_landmarks[5]['y']
        ]))

        v2 = np.linalg.norm(np.array([
            eye_landmarks[2]['x'] - eye_landmarks[4]['x'],
            eye_landmarks[2]['y'] - eye_landmarks[4]['y']
        ]))

        # 计算水平距离
        # 眼角之间的距离
        h = np.linalg.norm(np.array([
            eye_landmarks[0]['x'] - eye_landmarks[3]['x'],
            eye_landmarks[0]['y'] - eye_landmarks[3]['y']
        ]))

        # 计算 EAR
        if h == 0:
            return 0.0

        ear = (v1 + v2) / (2.0 * h)

        return ear

    def detect_fatigue(self, eye_states: List[Dict],
                      ear_threshold: float = 0.25,
                      consecutive_frames: int = 3) -> Dict:
        """
        检测疲劳状态

        Args:
            eye_states: 眼睛状态列表 (历史记录)
            ear_threshold: EAR 阈值
            consecutive_frames: 连续帧数阈值

        Returns:
            疲劳检测结果
        """
        if len(eye_states) < consecutive_frames:
            return {
                'is_fatigued': False,
                'fatigue_score': 0.0,
                'closed_eyes_count': 0
            }

        # 统计闭眼帧数
        closed_count = sum(1 for state in eye_states[-consecutive_frames:]
                          if state['state'] == 'closed')

        # 判断是否疲劳
        is_fatigued = closed_count >= consecutive_frames

        # 计算疲劳分数
        fatigue_score = closed_count / consecutive_frames

        return {
            'is_fatigued': is_fatigued,
            'fatigue_score': fatigue_score,
            'closed_eyes_count': closed_count
        }

    def draw_result(self, image: np.ndarray, bbox: List[int],
                   result: Dict) -> np.ndarray:
        """
        在图像上绘制检测结果

        Args:
            image: 输入图像
            bbox: 眼睛边界框 [x1, y1, x2, y2]
            result: 检测结果

        Returns:
            绘制了结果的图像
        """
        output_image = image.copy()
        x1, y1, x2, y2 = bbox

        # 根据状态选择颜色
        if result['state'] == 'open':
            color = (0, 255, 0)  # 绿色 - 睁眼
        else:
            color = (0, 0, 255)  # 红色 - 闭眼

        # 绘制边界框
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

        # 绘制标签
        label = f"{result['state']}: {result['confidence']:.2f}"
        cv2.putText(output_image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return output_image


class EyeStateDetectorFactory:
    """眼睛状态检测器工厂类"""

    @staticmethod
    def create_detector(config: Dict) -> EyeStateDetector:
        """创建眼睛状态检测器"""
        return EyeStateDetector(config)


# 便捷函数
def create_eye_state_detector(**kwargs) -> EyeStateDetector:
    """
    创建眼睛状态检测器的便捷函数

    Args:
        **kwargs: 配置参数

    Returns:
        眼睛状态检测器实例
    """
    config = {
        'model_path': kwargs.get('model_path', None),
        'num_classes': kwargs.get('num_classes', 2),
        'input_size': kwargs.get('input_size', [224, 224]),
        'confidence_threshold': kwargs.get('confidence_threshold', 0.8),
        'device': kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    }

    return EyeStateDetector(config)


if __name__ == "__main__":
    # 测试代码
    import sys

    # 创建检测器
    detector = create_eye_state_detector()

    # 读取图像
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "data/samples/test_eye.jpg"

    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        sys.exit(1)

    # 预测眼睛状态
    result = detector.predict(image)
    print(f"眼睛状态: {result['state']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"概率分布: {result['probabilities']}")

    # 绘制结果
    bbox = [50, 50, 150, 100]  # 示例边界框
    result_image = detector.draw_result(image, bbox, result)

    # 显示结果
    cv2.imshow("Eye State Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
