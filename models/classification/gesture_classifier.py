"""
手势分类模块
识别静态手势，如握拳、张开手掌、竖大拇指等
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from typing import Dict, Tuple, Optional, List
from PIL import Image


class GestureClassifier(nn.Module):
    """手势分类网络"""

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        """
        初始化手势分类网络

        Args:
            num_classes: 类别数
            pretrained: 是否使用预训练权重
        """
        super(GestureClassifier, self).__init__()

        # 使用 EfficientNet-B0 作为骨干网络
        self.backbone = models.efficientnet_b0(pretrained=pretrained)

        # 修改最后一层全连接层
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
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


class GestureRecognizer:
    """手势识别器"""

    def __init__(self, config: Dict):
        """
        初始化手势识别器

        Args:
            config: 配置字典
        """
        self.config = config
        self.num_classes = config.get('num_classes', 10)
        self.input_size = config.get('input_size', [224, 224])
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # 手势类别
        self.classes = config.get('gesture_classes', [
            'fist',        # 握拳
            'open_palm',   # 张开手掌
            'thumbs_up',   # 竖大拇指
            'thumbs_down', # 拇指向下
            'peace',       # 比耶
            'ok',          # OK 手势
            'pointing',    # 指向
            'wave',        # 挥手
            'pinch',       # 捏合
            'none'         # 无手势
        ])

        # 手势描述
        self.gesture_descriptions = {
            'fist': '握拳',
            'open_palm': '张开手掌',
            'thumbs_up': '竖大拇指',
            'thumbs_down': '拇指向下',
            'peace': '比耶',
            'ok': 'OK 手势',
            'pointing': '指向',
            'wave': '挥手',
            'pinch': '捏合',
            'none': '无手势'
        }

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
        model = GestureClassifier(
            num_classes=self.num_classes,
            pretrained=False
        )

        # 加载预训练权重
        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                print(f"加载手势分类模型: {model_path}")
            except Exception as e:
                print(f"加载模型失败: {e}")
                print("使用随机初始化的模型")
        else:
            print("未指定模型路径，使用随机初始化的模型")

        model = model.to(self.device)
        model.eval()

        return model

    def preprocess(self, hand_image: np.ndarray) -> torch.Tensor:
        """
        预处理手部图像

        Args:
            hand_image: 手部区域图像 (BGR格式)

        Returns:
            预处理后的张量
        """
        # 转换为 RGB
        rgb_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)

        # 转换为 PIL 图像
        pil_image = Image.fromarray(rgb_image)

        # 应用变换
        tensor = self.transform(pil_image)

        # 添加批次维度
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def predict(self, hand_image: np.ndarray) -> Dict:
        """
        预测手势

        Args:
            hand_image: 手部区域图像

        Returns:
            预测结果字典:
            - gesture: 手势名称
            - gesture_cn: 手势中文名称
            - confidence: 置信度
            - probabilities: 各类别概率
        """
        # 预处理
        input_tensor = self.preprocess(hand_image)

        # 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # 获取结果
        gesture = self.classes[predicted.item()]
        gesture_cn = self.gesture_descriptions.get(gesture, gesture)
        confidence = confidence.item()
        probs = probabilities[0].cpu().numpy()

        # 构建概率字典
        prob_dict = {}
        for i, cls in enumerate(self.classes):
            prob_dict[cls] = float(probs[i])

        return {
            'gesture': gesture,
            'gesture_cn': gesture_cn,
            'confidence': confidence,
            'probabilities': prob_dict
        }

    def predict_batch(self, hand_images: List[np.ndarray]) -> List[Dict]:
        """
        批量预测手势

        Args:
            hand_images: 手部图像列表

        Returns:
            预测结果列表
        """
        results = []

        for hand_image in hand_images:
            result = self.predict(hand_image)
            results.append(result)

        return results

    def get_gesture_command(self, gesture: str) -> Optional[str]:
        """
        将手势映射为控制命令

        Args:
            gesture: 手势名称

        Returns:
            控制命令，如果没有对应的命令则返回 None
        """
        gesture_commands = {
            'fist': 'stop',           # 握拳 - 停止
            'open_palm': 'start',     # 张开手掌 - 开始
            'thumbs_up': 'confirm',   # 竖大拇指 - 确认
            'thumbs_down': 'cancel',  # 拇指向下 - 取消
            'peace': 'peace',         # 比耶 - 和平
            'ok': 'ok',               # OK - 确认
            'pointing': 'select',     # 指向 - 选择
            'wave': 'greeting',       # 挥手 - 打招呼
            'pinch': 'zoom',          # 捏合 - 缩放
            'none': None              # 无手势
        }

        return gesture_commands.get(gesture)

    def draw_result(self, image: np.ndarray, bbox: List[int],
                   result: Dict, show_probabilities: bool = False) -> np.ndarray:
        """
        在图像上绘制检测结果

        Args:
            image: 输入图像
            bbox: 手部边界框 [x1, y1, x2, y2]
            result: 检测结果
            show_probabilities: 是否显示概率分布

        Returns:
            绘制了结果的图像
        """
        output_image = image.copy()
        x1, y1, x2, y2 = bbox

        # 绘制边界框
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

        # 绘制手势名称和置信度
        label = f"{result['gesture_cn']}: {result['confidence']:.2f}"
        cv2.putText(output_image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 绘制控制命令
        command = self.get_gesture_command(result['gesture'])
        if command:
            cv2.putText(output_image, f"Command: {command}", (x1, y2 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 显示概率分布
        if show_probabilities:
            y_offset = y2 + 50
            sorted_probs = sorted(result['probabilities'].items(),
                                 key=lambda x: x[1], reverse=True)[:5]

            for gesture, prob in sorted_probs:
                gesture_cn = self.gesture_descriptions.get(gesture, gesture)
                text = f"{gesture_cn}: {prob:.2f}"
                cv2.putText(output_image, text, (x1, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

        return output_image


class GestureRecognizerFactory:
    """手势识别器工厂类"""

    @staticmethod
    def create_recognizer(config: Dict) -> GestureRecognizer:
        """创建手势识别器"""
        return GestureRecognizer(config)


# 便捷函数
def create_gesture_recognizer(**kwargs) -> GestureRecognizer:
    """
    创建手势识别器的便捷函数

    Args:
        **kwargs: 配置参数

    Returns:
        手势识别器实例
    """
    config = {
        'model_path': kwargs.get('model_path', None),
        'num_classes': kwargs.get('num_classes', 10),
        'input_size': kwargs.get('input_size', [224, 224]),
        'confidence_threshold': kwargs.get('confidence_threshold', 0.7),
        'device': kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        'gesture_classes': kwargs.get('gesture_classes', [
            'fist', 'open_palm', 'thumbs_up', 'thumbs_down', 'peace',
            'ok', 'pointing', 'wave', 'pinch', 'none'
        ])
    }

    return GestureRecognizer(config)


if __name__ == "__main__":
    # 测试代码
    import sys

    # 创建识别器
    recognizer = create_gesture_recognizer()

    # 读取图像
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "data/samples/test_gesture.jpg"

    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        sys.exit(1)

    # 预测手势
    result = recognizer.predict(image)
    print(f"手势: {result['gesture_cn']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"概率分布: {result['probabilities']}")

    # 获取控制命令
    command = recognizer.get_gesture_command(result['gesture'])
    if command:
        print(f"控制命令: {command}")

    # 绘制结果
    bbox = [50, 50, 200, 200]  # 示例边界框
    result_image = recognizer.draw_result(image, bbox, result, show_probabilities=True)

    # 显示结果
    cv2.imshow("Gesture Recognition", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
