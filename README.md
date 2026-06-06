# 🚗 驾驶员状态与手势监控系统

基于深度学习的汽车座舱智能监控系统，实现驾驶员状态监控和手势识别功能。

## ✨ 功能特性

### 1. 驾驶员状态监控
- **人脸检测**：实时检测驾驶员面部位置
- **眼睛状态识别**：判断睁眼/闭眼状态，检测疲劳
- **嘴巴状态识别**：检测打哈欠行为
- **头部姿态估计**：分析头部方向，检测分心
- **注意力综合评估**：融合多模态信息，输出驾驶状态

### 2. 手势识别
- **手部检测**：实时检测手部区域
- **静态手势识别**：识别 OK、点赞、握拳等手势
- **动态手势识别**：识别挥手、指向等动作
- **手势控制映射**：将手势映射为车辆控制指令

### 3. 系统功能
- **实时视频处理**：支持摄像头和视频文件输入
- **Web 可视化界面**：直观展示检测结果
- **报警系统**：疲劳/分心状态自动报警
- **数据记录**：记录驾驶行为数据用于分析

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| **深度学习框架** | PyTorch |
| **计算机视觉** | OpenCV, MediaPipe |
| **目标检测** | YOLOv8 |
| **图像分类** | ResNet, EfficientNet |
| **行为识别** | LSTM, Transformer |
| **Web 框架** | FastAPI |
| **容器化** | Docker |

## 📦 安装

### 方式 1：直接安装

```bash
# 克隆仓库
git clone https://github.com/deeplearning9/driver-monitoring-system.git
cd driver-monitoring-system

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

### 方式 2：Docker 安装

```bash
# 构建镜像
docker build -t driver-monitoring .

# 运行容器
docker run -p 8000:8000 driver-monitoring
```

## 🚀 快速开始

### 1. 实时摄像头监控

```python
from inference.realtime_monitor import DriverMonitor

# 初始化监控系统
monitor = DriverMonitor()

# 启动摄像头监控
monitor.start_camera(camera_id=0)
```

### 2. 视频文件处理

```python
from inference.video_processor import VideoProcessor

# 初始化处理器
processor = VideoProcessor()

# 处理视频文件
processor.process_video(
    input_path="input.mp4",
    output_path="output.mp4"
)
```

### 3. Web 界面

```bash
# 启动 Web 服务
cd webapp
python app.py

# 访问 http://localhost:8000
```

## 📁 项目结构

```
driver-monitoring-system/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖包
├── setup.py                     # 安装脚本
├── Dockerfile                   # Docker 配置
│
├── configs/                     # 配置文件
│   ├── model_config.yaml
│   └── training_config.yaml
│
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据
│   ├── processed/               # 处理后数据
│   └── samples/                 # 示例数据
│
├── models/                      # 模型定义
│   ├── detection/               # 检测模型
│   │   ├── face_detector.py
│   │   └── hand_detector.py
│   ├── classification/          # 分类模型
│   │   ├── eye_state.py
│   │   ├── mouth_state.py
│   │   └── gesture_classifier.py
│   └── tracking/                # 跟踪模型
│       └── head_pose.py
│
├── utils/                       # 工具函数
│   ├── data_utils.py
│   ├── visualization.py
│   └── metrics.py
│
├── training/                    # 训练脚本
│   ├── train_detection.py
│   ├── train_classification.py
│   └── train_gesture.py
│
├── inference/                   # 推理脚本
│   ├── realtime_monitor.py
│   ├── video_processor.py
│   └── demo.py
│
├── webapp/                      # Web 界面
│   ├── app.py
│   ├── static/
│   └── templates/
│
├── notebooks/                   # Jupyter 笔记本
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── tests/                       # 测试代码
│   ├── test_models.py
│   └── test_utils.py
│
└── docs/                        # 文档
    ├── installation.md
    ├── usage.md
    └── api.md
```

## 📊 数据集

### 公开数据集
| 数据集 | 用途 | 规模 |
|--------|------|------|
| WIDER FACE | 人脸检测 | 32K 图片 |
| CelebA | 人脸属性 | 200K 图片 |
| CMU Hand Dataset | 手势识别 | 2.6K 图片 |
| Jester | 动态手势 | 150K 视频 |
| NTHU-DDD | 疲劳驾驶 | 360K 图片 |

### 自制数据集
使用 `utils/data_utils.py` 中的工具进行数据采集和标注。

## 🎯 模型性能

| 模型 | 任务 | mAP/准确率 | 速度(FPS) |
|------|------|------------|-----------|
| YOLOv8-face | 人脸检测 | 95.2% | 45 |
| ResNet-eye | 眼睛状态 | 97.8% | 120 |
| ResNet-mouth | 嘴巴状态 | 96.5% | 115 |
| MediaPipe-hand | 手部检测 | 93.8% | 60 |
| EfficientNet-gesture | 手势分类 | 94.2% | 80 |

## 📚 文档

- [安装指南](docs/installation.md)
- [使用说明](docs/usage.md)
- [API 文档](docs/api.md)
- [训练指南](docs/training.md)

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 👨‍💻 作者

**deeplearning9** - [GitHub](https://github.com/deeplearning9)

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [MediaPipe](https://github.com/google/mediapipe)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub Issues: [创建 Issue](https://github.com/deeplearning9/driver-monitoring-system/issues)
- Email: 1724022286@qq.com

---

⭐ 如果这个项目对你有帮助，请给个 Star 支持一下！
