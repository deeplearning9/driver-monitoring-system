# 使用说明

## 环境要求

- Python 3.10+
- Windows 10/11 或 Linux
- 摄像头（实时监控模式）

## 安装

```bash
# 克隆仓库
git clone https://github.com/deeplearning9/driver-monitoring-system.git
cd driver-monitoring-system

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install -e .
```

## 快速开始

### 1. 摄像头实时监控

```bash
python inference/demo.py --camera
```

按 `q` 退出。

### 2. 处理视频文件

```bash
python inference/demo.py --video input.mp4
```

### 3. Web 界面

```bash
python webapp/app.py
```

浏览器打开 http://localhost:8000

## 微调训练

### 生成演示数据

```bash
python tools/generate_demo_data.py --task all
```

### 训练眼睛状态模型

```bash
# 使用合成数据快速验证
python training/train_eye_state.py --demo --epochs 5

# 使用真实数据集
python training/train_eye_state.py --data-dir data/raw/eye_state --epochs 20
```

### 训练嘴巴状态模型

```bash
python training/train_mouth_state.py --demo --epochs 5
```

### 训练手势识别模型

```bash
python training/train_gesture.py --demo --epochs 10
```

### 评估模型

```bash
python tools/evaluate_model.py --task eye_state --demo
```

## 使用训练好的模型

训练完成后，权重保存在 `models/weights/` 目录。

修改 `configs/model_config.yaml` 指向训练好的权重：

```yaml
eye_state:
  model_path: "models/weights/eye_state_resnet_best.pth"

mouth_state:
  model_path: "models/weights/mouth_state_resnet_best.pth"

gesture_classification:
  model_path: "models/weights/gesture_efficientnet_best.pth"
```

## 训练参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--epochs` | 训练轮数 | 配置文件中的值 |
| `--batch-size` | 批次大小 | 32 |
| `--lr` | 学习率 | 0.0001 |
| `--demo` | 使用合成数据 | False |
| `--device` | 设备 (cpu/cuda) | 自动检测 |
| `--save-dir` | 权重保存目录 | models/weights |

## 项目结构

```
├── configs/                # 配置文件
│   ├── model_config.yaml   # 模型配置
│   └── training_config.yaml # 训练配置
├── inference/              # 推理模块
│   ├── demo.py             # 演示脚本
│   └── realtime_monitor.py # 实时监控
├── models/                 # 模型定义
│   ├── classification/     # 分类模型
│   └── detection/          # 检测模型
├── training/               # 训练模块
│   ├── engine.py           # 训练引擎
│   ├── data_utils.py       # 数据工具
│   ├── callbacks.py        # 回调函数
│   ├── train_eye_state.py  # 眼睛状态训练
│   ├── train_mouth_state.py# 嘴巴状态训练
│   └── train_gesture.py    # 手势识别训练
├── tools/                  # 工具脚本
│   ├── generate_demo_data.py # 生成演示数据
│   └── evaluate_model.py   # 模型评估
├── utils/                  # 工具函数
└── webapp/                 # Web 界面
```
