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

## 使用 PyCharm 运行项目

### 1. 打开项目

1. 打开 PyCharm
2. 选择 `File → Open`
3. 选择项目文件夹 `Driver_Status_and_Gesture_Monitoring_System`
4. 点击 `OK`

### 2. 配置 Python 解释器

1. 选择 `File → Settings → Project → Python Interpreter`
2. 点击右上角齿轮图标 → `Add`
3. 选择 `Existing environment`
4. 解释器路径选择项目下的虚拟环境：
   - Windows: `venv\Scripts\python.exe`
   - Linux/Mac: `venv/bin/python`
5. 点击 `OK` 确认

### 3. 安装依赖（PyCharm 内）

1. 打开 PyCharm 底部的 `Terminal`（终端）
2. 运行：
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

### 4. 运行脚本

#### 方法一：右键运行

1. 在左侧项目文件树中找到要运行的脚本，例如 `inference/demo.py`
2. 右键点击 → `Run 'demo'`

#### 方法二：配置运行参数

1. 点击右上角的运行配置下拉框 → `Edit Configurations`
2. 点击 `+` → `Python`
3. 配置：
   - **Name**: `摄像头监控`
   - **Script path**: `inference/demo.py`
   - **Parameters**: `--camera`
   - **Working directory**: 项目根目录
4. 点击 `OK`
5. 点击绿色三角 ▶ 运行

#### 常用运行配置

| 名称 | Script path | Parameters |
|---|---|---|
| 摄像头监控 | `inference/demo.py` | `--camera` |
| Web 界面 | `webapp/app.py` | （留空） |
| 训练眼睛模型 | `training/train_eye_state.py` | `--demo --epochs 5` |
| 训练嘴巴模型 | `training/train_mouth_state.py` | `--demo --epochs 5` |
| 训练手势模型 | `training/train_gesture.py` | `--demo --epochs 10` |
| 生成演示数据 | `tools/generate_demo_data.py` | `--task all` |
| 模型评估 | `tools/evaluate_model.py` | `--task eye_state --demo` |

### 5. 调试代码

1. 在代码行号左侧点击设置断点（红色圆点）
2. 右键脚本 → `Debug '脚本名'`
3. 使用调试工具栏：
   - **Step Over** (F8): 执行当前行
   - **Step Into** (F7): 进入函数内部
   - **Resume** (F9): 继续运行到下一个断点
   - **Stop** (Ctrl+F2): 停止调试

### 6. PyCharm 终端运行

也可以直接在 PyCharm 的 Terminal 中运行命令：

```bash
# 激活虚拟环境（PyCharm 通常自动激活）
.\venv\Scripts\Activate.ps1

# 运行训练
python training/train_eye_state.py --demo --epochs 5

# 启动 Web 服务
python webapp/app.py
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
