"""
演示数据生成器
生成合成数据集，用于快速验证训练流水线

学习笔记：
这个脚本生成简单的合成图片来测试训练流程。
合成数据不是用来训练实际可用的模型，而是用来验证：
1. 数据加载是否正常
2. 训练循环是否能跑通
3. 模型权重是否正确保存

实际训练需要使用真实数据集（见 tools/download_dataset.py）
"""

import os
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate_eye_image(is_open: bool, size: int = 224) -> Image.Image:
    """
    生成模拟眼睛图片

    Args:
        is_open: True 生成睁眼图片，False 生成闭眼图片
        size: 图片尺寸

    Returns:
        PIL Image 对象
    """
    # 创建背景
    img = Image.new('RGB', (size, size), color=(
        random.randint(180, 220),
        random.randint(150, 190),
        random.randint(130, 170)
    ))
    draw = ImageDraw.Draw(img)

    # 眼睛区域
    cx, cy = size // 2, size // 2
    eye_w = random.randint(60, 90)
    eye_h = random.randint(30, 50)

    # 绘制眼睛轮廓（椭圆）
    draw.ellipse(
        [cx - eye_w, cy - eye_h, cx + eye_w, cy + eye_h],
        outline=(80, 60, 40), width=3
    )

    if is_open:
        # 睁眼：绘制虹膜和瞳孔
        iris_r = random.randint(18, 28)
        iris_color = (random.randint(40, 100), random.randint(80, 140),
                      random.randint(30, 80))
        draw.ellipse(
            [cx - iris_r, cy - iris_r, cx + iris_r, cy + iris_r],
            fill=iris_color
        )
        # 瞳孔
        pupil_r = iris_r // 2
        draw.ellipse(
            [cx - pupil_r, cy - pupil_r, cx + pupil_r, cy + pupil_r],
            fill=(20, 20, 20)
        )
        # 高光
        highlight_r = pupil_r // 2
        draw.ellipse(
            [cx - highlight_r - 5, cy - highlight_r - 5,
             cx + highlight_r - 5, cy + highlight_r - 5],
            fill=(220, 220, 220)
        )
    else:
        # 闭眼：绘制一条线
        line_y = cy
        draw.line(
            [(cx - eye_w + 10, line_y), (cx + eye_w - 10, line_y)],
            fill=(60, 40, 30), width=4
        )
        # 睫毛
        for i in range(-eye_w + 20, eye_w - 20, 15):
            draw.line(
                [(cx + i, line_y - 3), (cx + i - 3, line_y - 10)],
                fill=(50, 30, 20), width=2
            )

    # 添加随机噪声
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    # 随机旋转
    angle = random.uniform(-10, 10)
    img = img.rotate(angle, fillcolor=(200, 170, 150))

    # 随机亮度调整
    factor = random.uniform(0.8, 1.2)
    img = Image.fromarray(np.clip(np.array(img) * factor, 0, 255).astype(np.uint8))

    return img


def generate_mouth_image(is_open: bool, size: int = 224) -> Image.Image:
    """
    生成模拟嘴巴图片

    Args:
        is_open: True 生成张嘴图片，False 生成闭嘴图片
        size: 图片尺寸

    Returns:
        PIL Image 对象
    """
    img = Image.new('RGB', (size, size), color=(
        random.randint(180, 220),
        random.randint(150, 190),
        random.randint(130, 170)
    ))
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2
    mouth_w = random.randint(50, 80)

    if is_open:
        # 张嘴：绘制椭圆
        mouth_h = random.randint(30, 50)
        draw.ellipse(
            [cx - mouth_w, cy - mouth_h, cx + mouth_w, cy + mouth_h],
            fill=(150, 50, 50), outline=(100, 30, 30), width=2
        )
        # 舌头
        tongue_h = mouth_h // 2
        draw.ellipse(
            [cx - mouth_w // 2, cy, cx + mouth_w // 2, cy + tongue_h],
            fill=(200, 100, 100)
        )
    else:
        # 闭嘴：绘制一条线
        draw.line(
            [(cx - mouth_w, cy), (cx + mouth_w, cy)],
            fill=(150, 80, 80), width=4
        )
        # 嘴唇
        draw.arc(
            [cx - mouth_w, cy - 10, cx + mouth_w, cy + 10],
            0, 180, fill=(180, 100, 100), width=3
        )

    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    angle = random.uniform(-5, 5)
    img = img.rotate(angle, fillcolor=(200, 170, 150))

    return img


def generate_gesture_image(gesture_type: str, size: int = 224) -> Image.Image:
    """
    生成模拟手势图片

    Args:
        gesture_type: 手势类型
        size: 图片尺寸

    Returns:
        PIL Image 对象
    """
    img = Image.new('RGB', (size, size), color=(
        random.randint(200, 240),
        random.randint(200, 240),
        random.randint(200, 240)
    ))
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2
    color = (random.randint(50, 150), random.randint(50, 150),
             random.randint(50, 150))

    if gesture_type == 'fist':
        # 握拳：绘制圆形
        r = random.randint(40, 60)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color,
                     outline=(30, 30, 30), width=3)
    elif gesture_type == 'open_palm':
        # 张开手掌：绘制五边形
        r = random.randint(45, 65)
        points = []
        for i in range(5):
            angle = i * 72 - 90
            px = cx + r * np.cos(np.radians(angle))
            py = cy + r * np.sin(np.radians(angle))
            points.append((px, py))
        draw.polygon(points, fill=color, outline=(30, 30, 30))
    elif gesture_type == 'thumbs_up':
        # 竖大拇指：绘制椭圆 + 竖线
        draw.ellipse([cx - 30, cy - 20, cx + 30, cy + 40], fill=color)
        draw.rectangle([cx - 8, cy - 70, cx + 8, cy - 10], fill=color)
    elif gesture_type == 'thumbs_down':
        # 拇指向下：倒置
        draw.ellipse([cx - 30, cy - 40, cx + 30, cy + 20], fill=color)
        draw.rectangle([cx - 8, cy + 10, cx + 8, cy + 70], fill=color)
    elif gesture_type == 'peace':
        # 比耶：绘制 V 形
        draw.line([(cx - 30, cy + 40), (cx - 10, cy - 50)], fill=color, width=8)
        draw.line([(cx + 10, cy - 50), (cx + 30, cy + 40)], fill=color, width=8)
        draw.ellipse([cx - 35, cy + 20, cx + 35, cy + 55], fill=color)
    elif gesture_type == 'ok':
        # OK 手势：圆形 + 线
        r = 30
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=6)
        draw.line([(cx + r, cy), (cx + r + 40, cy + 30)], fill=color, width=6)
    elif gesture_type == 'pointing':
        # 指向：竖线 + 圆
        draw.rectangle([cx - 5, cy - 60, cx + 5, cy + 10], fill=color)
        draw.ellipse([cx - 25, cy + 10, cx + 25, cy + 45], fill=color)
    elif gesture_type == 'wave':
        # 挥手：波浪线
        points = []
        for x in range(cx - 60, cx + 60, 5):
            y = cy + int(20 * np.sin((x - cx) * 0.1))
            points.append((x, y))
        draw.line(points, fill=color, width=6)
        draw.ellipse([cx - 20, cy - 40, cx + 20, cy], fill=color)
    elif gesture_type == 'pinch':
        # 捏合：两个小圆靠近
        draw.ellipse([cx - 35, cy - 20, cx - 5, cy + 20], fill=color)
        draw.ellipse([cx + 5, cy - 20, cx + 35, cy + 20], fill=color)
    else:  # none
        # 无手势：绘制随机形状
        draw.rectangle([cx - 40, cy - 40, cx + 40, cy + 40], fill=color)

    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, fillcolor=(220, 220, 220))

    return img


def generate_dataset(
    output_dir: str,
    task: str = 'eye_state',
    train_per_class: int = 150,
    val_per_class: int = 30,
    test_per_class: int = 20
):
    """
    生成完整的演示数据集

    Args:
        output_dir: 输出目录
        task: 任务类型 ('eye_state', 'mouth_state', 'gesture')
        train_per_class: 每类训练样本数
        val_per_class: 每类验证样本数
        test_per_class: 每类测试样本数
    """
    # 定义各类别的生成函数和标签
    if task == 'eye_state':
        classes = ['closed', 'open']
        generator = generate_eye_image
    elif task == 'mouth_state':
        classes = ['closed', 'open']
        generator = generate_mouth_image
    elif task == 'gesture':
        classes = ['fist', 'open_palm', 'thumbs_up', 'thumbs_down',
                   'peace', 'ok', 'pointing', 'wave', 'pinch', 'none']
        generator = generate_gesture_image
    else:
        raise ValueError(f"不支持的任务: {task}")

    splits = {
        'train': train_per_class,
        'val': val_per_class,
        'test': test_per_class,
    }

    print(f"\n正在生成 {task} 演示数据集...")
    print(f"  类别: {classes}")
    print(f"  每类样本: 训练={train_per_class}, 验证={val_per_class}, 测试={test_per_class}")

    total_generated = 0

    for split, count in splits.items():
        for cls in classes:
            # 创建目录
            cls_dir = os.path.join(output_dir, task, split, cls)
            os.makedirs(cls_dir, exist_ok=True)

            # 生成图片
            for i in range(count):
                if task == 'gesture':
                    img = generator(cls)
                else:
                    img = generator(is_open=(cls == 'open'))

                # 保存
                img_path = os.path.join(cls_dir, f"{cls}_{i:04d}.png")
                img.save(img_path)
                total_generated += 1

    print(f"  生成完成！共 {total_generated} 张图片")
    print(f"  保存目录: {os.path.join(output_dir, task)}")


def main():
    parser = argparse.ArgumentParser(description="生成演示数据集")
    parser.add_argument('--output', type=str, default='data/demo',
                        help='输出目录')
    parser.add_argument('--task', type=str, default='all',
                        choices=['eye_state', 'mouth_state', 'gesture', 'all'],
                        help='任务类型')
    parser.add_argument('--train', type=int, default=150,
                        help='每类训练样本数')
    parser.add_argument('--val', type=int, default=30,
                        help='每类验证样本数')
    parser.add_argument('--test', type=int, default=20,
                        help='每类测试样本数')

    args = parser.parse_args()

    print("=" * 50)
    print("演示数据集生成器")
    print("=" * 50)

    tasks = ['eye_state', 'mouth_state', 'gesture'] if args.task == 'all' \
        else [args.task]

    for task in tasks:
        generate_dataset(
            output_dir=args.output,
            task=task,
            train_per_class=args.train,
            val_per_class=args.val,
            test_per_class=args.test
        )

    print("\n所有数据集生成完成！")
    print(f"输出目录: {args.output}")


if __name__ == '__main__':
    main()
