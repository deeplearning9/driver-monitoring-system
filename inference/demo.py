"""
演示脚本
展示系统功能的快速演示
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.realtime_monitor import DriverMonitor


def demo_camera():
    """摄像头演示"""
    print("=" * 50)
    print("🚗 驾驶员监控系统 - 摄像头演示")
    print("=" * 50)
    print()
    print("功能说明:")
    print("  - 实时人脸检测")
    print("  - 眼睛状态识别（睁眼/闭眼）")
    print("  - 嘴巴状态识别（张嘴/闭嘴）")
    print("  - 手势识别")
    print("  - 疲劳驾驶检测")
    print()
    print("操作说明:")
    print("  - 按 'q' 退出演示")
    print("  - 按 's' 保存当前帧")
    print("  - 按 'r' 重置状态")
    print()

    # 创建监控系统
    print("正在初始化监控系统...")
    monitor = DriverMonitor()
    print()

    # 启动摄像头
    print("正在启动摄像头...")
    monitor.start_camera(camera_id=0, display=True, save_video=False)


def demo_video(video_path: str):
    """视频演示"""
    print("=" * 50)
    print("🚗 驾驶员监控系统 - 视频演示")
    print("=" * 50)
    print()
    print(f"视频文件: {video_path}")
    print()

    # 检查文件是否存在
    if not Path(video_path).exists():
        print(f"错误: 视频文件不存在 - {video_path}")
        return

    # 创建监控系统
    print("正在初始化监控系统...")
    monitor = DriverMonitor()
    print()

    # 处理视频
    output_path = video_path.replace('.mp4', '_output.mp4')
    print(f"处理后的视频将保存到: {output_path}")
    print()

    monitor.process_video(
        input_path=video_path,
        output_path=output_path,
        display=True
    )


def demo_image(image_path: str):
    """图像演示"""
    print("=" * 50)
    print("🚗 驾驶员监控系统 - 图像演示")
    print("=" * 50)
    print()
    print(f"图像文件: {image_path}")
    print()

    # 检查文件是否存在
    if not Path(image_path).exists():
        print(f"错误: 图像文件不存在 - {image_path}")
        return

    # 创建监控系统
    print("正在初始化监控系统...")
    monitor = DriverMonitor()
    print()

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 - {image_path}")
        return

    # 处理图像
    print("正在处理图像...")
    results = monitor.process_frame(image)

    # 绘制结果
    output_image = monitor.draw_results(image, results)

    # 显示结果
    print()
    print("检测结果:")
    print(f"  - 检测到人脸: {len(results.get('faces', []))} 个")
    print(f"  - 检测到手部: {len(results.get('hands', []))} 个")
    print(f"  - 驾驶员状态: {results.get('driver_state', {}).get('overall_state', '未知')}")
    print()

    # 保存结果
    output_path = image_path.replace('.', '_result.')
    cv2.imwrite(output_path, output_image)
    print(f"结果已保存到: {output_path}")

    # 显示图像
    cv2.imshow("Detection Result", output_image)
    print()
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demo_web():
    """Web 演示"""
    print("=" * 50)
    print("🚗 驾驶员监控系统 - Web 演示")
    print("=" * 50)
    print()
    print("正在启动 Web 服务...")
    print()
    print("访问地址:")
    print("  - 主页: http://localhost:8000")
    print("  - API 文档: http://localhost:8000/docs")
    print()
    print("按 Ctrl+C 停止服务")
    print()

    # 导入并运行 Web 应用
    from webapp.app import main
    main()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="驾驶员监控系统演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python demo.py --camera          # 摄像头演示
  python demo.py --video input.mp4 # 视频演示
  python demo.py --image test.jpg  # 图像演示
  python demo.py --web             # Web 演示
        """
    )

    parser.add_argument('--camera', action='store_true', help='摄像头演示')
    parser.add_argument('--video', type=str, help='视频演示')
    parser.add_argument('--image', type=str, help='图像演示')
    parser.add_argument('--web', action='store_true', help='Web 演示')

    args = parser.parse_args()

    # 如果没有指定参数，显示帮助
    if not any([args.camera, args.video, args.image, args.web]):
        parser.print_help()
        print()
        print("请选择一个演示模式:")
        print("  1. 摄像头实时演示")
        print("  2. 视频文件演示")
        print("  3. 图像检测演示")
        print("  4. Web 界面演示")
        print()

        choice = input("请输入选项 (1-4): ").strip()

        if choice == '1':
            demo_camera()
        elif choice == '2':
            video_path = input("请输入视频文件路径: ").strip()
            demo_video(video_path)
        elif choice == '3':
            image_path = input("请输入图像文件路径: ").strip()
            demo_image(image_path)
        elif choice == '4':
            demo_web()
        else:
            print("无效的选项")
            return
    else:
        if args.camera:
            demo_camera()
        elif args.video:
            demo_video(args.video)
        elif args.image:
            demo_image(args.image)
        elif args.web:
            demo_web()


if __name__ == "__main__":
    main()
