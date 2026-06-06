"""
实时监控模块
整合所有检测和分类模块，实现实时驾驶员监控
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional
from collections import deque
import threading
import queue

from models.detection.face_detector import create_face_detector
from models.detection.hand_detector import create_hand_detector
from models.classification.eye_state import create_eye_state_detector
from models.classification.mouth_state import create_mouth_state_detector
from models.classification.gesture_classifier import create_gesture_recognizer
from utils.visualization import create_visualizer


class DriverMonitor:
    """驾驶员监控系统"""

    def __init__(self, config: Dict = None):
        """
        初始化监控系统

        Args:
            config: 配置字典
        """
        self.config = config or {}

        # 初始化各个模块
        print("正在初始化监控系统...")

        # 人脸检测器
        self.face_detector = create_face_detector(
            model_type=self.config.get('face_model', 'mediapipe'),
            confidence_threshold=0.5
        )
        print("✓ 人脸检测器已加载")

        # 手部检测器
        self.hand_detector = create_hand_detector(
            model_type=self.config.get('hand_model', 'mediapipe'),
            confidence_threshold=0.7
        )
        print("✓ 手部检测器已加载")

        # 眼睛状态检测器
        self.eye_detector = create_eye_state_detector()
        print("✓ 眼睛状态检测器已加载")

        # 嘴巴状态检测器
        self.mouth_detector = create_mouth_state_detector()
        print("✓ 嘴巴状态检测器已加载")

        # 手势识别器
        self.gesture_recognizer = create_gesture_recognizer()
        print("✓ 手势识别器已加载")

        # 可视化工具
        self.visualizer = create_visualizer()
        print("✓ 可视化工具已加载")

        # 状态历史记录
        self.eye_state_history = deque(maxlen=30)
        self.mouth_state_history = deque(maxlen=30)
        self.gesture_history = deque(maxlen=10)

        # FPS 计算
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        # 报警状态
        self.alert_active = False
        self.alert_message = ""
        self.alert_severity = "info"

        # 疲劳检测参数
        self.fatigue_threshold = self.config.get('fatigue_threshold', 0.6)
        self.distraction_threshold = self.config.get('distraction_threshold', 0.5)

        print("监控系统初始化完成！")

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        处理单帧图像

        Args:
            frame: 输入图像 (BGR格式)

        Returns:
            处理结果字典
        """
        results = {
            'faces': [],
            'hands': [],
            'eyes': [],
            'mouths': [],
            'driver_state': {},
            'fps': 0
        }

        # 人脸检测
        face_detections = self.face_detector.detect(frame)
        results['faces'] = face_detections

        # 手部检测
        hand_detections = self.hand_detector.detect(frame)
        results['hands'] = hand_detections

        # 处理每张检测到的人脸
        for face in face_detections:
            x1, y1, x2, y2 = face['bbox']
            face_image = frame[y1:y2, x1:x2]

            if face_image.size == 0:
                continue

            # 眼睛状态检测（简化版：假设眼睛在脸上半部分）
            h, w = face_image.shape[:2]
            eye_region = face_image[0:int(h*0.4), :]

            if eye_region.size > 0:
                try:
                    eye_state = self.eye_detector.predict(eye_region)
                    self.eye_state_history.append(eye_state)
                    results['eyes'].append(([x1, y1, x1+w, y1+int(h*0.4)], eye_state))
                except Exception as e:
                    pass

            # 嘴巴状态检测（简化版：假设嘴巴在脸下半部分）
            mouth_region = face_image[int(h*0.6):h, :]

            if mouth_region.size > 0:
                try:
                    mouth_state = self.mouth_detector.predict(mouth_region)
                    self.mouth_state_history.append(mouth_state)
                    results['mouths'].append(([x1, y1+int(h*0.6), x1+w, y1+h], mouth_state))
                except Exception as e:
                    pass

        # 处理每只检测到的手
        for hand in hand_detections:
            x1, y1, x2, y2 = hand['bbox']
            hand_image = frame[y1:y2, x1:x2]

            if hand_image.size == 0:
                continue

            try:
                gesture_result = self.gesture_recognizer.predict(hand_image)
                self.gesture_history.append(gesture_result)
                hand['gesture'] = gesture_result
            except Exception as e:
                pass

        # 计算驾驶员状态
        driver_state = self._analyze_driver_state()
        results['driver_state'] = driver_state

        # 更新 FPS
        self._update_fps()
        results['fps'] = self.fps

        return results

    def _analyze_driver_state(self) -> Dict:
        """
        分析驾驶员状态

        Returns:
            驾驶员状态字典
        """
        state = {
            'fatigue_score': 0.0,
            'distraction_score': 0.0,
            'yawning': False,
            'eyes_closed': False,
            'overall_state': 'normal'
        }

        # 分析眼睛状态
        if len(self.eye_state_history) > 0:
            recent_eye_states = list(self.eye_state_history)[-10:]
            closed_count = sum(1 for s in recent_eye_states if s['state'] == 'closed')
            state['eyes_closed'] = closed_count > len(recent_eye_states) * 0.5
            state['fatigue_score'] = closed_count / len(recent_eye_states)

        # 分析嘴巴状态
        if len(self.mouth_state_history) > 0:
            recent_mouth_states = list(self.mouth_state_history)[-10:]
            open_count = sum(1 for s in recent_mouth_states if s['state'] == 'open')
            state['yawning'] = open_count > len(recent_mouth_states) * 0.6

        # 综合判断
        if state['fatigue_score'] > self.fatigue_threshold:
            state['overall_state'] = 'fatigued'
            self._trigger_alert("疲劳驾驶警告！", "danger")
        elif state['yawning']:
            state['overall_state'] = 'yawning'
            self._trigger_alert("检测到打哈欠", "warning")
        else:
            state['overall_state'] = 'normal'
            self._clear_alert()

        return state

    def _trigger_alert(self, message: str, severity: str) -> None:
        """触发报警"""
        self.alert_active = True
        self.alert_message = message
        self.alert_severity = severity

    def _clear_alert(self) -> None:
        """清除报警"""
        self.alert_active = False
        self.alert_message = ""
        self.alert_severity = "info"

    def _update_fps(self) -> None:
        """更新 FPS"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time

        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

    def draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        在图像上绘制检测结果

        Args:
            frame: 输入图像
            results: 检测结果

        Returns:
            绘制后的图像
        """
        output = frame.copy()

        # 绘制人脸检测
        output = self.visualizer.draw_face_detection(output, results['faces'])

        # 绘制眼睛状态
        output = self.visualizer.draw_eye_state(output, results['eyes'])

        # 绘制嘴巴状态
        output = self.visualizer.draw_mouth_state(output, results['mouths'])

        # 绘制手势
        for hand in results['hands']:
            if 'gesture' in hand:
                output = self.visualizer.draw_gesture(output, hand['bbox'], hand['gesture'])

        # 绘制驾驶员状态
        output = self.visualizer.draw_driver_state(output, results['driver_state'])

        # 绘制 FPS
        output = self.visualizer.draw_fps(output, results['fps'])

        # 绘制报警
        if self.alert_active:
            output = self.visualizer.draw_alert(output, self.alert_message, self.alert_severity)

        return output

    def start_camera(self, camera_id: int = 0,
                    display: bool = True,
                    save_video: bool = False,
                    output_path: str = "output.mp4") -> None:
        """
        启动摄像头监控

        Args:
            camera_id: 摄像头ID
            display: 是否显示画面
            save_video: 是否保存视频
            output_path: 视频保存路径
        """
        print(f"正在打开摄像头 {camera_id}...")

        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            return

        print("摄像头已打开，按 'q' 退出")

        # 视频写入器
        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("无法读取视频帧")
                    break

                # 处理帧
                results = self.process_frame(frame)

                # 绘制结果
                output_frame = self.draw_results(frame, results)

                # 显示画面
                if display:
                    cv2.imshow("Driver Monitoring System", output_frame)

                # 保存视频
                if writer:
                    writer.write(output_frame)

                # 按 'q' 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("用户退出")
                    break

        except KeyboardInterrupt:
            print("用户中断")
        finally:
            # 释放资源
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print("监控已停止")

    def process_video(self, input_path: str,
                     output_path: str = None,
                     display: bool = True) -> None:
        """
        处理视频文件

        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            display: 是否显示画面
        """
        print(f"正在处理视频: {input_path}")

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print(f"无法打开视频: {input_path}")
            return

        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")

        # 视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                # 处理帧
                results = self.process_frame(frame)

                # 绘制结果
                output_frame = self.draw_results(frame, results)

                # 显示进度
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"处理进度: {progress:.1f}% ({frame_count}/{total_frames})")

                # 显示画面
                if display:
                    cv2.imshow("Video Processing", output_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("用户中断")
                        break

                # 保存视频
                if writer:
                    writer.write(output_frame)

        except KeyboardInterrupt:
            print("用户中断")
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print(f"视频处理完成，共处理 {frame_count} 帧")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="驾驶员监控系统")
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID')
    parser.add_argument('--video', type=str, help='视频文件路径')
    parser.add_argument('--output', type=str, help='输出视频路径')
    parser.add_argument('--no-display', action='store_true', help='不显示画面')

    args = parser.parse_args()

    # 创建监控系统
    monitor = DriverMonitor()

    if args.video:
        # 处理视频文件
        monitor.process_video(
            input_path=args.video,
            output_path=args.output,
            display=not args.no_display
        )
    else:
        # 启动摄像头监控
        monitor.start_camera(
            camera_id=args.camera,
            display=not args.no_display,
            save_video=bool(args.output),
            output_path=args.output or "output.mp4"
        )


if __name__ == "__main__":
    main()
