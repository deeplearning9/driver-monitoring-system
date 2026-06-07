"""
Web 应用模块
提供 Web 界面用于监控和控制
"""

import cv2
import numpy as np
import base64
import json
import time
from typing import Dict, List, Optional
from pathlib import Path
import threading
import queue

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from inference.realtime_monitor import DriverMonitor


# 创建 FastAPI 应用
app = FastAPI(
    title="驾驶员监控系统",
    description="基于深度学习的汽车座舱智能监控系统",
    version="1.0.0"
)

# 静态文件和模板
static_dir = Path(__file__).parent / "static"
template_dir = Path(__file__).parent / "templates"

static_dir.mkdir(exist_ok=True)
template_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(template_dir))

# 全局监控系统实例
monitor = None
monitor_thread = None
frame_queue = queue.Queue(maxsize=10)
result_queue = queue.Queue(maxsize=10)


class WebSocketManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = WebSocketManager()


def init_monitor():
    """初始化监控系统"""
    global monitor
    if monitor is None:
        monitor = DriverMonitor()
    return monitor


def capture_frames(camera_id: int = 0):
    """捕获视频帧（在后台线程运行）"""
    global monitor, frame_queue, result_queue

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_id}")
        return

    print("开始捕获视频帧...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 放入帧队列
        if not frame_queue.full():
            frame_queue.put(frame)

        # 处理帧
        if monitor:
            results = monitor.process_frame(frame)
            output_frame = monitor.draw_results(frame, results)

            # 放入结果队列
            if not result_queue.full():
                result_queue.put((output_frame, results))

        time.sleep(0.01)  # 控制帧率

    cap.release()


@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    init_monitor()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """主页"""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/status")
async def get_status():
    """获取系统状态"""
    return {
        "status": "running",
        "monitor_initialized": monitor is not None,
        "timestamp": time.time()
    }


@app.get("/api/start/{camera_id}")
async def start_monitoring(camera_id: int = 0):
    """启动监控"""
    global monitor_thread

    if monitor_thread and monitor_thread.is_alive():
        return {"message": "监控已在运行中"}

    monitor_thread = threading.Thread(
        target=capture_frames,
        args=(camera_id,),
        daemon=True
    )
    monitor_thread.start()

    return {"message": f"已启动摄像头 {camera_id} 的监控"}


@app.get("/api/stop")
async def stop_monitoring():
    """停止监控"""
    global monitor_thread

    # 清空队列
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except:
            break

    while not result_queue.empty():
        try:
            result_queue.get_nowait()
        except:
            break

    return {"message": "已停止监控"}


@app.get("/api/results")
async def get_results():
    """获取最新结果"""
    if result_queue.empty():
        return {"message": "没有可用的结果"}

    frame, results = result_queue.get()

    # 将图像编码为 base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "frame": frame_base64,
        "results": {
            "faces_count": len(results.get('faces', [])),
            "hands_count": len(results.get('hands', [])),
            "driver_state": results.get('driver_state', {}),
            "fps": results.get('fps', 0)
        }
    }


@app.get("/video_feed")
async def video_feed():
    """视频流"""
    async def generate():
        while True:
            if not result_queue.empty():
                frame, _ = result_queue.get()
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n'
                       + frame_bytes + b'\r\n')

            await asyncio.sleep(0.03)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点"""
    await manager.connect(websocket)

    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            message = json.loads(data)

            # 处理命令
            if message.get("command") == "start":
                camera_id = message.get("camera_id", 0)
                await start_monitoring(camera_id)
                await websocket.send_json({"status": "started"})

            elif message.get("command") == "stop":
                await stop_monitoring()
                await websocket.send_json({"status": "stopped"})

            elif message.get("command") == "get_results":
                if not result_queue.empty():
                    frame, results = result_queue.get()
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')

                    await websocket.send_json({
                        "type": "results",
                        "frame": frame_base64,
                        "results": results
                    })

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/config")
async def get_config():
    """获取配置"""
    if monitor:
        return {
            "fatigue_threshold": monitor.fatigue_threshold,
            "distraction_threshold": monitor.distraction_threshold
        }
    return {}


@app.post("/api/config")
async def update_config(config: Dict):
    """更新配置"""
    global monitor

    if monitor:
        if "fatigue_threshold" in config:
            monitor.fatigue_threshold = config["fatigue_threshold"]
        if "distraction_threshold" in config:
            monitor.distraction_threshold = config["distraction_threshold"]

        return {"message": "配置已更新"}

    return {"message": "监控系统未初始化"}


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="驾驶员监控系统 Web 服务")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='监听地址')
    parser.add_argument('--port', type=int, default=8000, help='监听端口')
    parser.add_argument('--reload', action='store_true', help='自动重载')

    args = parser.parse_args()

    print(f"启动 Web 服务: http://{args.host}:{args.port}")
    print(f"API 文档: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    import asyncio
    main()
