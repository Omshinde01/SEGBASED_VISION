"""
kaggle_server.py - Runs on Kaggle Notebook with GPU acceleration.
Receives RGB + Depth frames from local client over WebSocket/HTTP,
runs Grounding DINO + SAM on CUDA, computes 6-DoF grasp poses,
supporting surface plane analysis, and collision-free placement spots,
and serves an interactive web dashboard with live video and JSON telemetry.
"""

import os
import io
import time
import json
import base64
import asyncio
from typing import Optional
import numpy as np
import cv2
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from perception_pipeline.pipeline import PerceptionPipeline
from perception_pipeline.camera.orbbec_camera import CameraIntrinsics

app = FastAPI(title="Kaggle 3D Perception & Pick-and-Place Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Optional[PerceptionPipeline] = None
current_prompt = "pick up the cup and place it near laptop"


@app.on_event("startup")
def startup_event():
    global pipeline
    print("[*] Initializing Perception & Pick-and-Place Pipeline on Kaggle GPU...")
    # Initialize pipeline with mock camera since frames will stream in over WebSocket
    pipeline = PerceptionPipeline(config_path="config.yaml", prefer_orbbec=False)
    print("[*] Models loaded and warm!")


HTML_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <title>Kaggle 3D Perception & Pick-and-Place</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #121214; color: #fff; margin: 0; padding: 20px; }
        .container { max-width: 1300px; margin: auto; }
        h1 { color: #00d2ff; margin-bottom: 5px; }
        .controls { margin-bottom: 15px; display: flex; gap: 10px; align-items: center; }
        input[type="text"] { padding: 10px 14px; border-radius: 6px; border: 1px solid #333; background: #1e1e24; color: #fff; font-size: 15px; width: 450px; }
        button { padding: 10px 20px; border-radius: 6px; border: none; background: #00d2ff; color: #000; font-weight: bold; cursor: pointer; font-size: 15px; }
        button:hover { background: #33e1ff; }
        .grid { display: grid; grid-template-columns: 1.8fr 1.2fr; gap: 20px; }
        .video-box { background: #000; border-radius: 8px; overflow: hidden; position: relative; border: 1px solid #2d2d38; }
        #live-img { width: 100%; height: auto; display: block; min-height: 480px; }
        .json-box { background: #1a1a22; border-radius: 8px; padding: 15px; overflow-y: auto; max-height: 520px; border: 1px solid #2d2d38; font-family: monospace; font-size: 12px; }
        .badge { background: #28a745; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-left: 10px; }
        pre { margin: 0; white-space: pre-wrap; word-break: break-all; }
        .helper-text { color: #888; font-size: 13px; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>3D Perception & Pick-and-Place <span class="badge" id="status-badge">Connected</span></h1>
        <p style="color: #aaa; margin-top: 0;">Grounding DINO + SAM + 6-DoF Grasp + Tabletop Placement Zone</p>
        
        <div class="controls">
            <input type="text" id="prompt-input" value="pick up the cup and place it near laptop" placeholder="e.g. pick up the cup and place it near laptop">
            <button onclick="updatePrompt()">Send Command</button>
            <span id="fps-label" style="color: #aaa; margin-left: auto;">FPS: --</span>
        </div>
        <div class="helper-text">Examples: "pick up the cup and place it near laptop" | "grasp bottle and put on table" | "cup"</div>

        <div class="grid" style="margin-top: 15px;">
            <div class="video-box">
                <img id="live-img" src="" alt="Waiting for camera stream from local PC...">
            </div>
            <div class="json-box">
                <h4 style="margin-top: 0; color: #00d2ff;">Robot 3D Telemetry (JSON)</h4>
                <pre id="json-output">Waiting for frame...</pre>
            </div>
        </div>
    </div>

    <script>
        const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/viewer`);
        let frameCount = 0;
        let lastTime = performance.now();

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.image) {
                document.getElementById('live-img').src = "data:image/jpeg;base64," + data.image;
            }
            if (data.payload) {
                document.getElementById('json-output').textContent = JSON.stringify(data.payload, null, 2);
            }
            frameCount++;
            const now = performance.now();
            if (now - lastTime >= 1000) {
                document.getElementById('fps-label').textContent = `FPS: ${frameCount}`;
                frameCount = 0;
                lastTime = now;
            }
        };

        ws.onclose = () => {
            document.getElementById('status-badge').style.background = "#dc3545";
            document.getElementById('status-badge').textContent = "Disconnected";
        };

        function updatePrompt() {
            const val = document.getElementById('prompt-input').value.trim();
            if (val) {
                fetch(`/set_prompt?prompt=${encodeURIComponent(val)}`, { method: "POST" });
            }
        }
    </script>
</body>
</html>
"""

active_viewers = set()


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML_DASHBOARD)


@app.post("/set_prompt")
def set_prompt(prompt: str):
    global current_prompt
    current_prompt = prompt.strip()
    print(f"[*] Target command updated to: {current_prompt}")
    return {"status": "ok", "prompt": current_prompt}


@app.websocket("/ws/viewer")
async def websocket_viewer(websocket: WebSocket):
    await websocket.accept()
    active_viewers.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_viewers.remove(websocket)


@app.websocket("/ws/camera_stream")
async def websocket_camera_stream(websocket: WebSocket):
    """
    WebSocket endpoint for local client to push compressed RGB + Depth frames.
    Client sends binary packet:
      [4 bytes metadata json length] + [metadata JSON bytes] + [JPEG color bytes] + [zlib-compressed uint16 depth bytes]
    """
    global current_prompt, pipeline
    await websocket.accept()
    print("[+] Local Orbbec Camera connected to Kaggle GPU Server!")

    try:
        while True:
            data = await websocket.receive_bytes()
            if len(data) < 4:
                continue

            # Unpack payload
            meta_len = int.from_bytes(data[:4], byteorder="big")
            meta_json = json.loads(data[4 : 4 + meta_len].decode("utf-8"))
            offset = 4 + meta_len

            color_len = meta_json["color_len"]
            color_bytes = data[offset : offset + color_len]
            depth_bytes = data[offset + color_len :]

            # Decode RGB
            color_bgr = cv2.imdecode(np.frombuffer(color_bytes, np.uint8), cv2.IMREAD_COLOR)

            # Decompress depth (uint16 in mm -> float32 in meters)
            h, w = meta_json["height"], meta_json["width"]
            import zlib
            decompressed_depth = zlib.decompress(depth_bytes)
            depth_raw = np.frombuffer(decompressed_depth, dtype=np.uint16).reshape(h, w)
            depth_scale = meta_json.get("depth_scale", 0.001)
            depth_m = depth_raw.astype(np.float32) * depth_scale

            # Intrinsics
            intr_dict = meta_json["intrinsics"]
            intrinsics = CameraIntrinsics(
                fx=intr_dict["fx"],
                fy=intr_dict["fy"],
                cx=intr_dict["cx"],
                cy=intr_dict["cy"],
                width=w,
                height=h,
            )

            # --- GPU INFERENCE ---
            t0 = time.time()
            prompt = meta_json.get("prompt", current_prompt)

            objects_3d, detections, annotated_image, payload = pipeline.process_frame(
                prompt=prompt,
                color_bgr=color_bgr,
                depth_m=depth_m,
                intrinsics=intrinsics,
            )
            payload["inference_time_ms"] = round((time.time() - t0) * 1000.0, 1)

            # Encode annotated frame to JPEG for viewers
            _, enc_jpg = cv2.imencode(".jpg", annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64_img = base64.b64encode(enc_jpg.tobytes()).decode("utf-8")

            # Send back to camera client
            response_msg = json.dumps({"payload": payload, "current_prompt": current_prompt})
            await websocket.send_text(response_msg)

            # Broadcast to web dashboard viewers
            if active_viewers:
                broadcast_data = json.dumps({"image": b64_img, "payload": payload})
                for v in list(active_viewers):
                    try:
                        await v.send_text(broadcast_data)
                    except Exception:
                        active_viewers.remove(v)

    except WebSocketDisconnect:
        print("[-] Local camera streamer disconnected.")


def start_server(port: int = 8000):
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    start_server()
