"""
kaggle_server.py - Runs on Kaggle Notebook with GPU acceleration.
Receives RGB + Depth frames from local client over WebSocket/HTTP,
runs Grounding DINO + SAM on CUDA, computes 6-DoF grasp poses,
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

from perception_pipeline.detector.grounded_sam import GroundedSAMDetector
from perception_pipeline.geometry.grasp_estimator import GraspEstimator
from perception_pipeline.camera.orbbec_camera import CameraIntrinsics
from perception_pipeline.visualizer import PerceptionVisualizer

app = FastAPI(title="Kaggle 3D Perception & Grasping Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global perception state
detector: Optional[GroundedSAMDetector] = None
grasp_estimator = GraspEstimator()
visualizer = PerceptionVisualizer()
device = "cuda" if torch.cuda.is_available() else "cpu"

current_prompt = "cup"
box_threshold = 0.25
text_threshold = 0.20


@app.on_event("startup")
def startup_event():
    global detector
    print(f"[*] Initializing Grounded-SAM on device: {device} (Kaggle GPU)...")
    detector = GroundedSAMDetector(
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    print("[*] Models loaded and warm!")


HTML_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <title>Kaggle 3D Perception Stream</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #121214; color: #fff; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: auto; }
        h1 { color: #00d2ff; margin-bottom: 5px; }
        .controls { margin-bottom: 15px; display: flex; gap: 10px; align-items: center; }
        input[type="text"] { padding: 10px 14px; border-radius: 6px; border: 1px solid #333; background: #1e1e24; color: #fff; font-size: 16px; width: 300px; }
        button { padding: 10px 20px; border-radius: 6px; border: none; background: #00d2ff; color: #000; font-weight: bold; cursor: pointer; font-size: 15px; }
        button:hover { background: #33e1ff; }
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        .video-box { background: #000; border-radius: 8px; overflow: hidden; position: relative; border: 1px solid #2d2d38; }
        #live-img { width: 100%; height: auto; display: block; min-height: 480px; }
        .json-box { background: #1a1a22; border-radius: 8px; padding: 15px; overflow-y: auto; max-height: 520px; border: 1px solid #2d2d38; font-family: monospace; font-size: 13px; }
        .badge { background: #28a745; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-left: 10px; }
        pre { margin: 0; white-space: pre-wrap; word-break: break-all; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Kaggle GPU 3D Perception & Grasping <span class="badge" id="status-badge">Connected</span></h1>
        <p style="color: #888; margin-top: 0;">Grounding DINO + SAM + 6-DoF Grasp Pose Stream</p>
        
        <div class="controls">
            <input type="text" id="prompt-input" value="cup" placeholder="Enter prompt e.g. bottle, cup, box">
            <button onclick="updatePrompt()">Update Target</button>
            <span id="fps-label" style="color: #aaa; margin-left: auto;">FPS: --</span>
        </div>

        <div class="grid">
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

# Viewer websocket clients
active_viewers = set()


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML_DASHBOARD)


@app.post("/set_prompt")
def set_prompt(prompt: str):
    global current_prompt
    current_prompt = prompt.strip()
    print(f"[*] Target prompt updated to: {current_prompt}")
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
    global current_prompt
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

            # 1. Grounded-SAM (CUDA)
            detections = detector.detect_and_segment(
                image_bgr=color_bgr,
                text_prompt=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            # 2. 3D & 6-DoF Grasp calculation
            objects_3d = []
            annotated_image = color_bgr.copy()

            for det in detections:
                obj_3d = grasp_estimator.compute_object_and_grasp(
                    label=det.label,
                    confidence=det.confidence,
                    mask=det.mask,
                    depth_m=depth_m,
                    intrinsics=intrinsics,
                )
                if obj_3d is not None:
                    objects_3d.append(obj_3d)

                color = visualizer.PALETTE[len(objects_3d) % len(visualizer.PALETTE)]
                visualizer.overlay_mask(annotated_image, det.mask, color=color, alpha=0.4)
                x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            # 3. 3D Coordinate frame projection & HUD
            annotated_image = visualizer.render(
                image_bgr=annotated_image,
                objects_3d=objects_3d,
                intrinsics=intrinsics,
                prompt=prompt,
                camera_name="Orbbec Gemini E (via Kaggle GPU)",
            )

            # Build JSON output
            payload = build_json_payload(prompt, objects_3d, intrinsics)
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


def build_json_payload(prompt, objects_3d, intrinsics):
    items = []
    for obj in objects_3d:
        p = obj.grasp_pose
        items.append({
            "label": obj.label,
            "confidence": obj.confidence,
            "num_3d_points": obj.num_points,
            "3d_location_wrt_camera": {
                "units": "meters",
                "x": obj.center_3d[0],
                "y": obj.center_3d[1],
                "z": obj.center_3d[2],
                "xyz_list": obj.center_3d,
                "xyz_mm": obj.center_3d_mm,
                "bounding_box_extent_xyz_m": obj.bbox_3d_extent_m,
            },
            "grasp": {
                "location_xyz": {
                    "units": "meters",
                    "x": p.location_xyz[0],
                    "y": p.location_xyz[1],
                    "z": p.location_xyz[2],
                    "xyz_list": p.location_xyz,
                    "xyz_mm": p.location_xyz_mm,
                },
                "orientation": {
                    "convention": p.rpy.convention,
                    "roll_pitch_yaw_deg": {
                        "roll": p.rpy.roll_deg,
                        "pitch": p.rpy.pitch_deg,
                        "yaw": p.rpy.yaw_deg,
                    },
                    "roll_pitch_yaw_rad": {
                        "roll": p.rpy.roll_rad,
                        "pitch": p.rpy.pitch_rad,
                        "yaw": p.rpy.yaw_rad,
                    },
                    "quaternion_wxyz": {
                        "qw": p.quaternion_wxyz[0],
                        "qx": p.quaternion_wxyz[1],
                        "qy": p.quaternion_wxyz[2],
                        "qz": p.quaternion_wxyz[3],
                        "wxyz_list": p.quaternion_wxyz,
                    },
                    "rotation_matrix_3x3": p.rotation_matrix,
                    "planar_yaw_angle_deg": p.planar_angle_deg,
                },
                "gripper": {
                    "required_width_m": p.grasp_width_m,
                    "required_width_mm": p.grasp_width_mm,
                    "approach_vector": p.approach_vector,
                    "pre_grasp_standoff_xyz_m": p.pre_grasp_xyz,
                },
            },
        })

    return {
        "status": "SUCCESS" if len(objects_3d) > 0 else "NO_OBJECT",
        "timestamp": time.time(),
        "prompt": prompt,
        "count": len(objects_3d),
        "objects": items,
    }


def start_server(port: int = 8000):
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    start_server()
