"""
client_streamer.py - Runs on your LOCAL machine with the Orbbec Gemini E camera plugged in.
Streams live Color + Depth frames over a tunnel (ngrok / localtunnel / bore)
to your Kaggle GPU server and renders the returned 3D grasp inferences.

Usage:
    python client_streamer.py --server-url wss://xxxx-xx-xx.ngrok-free.app --prompt "cup"
"""

import sys
import time
import json
import zlib
import argparse
import numpy as np
import cv2
import websocket  # pip install websocket-client

from perception_pipeline.camera.orbbec_camera import create_camera


def parse_args():
    parser = argparse.ArgumentParser(description="Stream local Orbbec Gemini E camera to Kaggle GPU Server.")
    parser.add_argument(
        "--server-url", "-s",
        type=str,
        required=True,
        help="Kaggle WebSocket tunnel URL (e.g., wss://xxxx.ngrok-free.app or ws://xxxx.loca.lt)",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="cup",
        help="Target object prompt (e.g. 'cup', 'bottle', 'charger')",
    )
    parser.add_argument(
        "--camera", "-c",
        type=str,
        default="auto",
        choices=["auto", "orbbec", "mock"],
        help="Camera feed source ('auto', 'orbbec', or 'mock')",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Format ws URL
    ws_url = args.server_url.strip().rstrip("/")
    if ws_url.startswith("http://"):
        ws_url = "ws://" + ws_url[7:]
    elif ws_url.startswith("https://"):
        ws_url = "wss://" + ws_url[8:]
    elif not ws_url.startswith("ws://") and not ws_url.startswith("wss://"):
        ws_url = "wss://" + ws_url

    endpoint = f"{ws_url}/ws/camera_stream"
    print(f"[*] Connecting to Kaggle GPU Server at: {endpoint}...")

    # Connect to WebSocket
    try:
        ws = websocket.create_connection(endpoint, timeout=10)
        print("[+] Connected to Kaggle GPU Server successfully!")
    except Exception as e:
        print(f"[!] Failed to connect to Kaggle tunnel: {e}")
        print("    Make sure ngrok/localtunnel is running inside your Kaggle notebook.")
        return

    # Start camera
    prefer_orbbec = (args.camera != "mock")
    camera, is_real = create_camera(prefer_orbbec=prefer_orbbec)
    cam_name = "Orbbec Gemini E" if is_real else "Mock Simulation"
    print(f"[*] Camera initialized: {cam_name}")

    current_prompt = args.prompt
    window_name = "Local Orbbec Stream -> Kaggle GPU (Press 'q' to quit, 'p' to change prompt)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            t_start = time.time()
            color_bgr, depth_m, intrinsics = camera.read()
            if color_bgr is None or depth_m is None:
                time.sleep(0.01)
                continue

            h, w = color_bgr.shape[:2]

            # 1. Compress RGB frame as JPEG
            _, jpg_bytes = cv2.imencode(".jpg", color_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            color_payload = jpg_bytes.tobytes()

            # 2. Compress raw depth as uint16 (millimeters) using fast zlib
            depth_mm = (depth_m * 1000.0).astype(np.uint16)
            depth_payload = zlib.compress(depth_mm.tobytes(), level=1)

            # 3. Create metadata header
            meta = {
                "width": w,
                "height": h,
                "prompt": current_prompt,
                "color_len": len(color_payload),
                "depth_scale": 0.001,
                "intrinsics": intrinsics.to_dict(),
            }
            meta_json_bytes = json.dumps(meta).encode("utf-8")
            meta_len_bytes = len(meta_json_bytes).to_bytes(4, byteorder="big")

            # Binary payload: [4 bytes meta length] + [meta] + [jpeg] + [zlib depth]
            binary_packet = meta_len_bytes + meta_json_bytes + color_payload + depth_payload

            # 4. Send to Kaggle
            ws.send_binary(binary_packet)

            # 5. Receive inference response from Kaggle
            resp_str = ws.recv()
            resp = json.loads(resp_str)
            payload = resp.get("payload", {})
            current_prompt = resp.get("current_prompt", current_prompt)

            # Print telemetry summary in terminal
            dt_roundtrip = (time.time() - t_start) * 1000.0
            inf_ms = payload.get("inference_time_ms", 0.0)
            status = payload.get("status", "UNKNOWN")
            count = payload.get("count", 0)

            print(f"\r[Kaggle GPU] Target: '{current_prompt}' | Objects: {count} | GPU Inf: {inf_ms:.0f}ms | RTT: {dt_roundtrip:.0f}ms", end="", flush=True)

            if count > 0:
                obj = payload["objects"][0]
                loc = obj["3d_location_wrt_camera"]["xyz_list"]
                rpy = obj["grasp"]["orientation"]["roll_pitch_yaw_deg"]
                print(f"\n  >> {obj['label']} at X={loc[0]:+.2f}m, Y={loc[1]:+.2f}m, Z={loc[2]:.2f}m | Grasp RPY: ({rpy['roll']:+.0f}°, {rpy['pitch']:+.0f}°, {rpy['yaw']:+.0f}°)")

            # Local preview
            display_img = color_bgr.copy()
            cv2.putText(
                display_img,
                f"Kaggle GPU Stream: '{current_prompt}' | RTT: {dt_roundtrip:.0f}ms",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window_name, display_img)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("p"):
                print("\nEnter new prompt: ", end="", flush=True)
                new_p = input().strip()
                if new_p:
                    current_prompt = new_p

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[*] Closing connections...")
        ws.close()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
