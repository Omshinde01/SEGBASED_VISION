"""
main.py - Interactive 3D Perception & Grasping Pipeline
Feeds live frames from Orbbec Gemini E (or Mock RGB-D Camera), runs Grounded-SAM
for user text prompts, computes 3D coordinates w.r.t camera and 6-DoF grasp pose
(location x,y,z and orientation Roll, Pitch, Yaw).

Run:
    python main.py --prompt "cup"
    python main.py --prompt "bottle"
    python main.py --prompt "box" --camera mock

Interactive commands (type in terminal or press key in GUI window):
    p <text>   Change prompt (e.g.  p bottle)
    s          Save current frame & JSON output to disk
    q / ESC    Quit
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
import cv2
import numpy as np

from perception_pipeline import PerceptionPipeline


class TerminalInputController:
    """Non-blocking terminal input thread for dynamic prompt switching."""
    def __init__(self, initial_prompt: str = "cup"):
        self.prompt = initial_prompt.strip()
        self.save_requested = False
        self.quit_requested = False
        self._lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._input_loop, daemon=True).start()

    def get_state(self):
        with self._lock:
            save = self.save_requested
            self.save_requested = False
            return self.prompt, save, self.quit_requested

    def set_prompt(self, new_prompt: str):
        with self._lock:
            self.prompt = new_prompt.strip()
            print(f"\n[Command] Updated target prompt to: '{self.prompt}'")

    def _input_loop(self):
        while not self.quit_requested:
            try:
                line = input().strip()
            except (EOFError, KeyboardInterrupt):
                with self._lock:
                    self.quit_requested = True
                return

            if not line:
                continue

            cmd, _, arg = line.partition(" ")
            cmd = cmd.lower()

            with self._lock:
                if cmd == "p" and arg:
                    self.prompt = arg.strip()
                    print(f"\n[Command] Target prompt set to: '{self.prompt}'")
                elif cmd == "s":
                    self.save_requested = True
                    print("\n[Command] Snapshot requested.")
                elif cmd in ("q", "quit", "exit"):
                    self.quit_requested = True
                    print("\n[Command] Quitting...")
                    return
                else:
                    # If user directly typed prompt without 'p '
                    self.prompt = line.strip()
                    print(f"\n[Command] Target prompt set to: '{self.prompt}'")


def save_snapshot(annotated_img: np.ndarray, json_payload: dict, output_dir: Path):
    """Saves annotated image and JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")

    img_path = output_dir / f"detection_{timestamp_str}.png"
    json_path = output_dir / f"detection_{timestamp_str}.json"

    cv2.imwrite(str(img_path), annotated_img)
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2)

    print(f"\n[Saved] Snapshot saved:")
    print(f"  Image: {img_path}")
    print(f"  JSON:  {json_path}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prompt-based 3D Perception & Grasping Pipeline using Orbbec Gemini E & Grounded-SAM."
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="cup",
        help="Initial object text prompt (e.g. 'cup', 'bottle', 'box')."
    )
    parser.add_argument(
        "--camera", "-c",
        type=str,
        default="auto",
        choices=["auto", "orbbec", "mock"],
        help="Camera feed source ('auto', 'orbbec', or 'mock')."
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.25,
        help="Grounding DINO bounding box confidence threshold."
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.20,
        help="Grounding DINO text alignment threshold."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI window, streaming JSON to stdout."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save snapshot images and JSON payloads."
    )
    parser.add_argument(
        "--single-frame",
        action="store_true",
        help="Process a single frame and exit."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prefer_orbbec = (args.camera != "mock")

    # Initialize Pipeline
    pipeline = PerceptionPipeline(
        prefer_orbbec=prefer_orbbec,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    output_dir = Path(args.output_dir)
    controller = TerminalInputController(initial_prompt=args.prompt)
    if not args.single_frame:
        controller.start()

    print("\n" + "=" * 65)
    print(" PIPELINE ACTIVE")
    print(f" Target Object Prompt: '{args.prompt}'")
    print(" Terminal commands:")
    print("   p <name> : change prompt   (e.g.  p bottle)")
    print("   s        : save snapshot & JSON")
    print("   q        : quit")
    print("=" * 65 + "\n")

    window_name = "Perception Pipeline - 3D Grasp & Grounded-SAM"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.time()
    try:
        while True:
            current_prompt, save_req, quit_req = controller.get_state()
            if quit_req:
                break

            t0 = time.time()
            objects_3d, detections, annotated_img, payload = pipeline.process_frame(
                prompt=current_prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
            dt = time.time() - t0
            fps = 1.0 / max(dt, 1e-4)

            # Print JSON summary
            if len(objects_3d) > 0:
                print(f"[Detected {len(objects_3d)} object(s) for '{current_prompt}'] (Inference: {dt*1000:.1f}ms):")
                for obj in objects_3d:
                    p = obj.grasp_pose
                    print(f"  >> Label: {obj.label} (Conf: {obj.confidence:.2f})")
                    print(f"     3D Loc (wrt camera): X={p.location_xyz[0]:+.3f}m, Y={p.location_xyz[1]:+.3f}m, Z={p.location_xyz[2]:.3f}m")
                    print(f"     Grasp Orientation:   Roll={p.rpy.roll_deg:+.1f} deg, Pitch={p.rpy.pitch_deg:+.1f} deg, Yaw={p.rpy.yaw_deg:+.1f} deg")
                    print(f"     Grasp Width:         {p.grasp_width_mm:.1f} mm | Planar Yaw: {p.planar_angle_deg:+.1f} deg")
                    print(f"     Approach Vector:     {p.approach_vector}")
                    print(f"     Pre-grasp Standoff:  {p.pre_grasp_xyz}")
            else:
                print(f"[No objects matching prompt '{current_prompt}'] (Inference: {dt*1000:.1f}ms)", end="\r")

            if save_req:
                save_snapshot(annotated_img, payload, output_dir)

            if args.single_frame:
                # Print full json payload in single frame mode
                print("\n--- JSON OUTPUT ---")
                print(json.dumps(payload, indent=2))
                if save_req or True:
                    save_snapshot(annotated_img, payload, output_dir)
                break

            if not args.headless:
                cv2.imshow(window_name, annotated_img)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):  # ESC or q
                    break
                elif key == ord('s'):
                    save_snapshot(annotated_img, payload, output_dir)
                elif key == ord('p'):
                    print("\nEnter new target prompt: ", end="", flush=True)
                    # Small wait to let terminal input capture
                    new_p = input().strip()
                    if new_p:
                        controller.set_prompt(new_p)

    finally:
        print("\n[Pipeline] Shutting down...")
        pipeline.close()
        if not args.headless:
            cv2.destroyAllWindows()
        print("[Pipeline] Closed cleanly.")


if __name__ == "__main__":
    main()
