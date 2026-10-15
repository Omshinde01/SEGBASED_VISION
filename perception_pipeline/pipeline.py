"""
Unified Perception Pipeline.
Integrates:
  1. Orbbec Gemini E Camera (or Mock camera fallback)
  2. Grounding DINO + SAM (zero-shot detection + instance segmentation)
  3. 3D spatial reconstruction & 6-DoF Grasp Pose Estimation (PCA, RPY, Quaternions)
  4. Visualization & JSON serialization
"""

import json
import time
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np

from .camera.orbbec_camera import OrbbecGeminiCamera, MockRGBDCamera, CameraIntrinsics, create_camera
from .detector.grounded_sam import GroundedSAMDetector, DetectionResult
from .geometry.grasp_estimator import GraspEstimator, Object3D, GraspPose, OrientationRPY
from .visualizer import PerceptionVisualizer


class PerceptionPipeline:
    """
    Main interface for the prompt-based 3D perception and grasping system.
    """

    def __init__(
        self,
        prefer_orbbec: bool = True,
        dino_model_id: str = "IDEA-Research/grounding-dino-base",
        sam_model_id: str = "facebook/sam-vit-base",
        device: Optional[str] = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
    ):
        print("=" * 65)
        print(" Initializing 3D Perception & Grasping Pipeline")
        print(" (Orbbec Gemini E + Grounding DINO + SAM + 6-DoF Grasp)")
        print("=" * 65)

        # 1. Camera
        self.camera, self.is_real_camera = create_camera(prefer_orbbec=prefer_orbbec)
        self.camera_name = "Orbbec Gemini E" if self.is_real_camera else "Mock RGB-D (Simulation)"

        # 2. Detector
        self.detector = GroundedSAMDetector(
            dino_model_id=dino_model_id,
            sam_model_id=sam_model_id,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # 3. 3D & Grasp Estimator
        self.grasp_estimator = GraspEstimator()

        # 4. Visualizer
        self.visualizer = PerceptionVisualizer()

        print("[Pipeline] System ready for perception queries!")

    def process_frame(
        self,
        prompt: str,
        color_bgr: Optional[np.ndarray] = None,
        depth_m: Optional[np.ndarray] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> Tuple[List[Object3D], List[DetectionResult], np.ndarray, dict]:
        """
        Processes a single RGB-D frame for the given text prompt.

        If color_bgr / depth_m / intrinsics are omitted, frames are read directly from camera.

        Returns:
            (objects_3d, detections_2d, annotated_image, json_payload)
        """
        # Read from camera if not explicitly passed
        if color_bgr is None or depth_m is None or intrinsics is None:
            color_bgr, depth_m, intrinsics = self.camera.read()
            if color_bgr is None or depth_m is None or intrinsics is None:
                empty_payload = {
                    "status": "NO_FRAME",
                    "timestamp": time.time(),
                    "prompt": prompt,
                    "objects": []
                }
                return [], [], color_bgr if color_bgr is not None else np.zeros((480, 640, 3), np.uint8), empty_payload

        annotated_image = color_bgr.copy()

        # 1. Detect and segment with Grounded-SAM
        detections: List[DetectionResult] = self.detector.detect_and_segment(
            image_bgr=color_bgr,
            text_prompt=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # 2. Compute 3D Position and 6-DoF Grasp Pose for each detection
        objects_3d: List[Object3D] = []
        for det in detections:
            obj_3d = self.grasp_estimator.compute_object_and_grasp(
                label=det.label,
                confidence=det.confidence,
                mask=det.mask,
                depth_m=depth_m,
                intrinsics=intrinsics,
            )
            if obj_3d is not None:
                objects_3d.append(obj_3d)

            # Draw mask overlay on annotated image
            color = self.visualizer.PALETTE[len(objects_3d) % len(self.visualizer.PALETTE)]
            self.visualizer.overlay_mask(annotated_image, det.mask, color=color, alpha=0.4)

            # Draw 2D bounding box
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

        # 3. Render 3D grasp axes and telemetry HUD
        annotated_image = self.visualizer.render(
            image_bgr=annotated_image,
            objects_3d=objects_3d,
            intrinsics=intrinsics,
            prompt=prompt,
            camera_name=self.camera_name,
        )

        # 4. Build robot-facing JSON payload
        json_payload = self._build_json_payload(prompt, objects_3d, intrinsics)

        return objects_3d, detections, annotated_image, json_payload

    def _build_json_payload(
        self,
        prompt: str,
        objects_3d: List[Object3D],
        intrinsics: CameraIntrinsics,
    ) -> dict:
        """Constructs clean nested JSON payload for robot consumption."""
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
                    }
                }
            })

        return {
            "status": "SUCCESS" if len(objects_3d) > 0 else "NO_OBJECT",
            "timestamp": time.time(),
            "camera": {
                "name": self.camera_name,
                "is_hardware": self.is_real_camera,
                "intrinsics": intrinsics.to_dict() if intrinsics else None,
            },
            "prompt": prompt,
            "count": len(objects_3d),
            "objects": items,
        }

    def close(self):
        """Stops the camera."""
        if self.camera is not None:
            self.camera.stop()
