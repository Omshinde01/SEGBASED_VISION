"""
Unified Perception & Manipulation Pipeline.
Integrates:
  1. YAML Config Loader (config.yaml)
  2. Natural Language Command Parser ("pick up cup and place it near laptop")
  3. Grounding DINO + SAM (zero-shot detection + instance segmentation)
  4. 3D Spatial Reconstruction & 6-DoF Grasp Pose Estimation (PCA, RPY, Quaternions)
  5. Supporting Tabletop Plane Fitting & Collision-Free Placement Spot Search
  6. Enhanced Visualization & Structured JSON Serialization
"""

import time
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np

from .config_loader import load_config
from .command_parser import CommandParser, PickAndPlaceTask
from .camera.orbbec_camera import OrbbecGeminiCamera, MockRGBDCamera, CameraIntrinsics, create_camera
from .detector.grounded_sam import GroundedSAMDetector, DetectionResult
from .geometry.grasp_estimator import GraspEstimator, Object3D, GraspPose
from .geometry.surface_detector import SurfaceAndPlacementDetector, PlacementSpot
from .visualizer import PerceptionVisualizer


class PerceptionPipeline:
    """
    Main interface for the prompt-based 3D perception, grasping, and placement system.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        prefer_orbbec: Optional[bool] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ):
        # 1. Load config
        self.config = load_config(config_path)

        cam_cfg = self.config.get("camera", {})
        model_cfg = self.config.get("models", {})
        grasp_cfg = self.config.get("grasp", {})
        place_cfg = self.config.get("placement", {})
        vis_cfg = self.config.get("visualization", {})

        print("=" * 68)
        print(" Initializing 3D Perception, Grasping & Pick-and-Place Pipeline")
        print(" (Orbbec Gemini E + Grounded-SAM + 6-DoF Grasp + Placement Zone)")
        print(f" Loaded configuration from: {config_path}")
        print("=" * 68)

        # 2. Camera setup
        if prefer_orbbec is None:
            prefer_orbbec = (cam_cfg.get("type", "auto") != "mock")
        self.camera, self.is_real_camera = create_camera(prefer_orbbec=prefer_orbbec)
        self.camera_name = "Orbbec Gemini E" if self.is_real_camera else "Mock RGB-D (Simulation)"

        # 3. Vision Models
        self.detector = GroundedSAMDetector(
            dino_model_id=model_cfg.get("dino_model_id", "IDEA-Research/grounding-dino-base"),
            sam_model_id=model_cfg.get("sam_model_id", "facebook/sam-vit-base"),
            device=None if model_cfg.get("device") == "auto" else model_cfg.get("device"),
            box_threshold=box_threshold if box_threshold is not None else model_cfg.get("box_threshold", 0.25),
            text_threshold=text_threshold if text_threshold is not None else model_cfg.get("text_threshold", 0.20),
        )

        # 4. 3D & Grasp Estimator
        self.grasp_estimator = GraspEstimator(
            min_depth=grasp_cfg.get("min_depth_m", 0.15),
            max_depth=grasp_cfg.get("max_depth_m", 3.50),
            depth_percentile_low=grasp_cfg.get("depth_percentile_low", 5.0),
            depth_percentile_high=grasp_cfg.get("depth_percentile_high", 95.0),
            standoff_dist=grasp_cfg.get("standoff_distance_m", 0.10),
        )

        # 5. Supporting Surface & Placement Spot Finder
        self.surface_detector = SurfaceAndPlacementDetector(
            ransac_distance_threshold=place_cfg.get("ransac_distance_threshold_m", 0.015),
            ransac_max_iterations=place_cfg.get("ransac_max_iterations", 1500),
            free_space_radius_m=place_cfg.get("free_space_radius_m", 0.12),
            target_standoff_distance_m=place_cfg.get("target_standoff_distance_m", 0.22),
            height_offset_m=place_cfg.get("height_offset_m", 0.02),
            preferred_sides=place_cfg.get("preferred_sides", ["right", "front", "left"]),
        )

        # 6. Natural Language Command Parser
        self.command_parser = CommandParser()

        # 7. Visualizer
        self.visualizer = PerceptionVisualizer(
            axis_length_m=vis_cfg.get("axis_length_m", 0.065)
        )

        print("[Pipeline] System ready for perception & manipulation queries!")

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
        Executes perception & placement pipeline for the given natural language prompt.
        """
        if color_bgr is None or depth_m is None or intrinsics is None:
            color_bgr, depth_m, intrinsics = self.camera.read()
            if color_bgr is None or depth_m is None or intrinsics is None:
                empty_payload = {
                    "status": "NO_FRAME",
                    "timestamp": time.time(),
                    "prompt": prompt,
                    "task": {"is_pick_and_place": False},
                    "objects": [],
                    "placement": None,
                }
                return [], [], color_bgr if color_bgr is not None else np.zeros((480, 640, 3), np.uint8), empty_payload

        annotated_image = color_bgr.copy()

        # 1. Parse natural language prompt
        task: PickAndPlaceTask = self.command_parser.parse(prompt)

        # 2. Formulate DINO query
        if task.is_pick_and_place:
            query = task.get_combined_dino_prompt()
        else:
            query = prompt.strip()
            if not query.endswith("."):
                query += " ."

        # 3. Detect and segment with Grounded-SAM
        detections: List[DetectionResult] = self.detector.detect_and_segment(
            image_bgr=color_bgr,
            text_prompt=query,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # 4. Separate tabletop surface mask from discrete objects
        objects_3d: List[Object3D] = []
        table_points_3d: Optional[np.ndarray] = None

        source_obj: Optional[Object3D] = None
        target_obj: Optional[Object3D] = None

        for det in detections:
            label_lower = det.label.lower().strip()
            # Check if this detection is the supporting surface itself
            if any(surf_word in label_lower for surf_word in ("table", "desk", "counter", "surface")):
                pts, _ = self.grasp_estimator.deproject_mask(det.mask, depth_m, intrinsics)
                if len(pts) > 0:
                    table_points_3d = pts
                continue

            # Compute 3D Position & 6-DoF Grasp
            obj_3d = self.grasp_estimator.compute_object_and_grasp(
                label=det.label,
                confidence=det.confidence,
                mask=det.mask,
                depth_m=depth_m,
                intrinsics=intrinsics,
            )
            if obj_3d is not None:
                objects_3d.append(obj_3d)

                # Match source and target objects
                if task.is_pick_and_place:
                    if task.source_object in label_lower and source_obj is None:
                        source_obj = obj_3d
                    elif task.target_object and task.target_object in label_lower and target_obj is None:
                        target_obj = obj_3d

                # Mask overlay
                color = self.visualizer.PALETTE[len(objects_3d) % len(self.visualizer.PALETTE)]
                self.visualizer.overlay_mask(annotated_image, det.mask, color=color, alpha=0.35)
                x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

        # If source object wasn't found by label match, pick highest confidence object
        if task.is_pick_and_place and source_obj is None and len(objects_3d) > 0:
            source_obj = objects_3d[0]

        # 5. Compute Collision-Free Placement Spot
        placement_spot: Optional[PlacementSpot] = None
        if task.is_pick_and_place:
            ref = target_obj if target_obj is not None else source_obj
            if ref is not None:
                obstacles = [o for o in objects_3d if o != ref]
                placement_spot = self.surface_detector.find_placement_spot(
                    reference_object=ref,
                    all_obstacles=obstacles,
                    table_points_3d=table_points_3d,
                    depth_m=depth_m,
                    intrinsics=intrinsics,
                )

        # 6. Render visual elements (3D axes, placement zone, trajectory arch)
        annotated_image = self.visualizer.render(
            image_bgr=annotated_image,
            objects_3d=objects_3d,
            intrinsics=intrinsics,
            prompt=prompt,
            camera_name=self.camera_name,
            placement_spot=placement_spot,
            source_object=source_obj,
            target_object=target_obj,
        )

        # 7. Construct robot JSON payload
        json_payload = self._build_json_payload(prompt, task, objects_3d, placement_spot, source_obj, target_obj, intrinsics)

        return objects_3d, detections, annotated_image, json_payload

    def _build_json_payload(
        self,
        raw_prompt: str,
        task: PickAndPlaceTask,
        objects_3d: List[Object3D],
        placement_spot: Optional[PlacementSpot],
        source_obj: Optional[Object3D],
        target_obj: Optional[Object3D],
        intrinsics: CameraIntrinsics,
    ) -> dict:
        """Constructs clean, structured JSON payload for robotic execution."""
        obj_items = []
        for obj in objects_3d:
            p = obj.grasp_pose
            obj_items.append({
                "label": obj.label,
                "confidence": obj.confidence,
                "is_pick_target": (obj == source_obj),
                "is_place_reference": (obj == target_obj),
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

        placement_payload = None
        if placement_spot is not None:
            placement_payload = {
                "status": "VALID_SPOT_FOUND",
                "reference_object": target_obj.label if target_obj else (source_obj.label if source_obj else "surface"),
                "placement_side": placement_spot.placement_side,
                "target_location_xyz": {
                    "units": "meters",
                    "x": placement_spot.location_xyz[0],
                    "y": placement_spot.location_xyz[1],
                    "z": placement_spot.location_xyz[2],
                    "xyz_list": placement_spot.location_xyz,
                    "xyz_mm": placement_spot.location_xyz_mm,
                },
                "pre_place_standoff_xyz_m": placement_spot.pre_place_standoff_xyz_m,
                "release_approach_vector": placement_spot.approach_vector,
                "surface_normal": placement_spot.surface_normal,
                "free_clearance_radius_m": placement_spot.free_clearance_radius_m,
                "distance_to_reference_m": placement_spot.distance_to_reference_m,
            }

        return {
            "status": "SUCCESS" if len(objects_3d) > 0 else "NO_OBJECT",
            "timestamp": time.time(),
            "camera": {
                "name": self.camera_name,
                "is_hardware": self.is_real_camera,
                "intrinsics": intrinsics.to_dict() if intrinsics else None,
            },
            "task": {
                "raw_prompt": raw_prompt,
                "is_pick_and_place": task.is_pick_and_place,
                "source_object": task.source_object,
                "target_reference": task.target_object,
                "target_surface": task.target_surface,
                "relation": task.relation,
            },
            "count": len(objects_3d),
            "objects": obj_items,
            "placement": placement_payload,
        }

    def close(self):
        """Stops the camera."""
        if self.camera is not None:
            self.camera.stop()
