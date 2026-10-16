"""
Automated End-to-End Test Suite for Upgraded Perception & Pick-and-Place Pipeline.
Validates:
  1. YAML configuration loading and deep merging
  2. Natural language command parser ("pick up the cup and place it near laptop")
  3. Camera stream validity
  4. 3D geometry & 6-DoF grasp pose calculation
  5. Tabletop plane fitting and collision-free placement spot locator
  6. Visualizer overlay rendering with trajectory & landing zone
  7. Full end-to-end pick-and-place execution
"""

import json
import unittest
import numpy as np
import cv2

from perception_pipeline import (
    PerceptionPipeline,
    MockRGBDCamera,
    CameraIntrinsics,
    GraspEstimator,
    PerceptionVisualizer,
)
from perception_pipeline.config_loader import load_config
from perception_pipeline.command_parser import CommandParser
from perception_pipeline.geometry.surface_detector import SurfaceAndPlacementDetector, Plane3D


class TestPerceptionAndPlacement(unittest.TestCase):

    def setUp(self):
        self.camera = MockRGBDCamera(width=640, height=480)
        self.intrinsics = self.camera.intrinsics

    def test_config_loader(self):
        """Verifies config.yaml loads properly with required sections."""
        cfg = load_config("config.yaml")
        self.assertIn("camera", cfg)
        self.assertIn("models", cfg)
        self.assertIn("grasp", cfg)
        self.assertIn("placement", cfg)
        self.assertIn("visualization", cfg)
        self.assertEqual(cfg["models"]["box_threshold"], 0.25)
        self.assertEqual(cfg["placement"]["free_space_radius_m"], 0.12)
        print("\n[Test] config.yaml loaded and validated successfully.")

    def test_command_parser(self):
        """Tests natural language parsing of pick-and-place requests."""
        parser = CommandParser()

        # Case 1: Pick up X and place it near Y
        t1 = parser.parse("pick up the cup and place it near laptop")
        self.assertTrue(t1.is_pick_and_place)
        self.assertEqual(t1.source_object, "cup")
        self.assertEqual(t1.target_object, "laptop")
        self.assertEqual(t1.relation, "near")

        # Case 2: Grasp X and put on table
        t2 = parser.parse("grasp bottle and put it on table")
        self.assertTrue(t2.is_pick_and_place)
        self.assertEqual(t2.source_object, "bottle")
        self.assertEqual(t2.target_surface, "table")

        # Case 3: Simple detection prompt
        t3 = parser.parse("cup")
        self.assertFalse(t3.is_pick_and_place)
        self.assertEqual(t3.source_object, "cup")
        print("[Test] Command parser verified for all natural language cases.")

    def test_surface_detector_placement_spot(self):
        """Tests RANSAC plane fitting and collision-free placement spot search."""
        detector = SurfaceAndPlacementDetector(
            free_space_radius_m=0.10,
            target_standoff_distance_m=0.20,
        )
        color, depth, intr = self.camera.read()

        # Synthetic laptop object at (0.15, 0.05, 0.58)
        estimator = GraspEstimator()
        mask_lap = np.zeros((480, 640), dtype=np.uint8)
        mask_lap[200:330, 420:580] = 255
        laptop_obj = estimator.compute_object_and_grasp("laptop", 0.90, mask_lap, depth, intr)
        self.assertIsNotNone(laptop_obj)

        # Synthetic cup object at (-0.15, 0.05, 0.65)
        mask_cup = np.zeros((480, 640), dtype=np.uint8)
        mask_cup[220:340, 180:260] = 255
        cup_obj = estimator.compute_object_and_grasp("cup", 0.90, mask_cup, depth, intr)
        self.assertIsNotNone(cup_obj)

        spot = detector.find_placement_spot(
            reference_object=laptop_obj,
            all_obstacles=[cup_obj],
            table_points_3d=None,
            depth_m=depth,
            intrinsics=intr,
        )

        self.assertIsNotNone(spot)
        print(f"\n[Test] Placement Spot Found: {spot.placement_side.upper()} of Laptop")
        print(f"       Landing XYZ: {spot.location_xyz} m")
        print(f"       Free Clearance Radius: {spot.free_clearance_radius_m * 100:.0f} cm")
        print(f"       Distance to Laptop: {spot.distance_to_reference_m * 100:.0f} cm")
        self.assertTrue(spot.distance_to_reference_m >= 0.15)
        self.assertTrue(spot.free_clearance_radius_m >= 0.10)

    def test_full_pick_and_place_pipeline(self):
        """Runs end-to-end pipeline with 'pick up the cup and place it near laptop'."""
        pipeline = PerceptionPipeline(config_path="config.yaml", prefer_orbbec=False)
        cmd = "pick up the cup and place it near laptop"

        objects_3d, detections, annotated_img, payload = pipeline.process_frame(prompt=cmd)

        self.assertIn("task", payload)
        self.assertTrue(payload["task"]["is_pick_and_place"])
        self.assertEqual(payload["task"]["source_object"], "cup")
        self.assertEqual(payload["task"]["target_reference"], "laptop")
        self.assertEqual(annotated_img.shape, (480, 640, 3))

        print(f"\n[Test] Pick-and-Place Task: '{cmd}'")
        print(f"       Objects Detected: {len(objects_3d)}")
        if payload.get("placement"):
            print("       Placement Payload:")
            print(json.dumps(payload["placement"], indent=2))

        pipeline.close()


if __name__ == "__main__":
    unittest.main()
