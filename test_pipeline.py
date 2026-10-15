"""
Automated End-to-End Test Suite for Perception Pipeline.
Validates:
  1. Camera interface and RGB-D depth frame validity
  2. Grounding DINO + SAM detection and mask generation
  3. 3D location [X, Y, Z] calculation w.r.t camera frame
  4. 6-DoF Grasp pose calculation (Location x,y,z, Roll, Pitch, Yaw, Quaternion, Grasp width)
  5. JSON payload schema and values
  6. Visualizer overlay rendering
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


class TestPerceptionPipeline(unittest.TestCase):

    def setUp(self):
        self.camera = MockRGBDCamera(width=640, height=480)
        self.intrinsics = self.camera.intrinsics

    def test_camera_stream(self):
        """Verifies RGB and Depth frames are properly formatted."""
        color, depth, intr = self.camera.read()
        self.assertIsNotNone(color)
        self.assertIsNotNone(depth)
        self.assertEqual(color.shape, (480, 640, 3))
        self.assertEqual(depth.shape, (480, 640))
        self.assertEqual(color.dtype, np.uint8)
        self.assertEqual(depth.dtype, np.float32)
        # Verify depth values are in realistic meter range (e.g. 0.2m to 2.0m)
        valid_depths = depth[depth > 0]
        self.assertTrue(len(valid_depths) > 1000)
        self.assertTrue(np.all((valid_depths >= 0.1) & (valid_depths <= 3.0)))

    def test_grasp_estimator_geometry(self):
        """Tests 3D deprojection and PCA grasp calculation with a synthetic mask."""
        estimator = GraspEstimator()
        color, depth, intr = self.camera.read()

        # Synthetic mask of the cup (around pixels 180..260 x 220..340)
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[220:340, 180:260] = 255

        obj_3d = estimator.compute_object_and_grasp(
            label="cup",
            confidence=0.92,
            mask=mask,
            depth_m=depth,
            intrinsics=intr,
        )

        self.assertIsNotNone(obj_3d)
        self.assertEqual(obj_3d.label, "cup")
        self.assertAlmostEqual(obj_3d.confidence, 0.92, places=2)

        # Check 3D location w.r.t camera
        loc = obj_3d.center_3d
        self.assertEqual(len(loc), 3)
        print(f"\n[Test] 3D Centroid w.r.t Camera: X={loc[0]:.4f}m, Y={loc[1]:.4f}m, Z={loc[2]:.4f}m")
        # In mock camera, cup is at X < 0 (left of image center 320), Y ~ center, Z ~ 0.65m
        self.assertTrue(loc[0] < 0.0)  # Left of optical center
        self.assertAlmostEqual(loc[2], 0.65, delta=0.05)  # Depth around 0.65m

        # Check Grasp Pose
        gp = obj_3d.grasp_pose
        self.assertIsNotNone(gp)
        self.assertEqual(len(gp.location_xyz), 3)
        self.assertEqual(len(gp.location_xyz_mm), 3)

        # Verify RPY
        rpy = gp.rpy
        print(f"[Test] Grasp Orientation RPY: Roll={rpy.roll_deg:.1f} deg, Pitch={rpy.pitch_deg:.1f} deg, Yaw={rpy.yaw_deg:.1f} deg")
        self.assertTrue(-180.0 <= rpy.roll_deg <= 180.0)
        self.assertTrue(-180.0 <= rpy.pitch_deg <= 180.0)
        self.assertTrue(-180.0 <= rpy.yaw_deg <= 180.0)

        # Verify Quaternion
        q = gp.quaternion_wxyz
        q_norm = np.linalg.norm(q)
        self.assertAlmostEqual(q_norm, 1.0, places=3)

        # Verify Rotation Matrix is SO(3)
        R = np.array(gp.rotation_matrix)
        self.assertEqual(R.shape, (3, 3))
        det = np.linalg.det(R)
        self.assertAlmostEqual(det, 1.0, places=3)
        identity = np.dot(R.T, R)
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-3)

        # Verify Grasp Width
        print(f"[Test] Grasp Width: {gp.grasp_width_mm:.1f} mm ({gp.grasp_width_m:.4f} m)")
        self.assertTrue(gp.grasp_width_m > 0.01)

        # Verify Standoff location
        self.assertEqual(len(gp.pre_grasp_xyz), 3)
        self.assertTrue(gp.pre_grasp_xyz[2] < gp.location_xyz[2])  # Standoff is closer to camera than object

    def test_visualizer(self):
        """Verifies visualizer renders without exceptions."""
        estimator = GraspEstimator()
        color, depth, intr = self.camera.read()

        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[220:340, 180:260] = 255

        obj_3d = estimator.compute_object_and_grasp("cup", 0.95, mask, depth, intr)
        vis = PerceptionVisualizer()

        annotated = vis.render(
            image_bgr=color,
            objects_3d=[obj_3d],
            intrinsics=intr,
            prompt="cup",
            fps=28.5,
            camera_name="Test Camera"
        )
        self.assertEqual(annotated.shape, color.shape)
        self.assertEqual(annotated.dtype, np.uint8)

    def test_full_pipeline_single_frame(self):
        """Runs the entire pipeline end-to-end with Grounded-SAM on mock camera."""
        pipeline = PerceptionPipeline(prefer_orbbec=False)
        objects_3d, detections, annotated_img, payload = pipeline.process_frame(prompt="cup")

        self.assertIn("status", payload)
        self.assertIn("objects", payload)
        self.assertEqual(annotated_img.shape, (480, 640, 3))
        print(f"\n[Test] Pipeline returned status: {payload['status']} with {len(objects_3d)} objects.")

        if len(objects_3d) > 0:
            obj = payload["objects"][0]
            self.assertIn("3d_location_wrt_camera", obj)
            self.assertIn("grasp", obj)
            self.assertIn("orientation", obj["grasp"])
            print("[Test] Verified JSON structure successfully:")
            print(json.dumps(obj["grasp"], indent=2))

        pipeline.close()


if __name__ == "__main__":
    unittest.main()
