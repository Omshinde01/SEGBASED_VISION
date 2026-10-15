"""
3D Geometry and 6-DoF Grasp Pose Estimator.
Computes 3D location w.r.t camera and grasp location (x, y, z) + orientation (Roll, Pitch, Yaw).
"""

from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List
import math
import cv2
import numpy as np

from ..camera.orbbec_camera import CameraIntrinsics


@dataclass
class OrientationRPY:
    """Roll, Pitch, Yaw in degrees and radians."""
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    convention: str = "XYZ_extrinsic_zyx_intrinsic"


@dataclass
class GraspPose:
    """Complete 6-DoF grasp pose and physical grasp parameters."""
    # 3D location in camera optical frame (X=right, Y=down, Z=forward) in meters
    location_xyz: List[float]       # [x, y, z] in meters
    location_xyz_mm: List[float]    # [x, y, z] in millimeters

    # Grasp orientation
    rpy: OrientationRPY
    quaternion_wxyz: List[float]    # [qw, qx, qy, qz]
    rotation_matrix: List[List[float]] # 3x3 SO(3) matrix

    # Planar 2D grasp angle (in image plane degrees [-90, +90])
    planar_angle_deg: float

    # Physical gripper opening required
    grasp_width_m: float
    grasp_width_mm: float

    # Approach vector (unit vector in camera frame)
    approach_vector: List[float]

    # Pre-grasp / standoff pose (10cm backed off along approach vector)
    pre_grasp_xyz: List[float]


@dataclass
class Object3D:
    """3D Object spatial attributes w.r.t camera coordinate frame."""
    label: str
    confidence: float
    center_3d: List[float]       # [X, Y, Z] centroid in meters
    center_3d_mm: List[float]    # [X, Y, Z] in millimeters
    bbox_3d_extent_m: List[float]# [width_x, height_y, depth_z] in meters
    grasp_pose: GraspPose
    num_points: int


class GraspEstimator:
    """
    Computes 3D object coordinates and robotic grasp poses from SAM masks and aligned depth maps.
    """

    def __init__(
        self,
        min_depth: float = 0.15,
        max_depth: float = 3.5,
        depth_percentile_low: float = 5.0,
        depth_percentile_high: float = 95.0,
        standoff_dist: float = 0.10,  # 10cm standoff for pre-grasp
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_percentile_low = depth_percentile_low
        self.depth_percentile_high = depth_percentile_high
        self.standoff_dist = standoff_dist

    def deproject_mask(
        self,
        mask: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Deprojects 2D mask pixels with valid depth to 3D camera points.

        Returns:
            points_3d: (N, 3) np.ndarray of [X, Y, Z] in meters
            valid_pixels: (N, 2) np.ndarray of [u, v] pixel coordinates
        """
        # Find pixels inside mask with valid depth
        v_idx, u_idx = np.where(mask > 0)
        if len(u_idx) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32)

        depths = depth_m[v_idx, u_idx]

        # Valid depth mask
        valid = (depths >= self.min_depth) & (depths <= self.max_depth) & np.isfinite(depths)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32)

        u = u_idx[valid].astype(np.float32)
        v = v_idx[valid].astype(np.float32)
        z = depths[valid]

        # Statistical depth outlier rejection (filter foreground/background edge bleeds)
        if len(z) > 10:
            z_low, z_high = np.percentile(z, [self.depth_percentile_low, self.depth_percentile_high])
            inliers = (z >= z_low) & (z <= z_high)
            u = u[inliers]
            v = v[inliers]
            z = z[inliers]

        if len(z) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32)

        # Pin-hole camera deprojection formula:
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = Z
        x = (u - intrinsics.cx) * z / intrinsics.fx
        y = (v - intrinsics.cy) * z / intrinsics.fy

        points_3d = np.stack([x, y, z], axis=-1).astype(np.float32)
        pixels = np.stack([u, v], axis=-1).astype(np.int32)

        return points_3d, pixels

    def compute_object_and_grasp(
        self,
        label: str,
        confidence: float,
        mask: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> Optional[Object3D]:
        """
        Full 3D object estimation and 6-DoF grasp pose calculation.
        """
        points_3d, pixels = self.deproject_mask(mask, depth_m, intrinsics)
        if len(points_3d) < 15:
            return None

        # 1. 3D Object Centroid (Location w.r.t camera)
        center_3d = np.median(points_3d, axis=0)  # robust median against noise

        # 3D bounding box extent
        min_pt = np.min(points_3d, axis=0)
        max_pt = np.max(points_3d, axis=0)
        bbox_extent = (max_pt - min_pt).tolist()

        # 2. Planar 2D grasp angle from mask contour
        planar_angle_deg = self._compute_planar_angle(mask)

        # 3. 3D PCA for 6-DoF Grasp Pose
        grasp_pose = self._compute_grasp_pose_pca(
            points_3d=points_3d,
            center_3d=center_3d,
            planar_angle_deg=planar_angle_deg,
        )

        return Object3D(
            label=label,
            confidence=float(confidence),
            center_3d=[round(float(c), 4) for c in center_3d],
            center_3d_mm=[round(float(c) * 1000.0, 1) for c in center_3d],
            bbox_3d_extent_m=[round(float(e), 4) for e in bbox_extent],
            grasp_pose=grasp_pose,
            num_points=len(points_3d)
        )

    def _compute_planar_angle(self, mask: np.ndarray) -> float:
        """Finds minimum bounding rectangle angle of the 2D mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)  # ((cx, cy), (w, h), angle)
        (cx, cy), (rw, rh), angle = rect

        # Normalize angle so that length is along primary orientation
        if rw < rh:
            angle = angle + 90.0
        # Wrap to [-90, +90]
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        return float(round(angle, 2))

    def _compute_grasp_pose_pca(
        self,
        points_3d: np.ndarray,
        center_3d: np.ndarray,
        planar_angle_deg: float,
    ) -> GraspPose:
        """
        Uses 3D Principal Component Analysis (PCA) to derive:
        - Approach vector (along optical ray or surface normal)
        - Gripper finger closing axis (along secondary/width axis)
        - Rotation matrix, Roll/Pitch/Yaw, and Quaternion
        """
        # Center the point cloud
        pts_centered = points_3d - center_3d

        # 3x3 Covariance Matrix
        cov = np.cov(pts_centered, rowvar=False)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort descending (lambda_1 >= lambda_2 >= lambda_3)
        sort_idx = np.argsort(eigenvalues)[::-1]
        v1 = eigenvectors[:, sort_idx[0]]  # Major axis (longitudinal)
        v2 = eigenvectors[:, sort_idx[1]]  # Intermediate axis (width)
        v3 = eigenvectors[:, sort_idx[2]]  # Minor axis (thickness/normal)

        # Gripper Approach Direction (z_gripper):
        # Approaches from camera side towards the object.
        # Ensure approach vector points forward along camera Z (+Z direction)
        approach = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if np.linalg.norm(center_3d) > 1e-4:
            # Ray from camera center to object centroid
            approach = center_3d / np.linalg.norm(center_3d)

        # Gripper Closing Direction (x_gripper):
        # We want jaws to close across the narrower width (v2), perpendicular to v1
        # Project v2 onto plane perpendicular to approach vector
        x_proj = v2 - np.dot(v2, approach) * approach
        norm_x = np.linalg.norm(x_proj)
        if norm_x > 1e-3:
            x_gripper = x_proj / norm_x
        else:
            # Fallback perpendicular vector
            arbitrary = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(approach, arbitrary)) > 0.9:
                arbitrary = np.array([0.0, 1.0, 0.0])
            x_gripper = np.cross(approach, arbitrary)
            x_gripper /= np.linalg.norm(x_gripper)

        # Ensure x_gripper points consistently (e.g. towards camera +X or positive horizontal)
        if x_gripper[0] < 0:
            x_gripper = -x_gripper

        # Gripper Binormal Direction (y_gripper):
        # y = z x x
        y_gripper = np.cross(approach, x_gripper)
        y_gripper /= np.linalg.norm(y_gripper)

        # Re-orthogonalize x = y x z to guarantee right-handed orthonormal matrix
        x_gripper = np.cross(y_gripper, approach)
        x_gripper /= np.linalg.norm(x_gripper)

        # Construct 3x3 Rotation Matrix R = [x_g, y_g, z_g]
        R = np.column_stack([x_gripper, y_gripper, approach])

        # Ensure det(R) == +1
        if np.linalg.det(R) < 0:
            y_gripper = -y_gripper
            R = np.column_stack([x_gripper, y_gripper, approach])

        # Convert R to Roll, Pitch, Yaw
        rpy = self._rotation_matrix_to_rpy(R)

        # Convert R to Quaternion [qw, qx, qy, qz]
        quat = self._rotation_matrix_to_quaternion(R)

        # Compute physical grasp width (spread along x_gripper axis)
        proj_x = np.dot(pts_centered, x_gripper)
        grasp_width_m = float(np.max(proj_x) - np.min(proj_x))
        # Add small safety padding for gripper fingers
        grasp_width_m = max(0.01, grasp_width_m)

        # Grasp Location (meters)
        grasp_loc = [round(float(c), 4) for c in center_3d]
        grasp_loc_mm = [round(float(c) * 1000.0, 1) for c in center_3d]

        # Pre-grasp standoff location (backed off 10cm along approach vector)
        pre_grasp = center_3d - approach * self.standoff_dist
        pre_grasp_xyz = [round(float(p), 4) for p in pre_grasp]

        return GraspPose(
            location_xyz=grasp_loc,
            location_xyz_mm=grasp_loc_mm,
            rpy=rpy,
            quaternion_wxyz=[round(float(q), 5) for q in quat],
            rotation_matrix=[[round(float(R[i, j]), 4) for j in range(3)] for i in range(3)],
            planar_angle_deg=planar_angle_deg,
            grasp_width_m=round(grasp_width_m, 4),
            grasp_width_mm=round(grasp_width_m * 1000.0, 1),
            approach_vector=[round(float(a), 4) for a in approach],
            pre_grasp_xyz=pre_grasp_xyz
        )

    def _rotation_matrix_to_rpy(self, R: np.ndarray) -> OrientationRPY:
        """
        Extracts Roll, Pitch, Yaw from 3x3 rotation matrix.
        Sequence: Roll (about X), Pitch (about Y), Yaw (about Z).
        Convention: Tait-Bryan angles.
        """
        # sy = sqrt(R[0,0]^2 + R[1,0]^2)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0.0

        return OrientationRPY(
            roll_deg=round(math.degrees(roll), 2),
            pitch_deg=round(math.degrees(pitch), 2),
            yaw_deg=round(math.degrees(yaw), 2),
            roll_rad=round(roll, 4),
            pitch_rad=round(pitch, 4),
            yaw_rad=round(yaw, 4)
        )

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> List[float]:
        """Converts 3x3 SO(3) matrix to unit quaternion [qw, qx, qy, qz]."""
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            S = math.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

        # Normalize quaternion
        norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        return [qw / norm, qx / norm, qy / norm, qz / norm]
