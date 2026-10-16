"""
surface_detector.py - Detects supporting tabletop surfaces, estimates the 3D plane,
and finds collision-free placement spots with clearance margins near reference objects.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import cv2

from ..camera.orbbec_camera import CameraIntrinsics
from .grasp_estimator import Object3D


@dataclass
class Plane3D:
    """Represents a 3D plane: a*X + b*Y + c*Z + d = 0."""
    a: float
    b: float
    c: float
    d: float
    inlier_ratio: float = 0.0

    @property
    def normal(self) -> np.ndarray:
        n = np.array([self.a, self.b, self.c], dtype=np.float64)
        norm = np.linalg.norm(n)
        return n / (norm + 1e-8)

    def distance_to_point(self, point_3d: np.ndarray) -> float:
        """Signed perpendicular distance from point to plane."""
        n = self.normal
        return float(np.dot(n, point_3d) + self.d)

    def project_point(self, point_3d: np.ndarray) -> np.ndarray:
        """Projects 3D point orthogonally onto the plane."""
        dist = self.distance_to_point(point_3d)
        return point_3d - dist * self.normal


@dataclass
class PlacementSpot:
    """Valid, collision-free release location on a supporting surface."""
    location_xyz: List[float]       # [X, Y, Z] in meters
    location_xyz_mm: List[float]    # [X, Y, Z] in millimeters
    approach_vector: List[float]    # Release approach vector (downwards / normal)
    surface_normal: List[float]     # Unit normal vector of surface
    free_clearance_radius_m: float  # Confirmed obstacle-free radius around the spot
    distance_to_reference_m: float  # Distance from reference object (e.g. laptop)
    placement_side: str             # "right", "front", "left", etc.
    pre_place_standoff_xyz_m: List[float] # Height offset before final release


class SurfaceAndPlacementDetector:
    """
    Identifies supporting surfaces (table, desk) using depth RANSAC plane fitting
    and searches for collision-free placement spots adjacent to target reference objects.
    """

    def __init__(
        self,
        ransac_distance_threshold: float = 0.015,
        ransac_max_iterations: int = 1500,
        free_space_radius_m: float = 0.12,
        target_standoff_distance_m: float = 0.22,
        height_offset_m: float = 0.02,
        preferred_sides: Optional[List[str]] = None,
    ):
        self.ransac_distance_threshold = ransac_distance_threshold
        self.ransac_max_iterations = ransac_max_iterations
        self.free_space_radius_m = free_space_radius_m
        self.target_standoff_distance_m = target_standoff_distance_m
        self.height_offset_m = height_offset_m
        self.preferred_sides = preferred_sides or ["right", "front", "left"]

    def fit_plane_ransac(self, points_3d: np.ndarray) -> Optional[Plane3D]:
        """Fits a dominant plane equation to 3D point cloud using RANSAC."""
        if len(points_3d) < 30:
            return None

        # Downsample if cloud is large for real-time speed
        if len(points_3d) > 2000:
            idx = np.random.choice(len(points_3d), 2000, replace=False)
            pts = points_3d[idx]
        else:
            pts = points_3d

        best_inliers = 0
        best_plane = None
        thresh = self.ransac_distance_threshold

        # RANSAC iterations
        for _ in range(self.ransac_max_iterations):
            sample_idx = np.random.choice(len(pts), 3, replace=False)
            p1, p2, p3 = pts[sample_idx[0]], pts[sample_idx[1]], pts[sample_idx[2]]

            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            n_len = np.linalg.norm(normal)
            if n_len < 1e-5:
                continue

            normal = normal / n_len
            d = -np.dot(normal, p1)

            # In the camera frame (Y down), a horizontal table surface has normal pointing UP (towards -Y)
            if normal[1] > 0:
                normal = -normal
                d = -d

            # Distance of all points to candidate plane
            dists = np.abs(np.dot(pts, normal) + d)
            inliers = np.count_nonzero(dists < thresh)

            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = (normal[0], normal[1], normal[2], d, float(inliers) / len(pts))

        if best_plane is not None and best_plane[4] > 0.15:
            return Plane3D(
                a=float(best_plane[0]),
                b=float(best_plane[1]),
                c=float(best_plane[2]),
                d=float(best_plane[3]),
                inlier_ratio=float(best_plane[4]),
            )
        return None

    def find_placement_spot(
        self,
        reference_object: Object3D,
        all_obstacles: List[Object3D],
        table_points_3d: Optional[np.ndarray],
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> Optional[PlacementSpot]:
        """
        Finds a collision-free spot on the supporting plane adjacent to the reference object.
        """
        ref_center = np.array(reference_object.center_3d, dtype=np.float64)

        # 1. Estimate table plane: use table points if available, otherwise fit around base of reference object
        plane = None
        if table_points_3d is not None and len(table_points_3d) >= 50:
            plane = self.fit_plane_ransac(table_points_3d)

        if plane is None:
            # Fallback horizontal plane directly beneath reference object base
            # In camera frame: +Y is down, so bottom of object is max Y
            extent_y = reference_object.bbox_3d_extent_m[1]
            surface_y = ref_center[1] + (extent_y / 2.0)
            # Plane Y = surface_y  --> 0*X + 1*Y + 0*Z - surface_y = 0
            plane = Plane3D(a=0.0, b=-1.0, c=0.0, d=float(surface_y), inlier_ratio=1.0)

        # Collect obstacle centers & radii to avoid collisions
        obstacle_spheres: List[Tuple[np.ndarray, float]] = []
        for obs in all_obstacles:
            c = np.array(obs.center_3d, dtype=np.float64)
            extent = obs.bbox_3d_extent_m
            radius = max(extent[0], extent[2]) / 2.0 + 0.05  # half width + 5cm clearance margin
            obstacle_spheres.append((c, radius))

        # Define candidate search angles around reference object (Right, Front, Left)
        # In Camera optical frame: +X is Right, +Y is Down, +Z is Forward
        side_offsets = {
            "right": np.array([1.0, 0.0, 0.0]),
            "front": np.array([0.0, 0.0, -0.8]),   # closer towards camera
            "left":  np.array([-1.0, 0.0, 0.0]),
            "back":  np.array([0.0, 0.0, 0.8]),    # further behind
        }

        dist_candidates = [
            self.target_standoff_distance_m,
            self.target_standoff_distance_m * 1.3,
            self.target_standoff_distance_m * 1.6,
        ]

        # Scan candidate locations
        best_candidate: Optional[np.ndarray] = None
        best_side: str = "right"
        best_score = -1.0

        for side in self.preferred_sides:
            dir_vec = side_offsets.get(side, np.array([1.0, 0.0, 0.0]))
            # Make sure direction is orthogonal to plane normal
            dir_proj = dir_vec - np.dot(dir_vec, plane.normal) * plane.normal
            norm_d = np.linalg.norm(dir_proj)
            if norm_d < 1e-4:
                continue
            dir_proj /= norm_d

            for r_dist in dist_candidates:
                candidate_3d = ref_center + dir_proj * r_dist
                # Project onto table plane
                candidate_on_plane = plane.project_point(candidate_3d)

                # Check clearance against all obstacle spheres
                min_clearance = float("inf")
                has_collision = False

                for obs_center, obs_radius in obstacle_spheres:
                    d_to_obs = np.linalg.norm(candidate_on_plane - obs_center)
                    clearance = d_to_obs - obs_radius
                    if clearance < self.free_space_radius_m:
                        has_collision = True
                        break
                    if clearance < min_clearance:
                        min_clearance = clearance

                if not has_collision:
                    # Score based on adequate clearance and proximity to target
                    score = min_clearance / (r_dist + 1e-3)
                    if score > best_score:
                        best_score = score
                        best_candidate = candidate_on_plane
                        best_side = side

        if best_candidate is None:
            # Fallback default offset to right of reference object
            best_candidate = ref_center + np.array([self.target_standoff_distance_m, 0.0, 0.0])
            best_candidate = plane.project_point(best_candidate)
            best_side = "right_default"

        # Apply small height release clearance (e.g. 2 cm above table surface)
        # Normal points UP towards -Y, so moving up is +plane.normal
        release_spot = best_candidate - plane.normal * self.height_offset_m
        dist_to_ref = float(np.linalg.norm(release_spot - ref_center))

        # Pre-place standoff pose (10 cm above release point along plane normal)
        pre_place = release_spot - plane.normal * 0.10

        return PlacementSpot(
            location_xyz=[round(float(c), 4) for c in release_spot],
            location_xyz_mm=[round(float(c) * 1000.0, 1) for c in release_spot],
            approach_vector=[round(float(n), 4) for n in (-plane.normal)],  # approaches downward into table
            surface_normal=[round(float(n), 4) for n in plane.normal],
            free_clearance_radius_m=round(self.free_space_radius_m, 3),
            distance_to_reference_m=round(dist_to_ref, 3),
            placement_side=best_side,
            pre_place_standoff_xyz_m=[round(float(p), 4) for p in pre_place],
        )
