"""
visualizer.py - Enhanced Visualizer:
Renders RGB-D camera feed with SAM segmentation masks,
projected 3D Grasp axes (Roll, Pitch, Yaw), bounding boxes,
Pick-and-Place arrows, and collision-free Placement Zones.
"""

from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np

from .geometry.grasp_estimator import Object3D
from .geometry.surface_detector import PlacementSpot
from .camera.orbbec_camera import CameraIntrinsics


class PerceptionVisualizer:
    """
    Renders 2D/3D perception and pick-and-place results onto the camera image.
    """

    # Distinct colors for multiple objects (BGR)
    PALETTE = [
        (0, 230, 115),   # Bright Green (Source Object)
        (255, 140, 0),   # Bright Blue/Orange (Target Object)
        (0, 215, 255),   # Yellow (Placement Zone)
        (255, 51, 153),  # Pink/Magenta
        (180, 105, 255), # Light Purple
        (0, 165, 255),   # Orange
    ]

    def __init__(self, axis_length_m: float = 0.065):
        self.axis_length_m = axis_length_m
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(
        self,
        image_bgr: np.ndarray,
        objects_3d: List[Object3D],
        intrinsics: CameraIntrinsics,
        prompt: str,
        fps: Optional[float] = None,
        camera_name: str = "Orbbec Gemini E",
        placement_spot: Optional[PlacementSpot] = None,
        source_object: Optional[Object3D] = None,
        target_object: Optional[Object3D] = None,
    ) -> np.ndarray:
        """
        Overlays segmentation masks, 3D grasp coordinate frames,
        placement landing zone, and trajectory guidance.
        """
        canvas = image_bgr.copy()
        h, w = canvas.shape[:2]

        # 1. Draw each detected 3D object
        for idx, obj in enumerate(objects_3d):
            # Pick distinct color
            if source_object is not None and obj == source_object:
                color = (0, 230, 115)  # Bright Green for pick target
            elif target_object is not None and obj == target_object:
                color = (255, 140, 0)  # Bright Blue for reference target
            else:
                color = self.PALETTE[idx % len(self.PALETTE)]

            # Overlay 3D grasp coordinate axes
            self._draw_3d_grasp_axes(canvas, obj, intrinsics)

            # Draw clean badge
            self._draw_object_tag(canvas, obj, intrinsics, color, is_source=(obj == source_object))

        # 2. Draw Placement Landing Spot if available
        if placement_spot is not None:
            self._draw_placement_spot(canvas, placement_spot, intrinsics)

            # Draw transfer curve/arrow from source object to placement spot
            if source_object is not None:
                self._draw_transfer_trajectory(canvas, source_object.center_3d, placement_spot.location_xyz, intrinsics)

        # 3. Draw Telemetry HUD panel at top
        self._draw_hud(canvas, objects_3d, prompt, fps, camera_name, placement_spot)

        return canvas

    def overlay_mask(self, canvas: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.40):
        """Alpha blends binary mask onto the canvas."""
        mask_bool = mask > 0
        if not np.any(mask_bool):
            return

        colored_mask = np.zeros_like(canvas, dtype=np.uint8)
        colored_mask[mask_bool] = color

        canvas[mask_bool] = cv2.addWeighted(canvas[mask_bool], 1.0 - alpha, colored_mask[mask_bool], alpha, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2)

    def _project_point(self, pt_3d: List[float], intrinsics: CameraIntrinsics) -> Optional[Tuple[int, int]]:
        """Projects 3D camera point [X, Y, Z] to 2D image pixel (u, v)."""
        x, y, z = pt_3d
        if z <= 0.05:
            return None
        u = int(round((intrinsics.fx * x / z) + intrinsics.cx))
        v = int(round((intrinsics.fy * y / z) + intrinsics.cy))
        return (u, v)

    def _draw_3d_grasp_axes(self, canvas: np.ndarray, obj: Object3D, intrinsics: CameraIntrinsics):
        """Draws 3D Grasp orientation coordinate frame on object."""
        pose = obj.grasp_pose
        origin_3d = pose.location_xyz
        origin_2d = self._project_point(origin_3d, intrinsics)
        if origin_2d is None:
            return

        R = np.array(pose.rotation_matrix)
        length = self.axis_length_m

        x_axis_3d = (np.array(origin_3d) + R[:, 0] * length).tolist()
        y_axis_3d = (np.array(origin_3d) + R[:, 1] * length).tolist()
        z_axis_3d = (np.array(origin_3d) + R[:, 2] * length).tolist()

        x_2d = self._project_point(x_axis_3d, intrinsics)
        y_2d = self._project_point(y_axis_3d, intrinsics)
        z_2d = self._project_point(z_axis_3d, intrinsics)

        thickness = 2
        # Z axis (Approach) -> Blue
        if z_2d is not None:
            cv2.arrowedLine(canvas, origin_2d, z_2d, (255, 60, 0), thickness, tipLength=0.25)
        # Y axis (Binormal) -> Green
        if y_2d is not None:
            cv2.arrowedLine(canvas, origin_2d, y_2d, (0, 220, 0), thickness, tipLength=0.25)
        # X axis (Finger closing) -> Red
        if x_2d is not None:
            cv2.arrowedLine(canvas, origin_2d, x_2d, (0, 0, 255), thickness, tipLength=0.25)

        # Origin center circle
        cv2.circle(canvas, origin_2d, 5, (255, 255, 255), -1)
        cv2.circle(canvas, origin_2d, 5, (0, 0, 0), 1)

        # Gripper finger tips along X axis
        half_w = pose.grasp_width_m / 2.0
        finger1_3d = (np.array(origin_3d) + R[:, 0] * half_w).tolist()
        finger2_3d = (np.array(origin_3d) - R[:, 0] * half_w).tolist()
        f1_2d = self._project_point(finger1_3d, intrinsics)
        f2_2d = self._project_point(finger2_3d, intrinsics)

        if f1_2d is not None and f2_2d is not None:
            cv2.line(canvas, f1_2d, f2_2d, (0, 255, 255), 2)
            cv2.circle(canvas, f1_2d, 4, (0, 255, 255), -1)
            cv2.circle(canvas, f2_2d, 4, (0, 255, 255), -1)

    def _draw_object_tag(
        self,
        canvas: np.ndarray,
        obj: Object3D,
        intrinsics: CameraIntrinsics,
        color: tuple,
        is_source: bool = False,
    ):
        """Displays readable badge next to the object with 3D metrics."""
        origin_2d = self._project_point(obj.center_3d, intrinsics)
        if origin_2d is None:
            return

        tx, ty = origin_2d[0] + 15, origin_2d[1] - 45
        p = obj.grasp_pose

        title_prefix = "[PICK] " if is_source else ""
        lines = [
            f"{title_prefix}{obj.label.upper()} ({obj.confidence:.2f})",
            f"3D: X={obj.center_3d[0]:+.2f}m Y={obj.center_3d[1]:+.2f}m Z={obj.center_3d[2]:.2f}m",
            f"RPY: ({p.rpy.roll_deg:+.0f}*, {p.rpy.pitch_deg:+.0f}*, {p.rpy.yaw_deg:+.0f}*) | W={p.grasp_width_mm:.0f}mm",
        ]

        line_height = 17
        max_w = 230
        total_h = len(lines) * line_height + 8

        tx = max(10, min(canvas.shape[1] - max_w - 10, tx))
        ty = max(total_h + 10, min(canvas.shape[0] - 10, ty))

        overlay = canvas.copy()
        cv2.rectangle(overlay, (tx - 5, ty - total_h + 5), (tx + max_w, ty + 5), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
        cv2.rectangle(canvas, (tx - 5, ty - total_h + 5), (tx + max_w, ty + 5), color, 1)

        for i, text in enumerate(lines):
            curr_y = ty - total_h + 18 + i * line_height
            t_color = (0, 255, 255) if i == 0 else (240, 240, 240)
            cv2.putText(canvas, text, (tx, curr_y), self.font, 0.38, t_color, 1, cv2.LINE_AA)

    def _draw_placement_spot(self, canvas: np.ndarray, spot: PlacementSpot, intrinsics: CameraIntrinsics):
        """Draws candidate placement ellipse and free clearance perimeter."""
        spot_2d = self._project_point(spot.location_xyz, intrinsics)
        if spot_2d is None:
            return

        cx, cy = spot_2d

        # Estimate projected pixel radius for clearance circle
        z = spot.location_xyz[2]
        r_px = int((intrinsics.fx * spot.free_clearance_radius_m) / max(z, 0.1))
        r_px = max(18, min(120, r_px))

        # 1. Clearance Zone Circle (Translucent Yellow/Green disc)
        overlay = canvas.copy()
        cv2.ellipse(overlay, (cx, cy), (r_px, int(r_px * 0.45)), 0, 0, 360, (0, 230, 255), -1)
        cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0, canvas)

        # 2. Outer border and target crosshair
        cv2.ellipse(canvas, (cx, cy), (r_px, int(r_px * 0.45)), 0, 0, 360, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 5, (0, 255, 255), -1)
        cv2.drawMarker(canvas, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, 10, 1)

        # 3. Surface normal arrow pointing UP
        normal_tip_3d = (np.array(spot.location_xyz) + np.array(spot.surface_normal) * 0.08).tolist()
        normal_tip_2d = self._project_point(normal_tip_3d, intrinsics)
        if normal_tip_2d is not None:
            cv2.arrowedLine(canvas, (cx, cy), normal_tip_2d, (0, 255, 120), 2, tipLength=0.3)

        # 4. Placement info badge
        tag_lines = [
            f"[PLACE ZONE ({spot.placement_side.upper()})]",
            f"Loc: X={spot.location_xyz[0]:+.2f}m Y={spot.location_xyz[1]:+.2f}m Z={spot.location_xyz[2]:.2f}m",
            f"Clearance: {spot.free_clearance_radius_m*100:.0f}cm | Ref Dist: {spot.distance_to_reference_m*100:.0f}cm",
        ]

        tx, ty = cx - 90, cy + 35
        tx = max(10, min(canvas.shape[1] - 210, tx))
        ty = max(50, min(canvas.shape[0] - 10, ty))

        overlay_tag = canvas.copy()
        cv2.rectangle(overlay_tag, (tx - 5, ty - 45), (tx + 205, ty + 10), (20, 20, 30), -1)
        cv2.addWeighted(overlay_tag, 0.82, canvas, 0.18, 0, canvas)
        cv2.rectangle(canvas, (tx - 5, ty - 45), (tx + 205, ty + 10), (0, 255, 255), 1)

        for i, text in enumerate(tag_lines):
            curr_y = ty - 32 + i * 16
            col = (0, 255, 255) if i == 0 else (230, 230, 230)
            cv2.putText(canvas, text, (tx, curr_y), self.font, 0.36, col, 1, cv2.LINE_AA)

    def _draw_transfer_trajectory(
        self,
        canvas: np.ndarray,
        pick_3d: List[float],
        place_3d: List[float],
        intrinsics: CameraIntrinsics,
    ):
        """Draws dotted arch/curve representing pick-to-place motion trajectory."""
        p1 = self._project_point(pick_3d, intrinsics)
        p2 = self._project_point(place_3d, intrinsics)
        if p1 is None or p2 is None:
            return

        # Midpoint arched upwards (in camera frame, -Y is up)
        mid_3d = [
            (pick_3d[0] + place_3d[0]) / 2.0,
            min(pick_3d[1], place_3d[1]) - 0.12,  # 12 cm lift in air
            (pick_3d[2] + place_3d[2]) / 2.0,
        ]
        p_mid = self._project_point(mid_3d, intrinsics)
        if p_mid is None:
            return

        # Quadratic Bezier curve points
        curve_pts = []
        for t in np.linspace(0, 1, 25):
            bx = (1 - t)**2 * p1[0] + 2 * (1 - t) * t * p_mid[0] + t**2 * p2[0]
            by = (1 - t)**2 * p1[1] + 2 * (1 - t) * t * p_mid[1] + t**2 * p2[1]
            curve_pts.append((int(bx), int(by)))

        # Draw trajectory
        for i in range(len(curve_pts) - 1):
            if i % 2 == 0:  # dashed pattern
                cv2.line(canvas, curve_pts[i], curve_pts[i + 1], (0, 255, 255), 2, cv2.LINE_AA)

        # Arrowhead at destination
        cv2.arrowedLine(canvas, curve_pts[-3], curve_pts[-1], (0, 255, 255), 2, tipLength=0.4)

    def _draw_hud(
        self,
        canvas: np.ndarray,
        objects_3d: List[Object3D],
        prompt: str,
        fps: Optional[float],
        camera_name: str,
        placement_spot: Optional[PlacementSpot] = None,
    ):
        """Draws top information banner."""
        h, w = canvas.shape[:2]

        hud_h = 44
        hud_bg = canvas[:hud_h, :].copy()
        cv2.rectangle(hud_bg, (0, 0), (w, hud_h), (15, 15, 20), -1)
        cv2.addWeighted(hud_bg, 0.85, canvas[:hud_h, :], 0.15, 0, canvas[:hud_h, :])
        cv2.line(canvas, (0, hud_h), (w, hud_h), (70, 70, 90), 1)

        fps_str = f"FPS: {fps:.1f}" if fps is not None else "FPS: --"
        cv2.putText(canvas, f"CAMERA: {camera_name} | {fps_str}", (15, 18), self.font, 0.42, (180, 220, 255), 1, cv2.LINE_AA)

        task_status = f"PROMPT: '{prompt}'"
        if placement_spot is not None:
            task_status += f" -> [PLACE SPOT READY on {placement_spot.placement_side.upper()}]"
        cv2.putText(canvas, task_status, (15, 36), self.font, 0.42, (0, 255, 180), 1, cv2.LINE_AA)

        # Legend on the right
        legend_x = max(10, w - 240)
        cv2.putText(canvas, "Grasp Axes:", (legend_x, 18), self.font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, "X: Close(R)", (legend_x + 75, 18), self.font, 0.38, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Y: Binormal(G)", (legend_x + 75, 34), self.font, 0.38, (0, 220, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Z: Approach(B)", (legend_x + 160, 18), self.font, 0.38, (255, 100, 0), 1, cv2.LINE_AA)
