"""
Perception Visualizer: Renders RGB-D camera feed with SAM segmentation mask,
projected 3D Grasp axes (Roll, Pitch, Yaw), bounding boxes, and robotics telemetry HUD.
"""

from typing import List, Optional
import cv2
import numpy as np

from .geometry.grasp_estimator import Object3D
from .camera.orbbec_camera import CameraIntrinsics


class PerceptionVisualizer:
    """
    Renders 2D/3D perception results onto the camera image.
    """

    # Distinct colors for multiple objects (BGR)
    PALETTE = [
        (0, 230, 115),   # Bright Green
        (255, 140, 0),   # Bright Blue/Orange
        (255, 51, 153),  # Pink/Magenta
        (0, 215, 255),   # Yellow
        (180, 105, 255), # Light Purple
        (0, 165, 255),   # Orange
    ]

    def __init__(self, axis_length_m: float = 0.06):
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
    ) -> np.ndarray:
        """
        Overlays segmentation masks, 3D grasp coordinate frames, and telemetry.
        """
        canvas = image_bgr.copy()
        h, w = canvas.shape[:2]

        # Draw each detected 3D object
        for idx, obj in enumerate(objects_3d):
            color = self.PALETTE[idx % len(self.PALETTE)]

            # 1. Overlay 3D grasp coordinate axes projected onto the image
            self._draw_3d_grasp_axes(canvas, obj, intrinsics)

            # 2. Draw 3D position and Grasp RPY tag near object
            self._draw_object_tag(canvas, obj, intrinsics, color)

        # 3. Draw Telemetry HUD panel at the top
        self._draw_hud(canvas, objects_3d, prompt, fps, camera_name)

        return canvas

    def overlay_mask(self, canvas: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.45):
        """Alpha blends binary mask onto the canvas."""
        mask_bool = mask > 0
        if not np.any(mask_bool):
            return

        colored_mask = np.zeros_like(canvas, dtype=np.uint8)
        colored_mask[mask_bool] = color

        # Blend where mask is positive
        canvas[mask_bool] = cv2.addWeighted(canvas[mask_bool], 1.0 - alpha, colored_mask[mask_bool], alpha, 0)

        # Draw contour border
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2)

    def _project_point(self, pt_3d: List[float], intrinsics: CameraIntrinsics) -> Optional[tuple]:
        """Projects 3D camera point [X, Y, Z] to 2D image pixel (u, v)."""
        x, y, z = pt_3d
        if z <= 0.01:
            return None
        u = int(round((intrinsics.fx * x / z) + intrinsics.cx))
        v = int(round((intrinsics.fy * y / z) + intrinsics.cy))
        return (u, v)

    def _draw_3d_grasp_axes(self, canvas: np.ndarray, obj: Object3D, intrinsics: CameraIntrinsics):
        """
        Draws 3D Grasp orientation coordinate frame on the object:
        - Red:   X-axis (Gripper finger closing vector)
        - Green: Y-axis (Gripper binormal vector)
        - Blue:  Z-axis (Gripper approach vector)
        """
        pose = obj.grasp_pose
        origin_3d = pose.location_xyz
        origin_2d = self._project_point(origin_3d, intrinsics)
        if origin_2d is None:
            return

        R = np.array(pose.rotation_matrix)  # 3x3 [x_g, y_g, z_g]
        length = self.axis_length_m

        # End points in 3D
        x_axis_3d = (np.array(origin_3d) + R[:, 0] * length).tolist()
        y_axis_3d = (np.array(origin_3d) + R[:, 1] * length).tolist()
        z_axis_3d = (np.array(origin_3d) + R[:, 2] * length).tolist()

        x_2d = self._project_point(x_axis_3d, intrinsics)
        y_2d = self._project_point(y_axis_3d, intrinsics)
        z_2d = self._project_point(z_axis_3d, intrinsics)

        # Draw axes
        thickness = 3
        # Z axis (Approach) -> Blue
        if z_2d is not None:
            cv2.arrowedLine(canvas, origin_2d, z_2d, (255, 60, 0), thickness, tipLength=0.25)
        # Y axis (Binormal) -> Green
        if y_2d is not None:
            cv2.arrowedLine(canvas, origin_2d, y_2d, (0, 220, 0), thickness, tipLength=0.25)
        # X axis (Finger closing) -> Red
        if x_2d is not None:
            cv2.arrowedLine(canvas, origin_2d, x_2d, (0, 0, 255), thickness, tipLength=0.25)

        # Draw origin center circle
        cv2.circle(canvas, origin_2d, 5, (255, 255, 255), -1)
        cv2.circle(canvas, origin_2d, 5, (0, 0, 0), 1)

        # Draw gripper finger tips along X axis
        half_w = (pose.grasp_width_m / 2.0)
        finger1_3d = (np.array(origin_3d) + R[:, 0] * half_w).tolist()
        finger2_3d = (np.array(origin_3d) - R[:, 0] * half_w).tolist()
        f1_2d = self._project_point(finger1_3d, intrinsics)
        f2_2d = self._project_point(finger2_3d, intrinsics)

        if f1_2d is not None and f2_2d is not None:
            # Gripper jaw bar
            cv2.line(canvas, f1_2d, f2_2d, (0, 255, 255), 2)
            # Gripper fingertips
            cv2.circle(canvas, f1_2d, 4, (0, 255, 255), -1)
            cv2.circle(canvas, f2_2d, 4, (0, 255, 255), -1)

    def _draw_object_tag(self, canvas: np.ndarray, obj: Object3D, intrinsics: CameraIntrinsics, color: tuple):
        """Displays 3D position and orientation badge next to the object."""
        origin_2d = self._project_point(obj.center_3d, intrinsics)
        if origin_2d is None:
            return

        tx, ty = origin_2d[0] + 15, origin_2d[1] - 10
        p = obj.grasp_pose

        lines = [
            f"{obj.label.upper()} ({obj.confidence:.2f})",
            f"3D: X={obj.center_3d[0]:+.3f}m Y={obj.center_3d[1]:+.3f}m Z={obj.center_3d[2]:.3f}m",
            f"RPY: R={p.rpy.roll_deg:+.1f} P={p.rpy.pitch_deg:+.1f} Y={p.rpy.yaw_deg:+.1f} deg",
            f"Grasp W: {p.grasp_width_mm:.1f}mm | Yaw2D: {p.planar_angle_deg:+.1f} deg",
        ]

        # Draw background bubble
        line_height = 18
        max_w = 260
        total_h = len(lines) * line_height + 10

        tx = max(10, min(canvas.shape[1] - max_w - 10, tx))
        ty = max(total_h + 10, min(canvas.shape[0] - 10, ty))

        overlay = canvas.copy()
        cv2.rectangle(overlay, (tx - 5, ty - total_h + 5), (tx + max_w, ty + 5), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.rectangle(canvas, (tx - 5, ty - total_h + 5), (tx + max_w, ty + 5), color, 1)

        for i, text in enumerate(lines):
            curr_y = ty - total_h + 20 + i * line_height
            t_color = (0, 255, 255) if i == 0 else (240, 240, 240)
            cv2.putText(canvas, text, (tx, curr_y), self.font, 0.38, t_color, 1, cv2.LINE_AA)

    def _draw_hud(
        self,
        canvas: np.ndarray,
        objects_3d: List[Object3D],
        prompt: str,
        fps: Optional[float],
        camera_name: str,
    ):
        """Draws top information banner."""
        h, w = canvas.shape[:2]

        hud_h = 42
        hud_bg = canvas[:hud_h, :].copy()
        cv2.rectangle(hud_bg, (0, 0), (w, hud_h), (15, 15, 20), -1)
        cv2.addWeighted(hud_bg, 0.82, canvas[:hud_h, :], 0.18, 0, canvas[:hud_h, :])
        cv2.line(canvas, (0, hud_h), (w, hud_h), (70, 70, 90), 1)

        fps_str = f"FPS: {fps:.1f}" if fps is not None else "FPS: --"
        cv2.putText(canvas, f"CAMERA: {camera_name} | {fps_str}", (15, 18), self.font, 0.42, (180, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"PROMPT: '{prompt}' | Objects Detected: {len(objects_3d)}", (15, 34), self.font, 0.42, (0, 255, 180), 1, cv2.LINE_AA)

        # Legend on the right
        legend_x = w - 240
        cv2.putText(canvas, "Grasp Axes:", (legend_x, 18), self.font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, "X: Close(R)", (legend_x + 75, 18), self.font, 0.38, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Y: Binormal(G)", (legend_x + 75, 34), self.font, 0.38, (0, 220, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Z: Approach(B)", (legend_x + 160, 18), self.font, 0.38, (255, 100, 0), 1, cv2.LINE_AA)
