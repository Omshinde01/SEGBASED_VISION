"""
Orbbec Gemini E Camera Interface and Mock RGB-D Camera.
Provides aligned Color and Depth frames, depth scale, and intrinsic matrix.
"""

import time
import cv2
import numpy as np

try:
    import pyorbbecsdk
    from pyorbbecsdk import (
        Pipeline,
        Config,
        OBSensorType,
        OBFormat,
        OBAlignMode,
        AlignFilter,
        OBStreamType
    )
    ORBBEC_AVAILABLE = True
except ImportError:
    pyorbbecsdk = None
    ORBBEC_AVAILABLE = False


class CameraIntrinsics:
    """Camera intrinsic parameters."""
    def __init__(self, fx: float, fy: float, cx: float, cy: float, width: int, height: int):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = int(width)
        self.height = int(height)

    @property
    def K(self) -> np.ndarray:
        """3x3 Intrinsic matrix."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def to_dict(self) -> dict:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height
        }


class OrbbecGeminiCamera:
    """
    Driver for Orbbec Gemini E RGB-D Camera using pyorbbecsdk.
    Enables hardware or software depth-to-color alignment and robust frame decoding.
    """

    def __init__(self, color_width: int = 640, color_height: int = 480, fps: int = 30):
        if not ORBBEC_AVAILABLE:
            raise RuntimeError("pyorbbecsdk is not installed or available.")

        self.color_width = color_width
        self.color_height = color_height
        self.fps = fps

        self.pipeline = None
        self.align_filter = None
        self.depth_scale = 1.0
        self.intrinsics = None
        self.is_running = False

        # Frame buffers to gracefully handle interleaved stream frames
        self.latest_color = None
        self.latest_depth = None

    def start(self):
        """Initializes and starts the Orbbec pipeline with Color + Depth."""
        ctx = pyorbbecsdk.Context()
        dev_list = ctx.query_devices()
        if dev_list.get_count() == 0:
            raise RuntimeError("No Orbbec camera device found. Please connect your Orbbec Gemini E.")

        self.pipeline = Pipeline()
        config = Config()

        # Find Color profile (prefer uncompressed RGB 640x480 for USB2/3 stability)
        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if color_profiles is None or color_profiles.get_count() == 0:
            raise RuntimeError("No Color sensor profile found on Orbbec device.")

        color_profile = None
        try:
            color_profile = color_profiles.get_video_stream_profile(
                self.color_width, self.color_height, OBFormat.RGB, self.fps
            )
        except Exception:
            pass

        if color_profile is None:
            color_profile = color_profiles.get_default_video_stream_profile()

        # Find Depth profile (match resolution if possible)
        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if depth_profiles is None or depth_profiles.get_count() == 0:
            raise RuntimeError("No Depth sensor profile found on Orbbec device.")

        depth_profile = None
        for fmt in (OBFormat.Y11, OBFormat.Y12, OBFormat.Y16):
            try:
                depth_profile = depth_profiles.get_video_stream_profile(
                    self.color_width, self.color_height, fmt, self.fps
                )
                if depth_profile is not None:
                    break
            except Exception:
                pass

        if depth_profile is None:
            depth_profile = depth_profiles.get_default_video_stream_profile()

        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)

        # Attempt alignment
        try:
            config.set_align_mode(OBAlignMode.SW_MODE)
        except Exception:
            pass

        self.pipeline.start(config)
        self.is_running = True

        # Extract intrinsics from color profile
        try:
            intr = color_profile.get_intrinsic()
            self.intrinsics = CameraIntrinsics(
                fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy,
                width=color_profile.get_width(), height=color_profile.get_height()
            )
        except Exception:
            w = color_profile.get_width()
            h = color_profile.get_height()
            self.intrinsics = CameraIntrinsics(
                fx=w * 0.9, fy=h * 0.9, cx=w / 2.0, cy=h / 2.0, width=w, height=h
            )

        # Camera warm-up loop: discard initial invalid frames
        print("[Camera] Warming up sensor streams...")
        t_start = time.time()
        while time.time() - t_start < 2.0:
            frames = self.pipeline.wait_for_frames(200)
            if frames is not None:
                cf = frames.get_color_frame()
                df = frames.get_depth_frame()
                if cf is not None:
                    c_img = self._parse_color_frame(cf)
                    if c_img is not None:
                        self.latest_color = c_img
                if df is not None:
                    d_img, scale = self._parse_depth_frame(df)
                    if d_img is not None:
                        self.latest_depth = d_img
                        self.depth_scale = scale
            if self.latest_color is not None and self.latest_depth is not None:
                break
        print(f"[Camera] Orbbec Gemini E ready! Resolution: {color_profile.get_width()}x{color_profile.get_height()}")

    def read(self, timeout_ms: int = 500):
        """
        Reads one synchronized RGB-D frame.
        Returns:
            color_bgr: np.ndarray (H, W, 3) uint8
            depth_m: np.ndarray (H, W) float32 in meters
            intrinsics: CameraIntrinsics
        """
        if not self.is_running or self.pipeline is None:
            return None, None, None

        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is not None:
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame is not None:
                c_img = self._parse_color_frame(color_frame)
                if c_img is not None:
                    self.latest_color = c_img

            if depth_frame is not None:
                d_img, scale = self._parse_depth_frame(depth_frame)
                if d_img is not None:
                    self.latest_depth = d_img
                    self.depth_scale = scale

        if self.latest_color is None or self.latest_depth is None:
            return None, None, None

        # Ensure spatial resolution matches between color and depth
        ch, cw = self.latest_color.shape[:2]
        dh, dw = self.latest_depth.shape[:2]
        depth_m = self.latest_depth
        if (dh, dw) != (ch, cw):
            depth_m = cv2.resize(depth_m, (cw, ch), interpolation=cv2.INTER_NEAREST)

        return self.latest_color.copy(), depth_m.copy(), self.intrinsics

    def _parse_color_frame(self, frame) -> np.ndarray:
        w, h = frame.get_width(), frame.get_height()
        raw_bytes = bytes(frame.get_data())
        if not raw_bytes:
            return None

        # Check if raw RGB array (size == w * h * 3)
        # OBFormat.RGB delivers bytes in R-G-B order; convert to OpenCV BGR
        if len(raw_bytes) >= w * h * 3:
            raw = np.frombuffer(raw_bytes[:w * h * 3], dtype=np.uint8).reshape(h, w, 3)
            bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            return bgr

        # Check for MJPG / JPEG stream
        if len(raw_bytes) > 10 and raw_bytes[:2] == b'\xff\xd8':
            img = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            return img

        # Try imdecode fallback if size is reasonable
        if len(raw_bytes) > 2000:
            try:
                img = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            except Exception:
                pass

        return None

    def _parse_depth_frame(self, frame):
        """Extracts depth buffer and converts to meters (float32)."""
        w, h = frame.get_width(), frame.get_height()
        raw_bytes = bytes(frame.get_data())
        if len(raw_bytes) < w * h * 2:
            return None, 1.0

        scale = float(frame.get_depth_scale())
        depth_raw = np.frombuffer(raw_bytes[:w * h * 2], dtype=np.uint16).reshape(h, w)

        # Convert to meters
        if scale > 0.05:
            depth_m = depth_raw.astype(np.float32) * (scale / 1000.0)
        else:
            depth_m = depth_raw.astype(np.float32) * scale

        return depth_m, scale

    def stop(self):
        """Stops the pipeline."""
        if self.pipeline is not None and self.is_running:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.is_running = False


class MockRGBDCamera:
    """
    Synthetic RGB-D camera generator for testing the perception pipeline
    when physical Orbbec camera is not connected.
    Generates realistic desktop scene with objects (e.g. cup, bottle, box) and planar depth.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.intrinsics = CameraIntrinsics(
            fx=580.0, fy=580.0, cx=width / 2.0, cy=height / 2.0,
            width=width, height=height
        )
        self.frame_idx = 0

    def start(self):
        pass

    def read(self, timeout_ms: int = 1000):
        self.frame_idx += 1
        time.sleep(0.03)

        # Create tabletop background
        color = np.ones((self.height, self.width, 3), dtype=np.uint8) * 210

        # Gradient depth for tabletop surface: distance slopes from 0.95m at top to 0.60m at bottom
        y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]
        depth_m = 0.95 - (y_coords / self.height) * 0.35

        # Object 1: Coffee Cup (Blue cylinder)
        cup_x1, cup_y1, cup_x2, cup_y2 = 180, 220, 260, 340
        cv2.rectangle(color, (cup_x1, cup_y1), (cup_x2, cup_y2), (180, 60, 40), -1)
        cv2.ellipse(color, ((cup_x1 + cup_x2) // 2, cup_y1), ((cup_x2 - cup_x1) // 2, 10), 0, 0, 360, (210, 80, 50), -1)
        cv2.ellipse(color, (cup_x1 - 10, (cup_y1 + cup_y2) // 2), (12, 25), 0, 0, 360, (180, 60, 40), 5)
        cv2.putText(color, "CUP", (cup_x1 + 10, cup_y2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        depth_m[cup_y1:cup_y2, cup_x1:cup_x2] = 0.65

        # Object 2: Bottle (Green cylinder)
        bot_x1, bot_y1, bot_x2, bot_y2 = 360, 160, 430, 350
        cv2.rectangle(color, (bot_x1, bot_y1 + 40), (bot_x2, bot_y2), (40, 160, 50), -1)
        cv2.rectangle(color, (bot_x1 + 15, bot_y1), (bot_x2 - 15, bot_y1 + 40), (50, 180, 60), -1)
        cv2.putText(color, "BOTTLE", (bot_x1 + 5, bot_y2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        depth_m[bot_y1:bot_y2, bot_x1:bot_x2] = 0.68

        # Object 3: Laptop (Grey clam-shell laptop next to cup)
        lap_x1, lap_y1, lap_x2, lap_y2 = 420, 200, 580, 330
        cv2.rectangle(color, (lap_x1, lap_y1), (lap_x2, lap_y2), (70, 70, 75), -1)
        cv2.rectangle(color, (lap_x1 + 10, lap_y1 + 10), (lap_x2 - 10, lap_y2 - 35), (20, 20, 25), -1)
        cv2.rectangle(color, (lap_x1, lap_y2 - 30), (lap_x2, lap_y2), (100, 100, 105), -1)
        cv2.putText(color, "LAPTOP", (lap_x1 + 30, lap_y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        depth_m[lap_y1:lap_y2, lap_x1:lap_x2] = 0.58

        noise = np.random.normal(0, 0.002, depth_m.shape).astype(np.float32)
        depth_m = np.clip(depth_m + noise, 0.1, 5.0)

        return color, depth_m.astype(np.float32), self.intrinsics

    def stop(self):
        pass


def create_camera(prefer_orbbec: bool = True) -> tuple:
    """
    Factory function: tries to connect to Orbbec Gemini E.
    If unavailable or not plugged in, falls back to MockRGBDCamera.
    Returns:
        (camera_instance, is_real_camera: bool)
    """
    if prefer_orbbec and ORBBEC_AVAILABLE:
        try:
            cam = OrbbecGeminiCamera()
            cam.start()
            print("[Camera] Successfully connected to Orbbec Gemini E camera!")
            return cam, True
        except Exception as e:
            print(f"[Camera] Orbbec Gemini E not detected ({e}). Using Mock RGB-D Camera.")
    else:
        print("[Camera] pyorbbecsdk not available or prefer_orbbec is False. Using Mock RGB-D Camera.")

    cam = MockRGBDCamera()
    cam.start()
    return cam, False
