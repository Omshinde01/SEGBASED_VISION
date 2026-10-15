"""
3D Perception & Grasping Pipeline with Orbbec Gemini E and Grounded-SAM.
"""

from .pipeline import PerceptionPipeline
from .camera.orbbec_camera import OrbbecGeminiCamera, MockRGBDCamera, CameraIntrinsics, create_camera
from .detector.grounded_sam import GroundedSAMDetector, DetectionResult
from .geometry.grasp_estimator import GraspEstimator, Object3D, GraspPose, OrientationRPY
from .visualizer import PerceptionVisualizer

__version__ = "0.1.0"
__all__ = [
    "PerceptionPipeline",
    "OrbbecGeminiCamera",
    "MockRGBDCamera",
    "CameraIntrinsics",
    "create_camera",
    "GroundedSAMDetector",
    "DetectionResult",
    "GraspEstimator",
    "Object3D",
    "GraspPose",
    "OrientationRPY",
    "PerceptionVisualizer",
]
