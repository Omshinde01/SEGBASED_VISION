"""
Grounded-SAM Detector: Combines Grounding DINO (zero-shot object detection)
and SAM (Segment Anything Model) for prompt-based instance segmentation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch
from PIL import Image
import cv2

try:
    from transformers import (
        AutoProcessor,
        AutoModelForZeroShotObjectDetection,
        SamModel,
        SamProcessor,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class DetectionResult:
    """Represents a single detected and segmented object."""
    label: str
    confidence: float
    bbox_xyxy: List[float]  # [x1, y1, x2, y2] in pixels
    mask: np.ndarray        # (H, W) uint8 binary mask (0 or 255)
    area_px: int = 0
    center_px: Tuple[int, int] = (0, 0)
    extra: Dict[str, Any] = field(default_factory=dict)


class GroundedSAMDetector:
    """
    Open-vocabulary detector combining Grounding DINO and SAM.
    """

    def __init__(
        self,
        dino_model_id: str = "IDEA-Research/grounding-dino-base",
        sam_model_id: str = "facebook/sam-vit-base",
        device: Optional[str] = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers is not installed. Please install transformers torch torchvision.")

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.dino_model_id = dino_model_id
        self.sam_model_id = sam_model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.dino_processor = None
        self.dino_model = None
        self.sam_processor = None
        self.sam_model = None

        self._load_models()

    def _load_models(self):
        """Loads Grounding DINO and SAM models onto the target device."""
        print(f"[Grounded-SAM] Loading Grounding DINO from {self.dino_model_id} on {self.device}...")
        self.dino_processor = AutoProcessor.from_pretrained(self.dino_model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.dino_model_id).to(self.device)
        self.dino_model.eval()

        print(f"[Grounded-SAM] Loading SAM from {self.sam_model_id} on {self.device}...")
        self.sam_processor = SamProcessor.from_pretrained(self.sam_model_id)
        self.sam_model = SamModel.from_pretrained(self.sam_model_id).to(self.device)
        self.sam_model.eval()
        print("[Grounded-SAM] Models successfully loaded and initialized!")

    def detect_and_segment(
        self,
        image_bgr: np.ndarray,
        text_prompt: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> List[DetectionResult]:
        """
        Runs Grounding DINO with text_prompt, then runs SAM for each detected box.

        Args:
            image_bgr: BGR image from camera (H, W, 3)
            text_prompt: User text prompt, e.g. "cup", "bottle", "red box"
            box_threshold: Detection threshold
            text_threshold: Text alignment threshold

        Returns:
            List of DetectionResult instances
        """
        if not text_prompt or not text_prompt.strip():
            return []

        # Grounding DINO works best with period at the end
        prompt = text_prompt.strip()
        if not prompt.endswith("."):
            prompt += "."

        b_thresh = box_threshold if box_threshold is not None else self.box_threshold
        t_thresh = text_threshold if text_threshold is not None else self.text_threshold

        # Convert OpenCV BGR to PIL RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        h, w = image_bgr.shape[:2]

        # 1. Grounding DINO Inference
        inputs = self.dino_processor(images=pil_img, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        # Post-process bounding boxes
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=b_thresh,
            text_threshold=t_thresh,
            target_sizes=[(h, w)]
        )

        detections = results[0]
        boxes = detections["boxes"].cpu().numpy()
        scores = detections["scores"].cpu().numpy()
        labels = detections.get("text_labels", detections.get("labels", []))

        if len(boxes) == 0:
            return []

        # 2. SAM Inference using detected bounding boxes
        # Format boxes for SAM: [[[[x1, y1, x2, y2], ...]]]
        boxes_list = boxes.tolist()
        sam_boxes = [[box] for box in boxes_list]

        sam_inputs = self.sam_processor(
            pil_img,
            input_boxes=[sam_boxes],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs)

        masks_tensor = self.sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu()
        )

        # masks_tensor has shape [batch, num_boxes, num_masks, H, W]
        # SAM produces 3 candidate masks per box by default, index 0 is usually highest IoU score
        detected_objects: List[DetectionResult] = []
        box_masks = masks_tensor[0]  # shape: (num_boxes, 3, H, W)

        for i in range(len(boxes)):
            box = boxes[i].tolist()
            score = float(scores[i])
            label_name = labels[i] if i < len(labels) else text_prompt

            # Best mask for box i
            mask_bool = box_masks[i, 0].numpy()
            mask_uint8 = (mask_bool * 255).astype(np.uint8)

            area = int(np.count_nonzero(mask_uint8))
            if area < 20:  # Skip degenerate small masks
                continue

            # Compute mask centroid
            moments = cv2.moments(mask_uint8)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx = int((box[0] + box[2]) / 2.0)
                cy = int((box[1] + box[3]) / 2.0)

            res = DetectionResult(
                label=label_name,
                confidence=score,
                bbox_xyxy=[float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                mask=mask_uint8,
                area_px=area,
                center_px=(cx, cy)
            )
            detected_objects.append(res)

        # Sort by confidence descending
        detected_objects.sort(key=lambda d: d.confidence, reverse=True)
        return detected_objects
