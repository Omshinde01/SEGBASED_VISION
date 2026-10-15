
"""
Grounded-SAM Detector with dual-GPU inference.

GPU 0: Grounding DINO
GPU 1: SAM

Automatically falls back to one GPU or CPU if needed.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import time

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
    """Detected and segmented object."""
    label: str
    confidence: float
    bbox_xyxy: List[float]
    mask: np.ndarray
    area_px: int = 0
    center_px: Tuple[int, int] = (0, 0)
    extra: Dict[str, Any] = field(default_factory=dict)


class GroundedSAMDetector:
    """
    Grounding DINO + SAM with independent GPU placement.

    Default on a two-GPU Kaggle runtime:
        Grounding DINO -> cuda:0
        SAM            -> cuda:1
    """

    def __init__(
        self,
        dino_model_id: str = "IDEA-Research/grounding-dino-base",
        sam_model_id: str = "facebook/sam-vit-base",
        device: Optional[str] = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        dino_device: Optional[str] = None,
        sam_device: Optional[str] = None,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers is not installed. Install transformers, "
                "torch and torchvision."
            )

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.dino_model_id = dino_model_id
        self.sam_model_id = sam_model_id

        cuda_count = torch.cuda.device_count()

        # Preserve compatibility with the existing server, which passes
        # device="cuda". Split the models automatically when possible.
        if dino_device is not None or sam_device is not None:
            fallback = device or (
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
            self.dino_device = dino_device or fallback
            self.sam_device = sam_device or fallback

        elif cuda_count >= 2 and device in (None, "cuda"):
            self.dino_device = "cuda:0"
            self.sam_device = "cuda:1"

        else:
            fallback = device or (
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
            if fallback == "cuda" and not torch.cuda.is_available():
                fallback = "cpu"

            self.dino_device = fallback
            self.sam_device = fallback

        self.device = self.dino_device  # Backward compatibility

        self.dino_processor = None
        self.dino_model = None
        self.sam_processor = None
        self.sam_model = None

        self._load_models()

    def _load_models(self):
        """Load each model onto its assigned device."""

        print(
            f"[Grounded-SAM] Loading Grounding DINO on "
            f"{self.dino_device}..."
        )

        self.dino_processor = AutoProcessor.from_pretrained(
            self.dino_model_id
        )
        self.dino_model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(self.dino_model_id)
            .to(self.dino_device)
            .eval()
        )

        print(
            f"[Grounded-SAM] Loading SAM on {self.sam_device}..."
        )

        self.sam_processor = SamProcessor.from_pretrained(
            self.sam_model_id
        )
        self.sam_model = (
            SamModel.from_pretrained(self.sam_model_id)
            .to(self.sam_device)
            .eval()
        )

        print("[Grounded-SAM] Models loaded successfully.")
        self.print_gpu_status()

    @staticmethod
    def _move_batch(batch, device):
        """Move processor-generated tensors to the target device."""
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    @staticmethod
    def _sync(device):
        """Wait for queued GPU work before measuring latency."""
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))

    def print_gpu_status(self):
        """Display model placement and GPU memory usage."""
        print(
            f"[GPU assignment] DINO={self.dino_device}, "
            f"SAM={self.sam_device}"
        )

        if not torch.cuda.is_available():
            print("[GPU] CUDA unavailable; running on CPU.")
            return

        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(
                f"[GPU {i}] {torch.cuda.get_device_name(i)} | "
                f"allocated={allocated:.2f} GB | "
                f"reserved={reserved:.2f} GB"
            )

    def detect_and_segment(
        self,
        image_bgr: np.ndarray,
        text_prompt: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> List[DetectionResult]:

        if not text_prompt or not text_prompt.strip():
            return []

        total_start = time.perf_counter()

        prompt = text_prompt.strip()
        if not prompt.endswith("."):
            prompt += "."

        b_thresh = (
            box_threshold
            if box_threshold is not None
            else self.box_threshold
        )
        t_thresh = (
            text_threshold
            if text_threshold is not None
            else self.text_threshold
        )

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        h, w = image_bgr.shape[:2]

        # 1. Grounding DINO on GPU 0.
        dino_start = time.perf_counter()

        inputs = self.dino_processor(
            images=pil_img,
            text=prompt,
            return_tensors="pt",
        )
        inputs = self._move_batch(inputs, self.dino_device)

        with torch.inference_mode():
            outputs = self.dino_model(**inputs)

        self._sync(self.dino_device)
        dino_ms = (time.perf_counter() - dino_start) * 1000

        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=b_thresh,
            text_threshold=t_thresh,
            target_sizes=[(h, w)],
        )

        detections = results[0]
        boxes = detections["boxes"].detach().cpu().numpy()
        scores = detections["scores"].detach().cpu().numpy()
        labels = detections.get(
            "text_labels", detections.get("labels", [])
        )

        if len(boxes) == 0:
            print(
                f"[Inference] No detections | "
                f"DINO={dino_ms:.1f} ms"
            )
            return []

        # 2. Prepare SAM box prompts from DINO output.
        # DINO outputs are converted to CPU lists, then the SAM processor
        # creates new tensors which are explicitly transferred to GPU 1.
        boxes_list = boxes.tolist()
        sam_boxes = [[box] for box in boxes_list]

        sam_inputs = self.sam_processor(
            images=pil_img,
            input_boxes=[sam_boxes],
            return_tensors="pt",
        )
        sam_inputs = self._move_batch(sam_inputs, self.sam_device)

        # 3. SAM on GPU 1.
        sam_start = time.perf_counter()

        with torch.inference_mode():
            sam_outputs = self.sam_model(**sam_inputs)

        self._sync(self.sam_device)
        sam_ms = (time.perf_counter() - sam_start) * 1000

        masks_tensor = self.sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.detach().cpu(),
            sam_inputs["original_sizes"].detach().cpu(),
            sam_inputs["reshaped_input_sizes"].detach().cpu(),
        )

        box_masks = masks_tensor[0]
        detected_objects: List[DetectionResult] = []

        for i in range(len(boxes)):
            box = boxes[i].tolist()
            score = float(scores[i])
            label_name = (
                labels[i] if i < len(labels) else text_prompt
            )

            # Preserve the existing behavior: use the first SAM mask.
            mask_bool = box_masks[i, 0].numpy()
            mask_uint8 = (mask_bool * 255).astype(np.uint8)

            area = int(np.count_nonzero(mask_uint8))
            if area < 20:
                continue

            moments = cv2.moments(mask_uint8)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)

            detected_objects.append(
                DetectionResult(
                    label=label_name,
                    confidence=score,
                    bbox_xyxy=[float(v) for v in box],
                    mask=mask_uint8,
                    area_px=area,
                    center_px=(cx, cy),
                    extra={
                        "dino_device": self.dino_device,
                        "sam_device": self.sam_device,
                    },
                )
            )

        detected_objects.sort(
            key=lambda detection: detection.confidence,
            reverse=True,
        )

        total_ms = (time.perf_counter() - total_start) * 1000

        print(
            f"[Inference] DINO={dino_ms:.1f} ms | "
            f"SAM={sam_ms:.1f} ms | "
            f"total={total_ms:.1f} ms | "
            f"detections={len(detected_objects)} | "
            f"DINO device={self.dino_device} | "
            f"SAM device={self.sam_device}"
        )

        return detected_objects
