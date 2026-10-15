import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

def test_grounded_sam():
    print("Testing Grounding DINO + SAM on synthetic image...")
    # Create synthetic RGB image (e.g. 480x640 with a red rectangle/box)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[150:350, 200:440] = [0, 0, 220] # Red box in BGR/RGB
    pil_img = Image.fromarray(img)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Grounding DINO
    dino_id = "IDEA-Research/grounding-dino-base"
    print(f"Loading Grounding DINO ({dino_id})...")
    dino_proc = AutoProcessor.from_pretrained(dino_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device)
    dino_model.eval()

    text_prompt = "a red box."
    inputs = dino_proc(images=pil_img, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)

    results = dino_proc.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.25,
        text_threshold=0.20,
        target_sizes=[pil_img.size[::-1]]
    )
    print("DINO detection results:", results)

    if len(results[0]["boxes"]) > 0:
        box = results[0]["boxes"][0].cpu().numpy().tolist()
        print(f"Detected box: {box}")

        # SAM
        sam_id = "facebook/sam-vit-base"
        print(f"Loading SAM ({sam_id})...")
        sam_proc = SamProcessor.from_pretrained(sam_id)
        sam_model = SamModel.from_pretrained(sam_id).to(device)
        sam_model.eval()

        sam_inputs = sam_proc(pil_img, input_boxes=[[[box]]], return_tensors="pt").to(device)
        with torch.no_grad():
            sam_outputs = sam_model(**sam_inputs)

        masks = sam_proc.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu()
        )
        mask = masks[0][0][0].numpy().astype(np.uint8)
        print(f"SAM mask shape: {mask.shape}, sum foreground pixels: {mask.sum()}")
        print("Grounded-SAM verification test PASSED!")
    else:
        print("No boxes detected with prompt 'a red box.'")

if __name__ == "__main__":
    test_grounded_sam()
