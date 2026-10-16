"""
config_loader.py - Loads and validates config.yaml with sensible fallbacks.
"""

from pathlib import Path
from typing import Dict, Any
import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "camera": {
        "type": "auto",
        "color_width": 640,
        "color_height": 480,
        "fps": 30,
        "align_depth": True,
        "depth_scale": 0.001,
    },
    "models": {
        "dino_model_id": "IDEA-Research/grounding-dino-base",
        "sam_model_id": "facebook/sam-vit-base",
        "device": "auto",
        "box_threshold": 0.25,
        "text_threshold": 0.20,
        "min_mask_area_px": 25,
    },
    "grasp": {
        "min_depth_m": 0.15,
        "max_depth_m": 3.50,
        "depth_percentile_low": 5.0,
        "depth_percentile_high": 95.0,
        "standoff_distance_m": 0.10,
        "gripper_finger_clearance_mm": 15.0,
        "min_points_threshold": 15,
    },
    "placement": {
        "surface_prompt": "table . desk . surface .",
        "ransac_distance_threshold_m": 0.015,
        "ransac_max_iterations": 1500,
        "grid_search_step_m": 0.04,
        "free_space_radius_m": 0.12,
        "target_standoff_distance_m": 0.22,
        "preferred_sides": ["right", "front", "left"],
        "max_placement_distance_m": 0.50,
        "height_offset_m": 0.02,
    },
    "nlp": {
        "default_action": "pick_and_place",
        "fallback_surface": "table",
    },
    "visualization": {
        "axis_length_m": 0.065,
        "mask_alpha": 0.45,
        "show_surface_grid": True,
        "window_name": "Perception & Pick-and-Place (3D Grasp + Placement Pose)",
    },
}


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads configuration from YAML file, merging with default values."""
    path = Path(config_path)
    if not path.is_file():
        # Try finding config.yaml in repo root if path relative
        repo_root = Path(__file__).resolve().parent.parent
        alt_path = repo_root / "config.yaml"
        if alt_path.is_file():
            path = alt_path

    if not path.is_file():
        print(f"[Config] Warning: {config_path} not found. Using default internal config.")
        return DEFAULT_CONFIG.copy()

    try:
        with open(path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}

        # Deep merge user_config on top of DEFAULT_CONFIG
        merged = deep_merge(DEFAULT_CONFIG, user_config)
        return merged
    except Exception as e:
        print(f"[Config] Error reading {path} ({e}). Falling back to defaults.")
        return DEFAULT_CONFIG.copy()


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges override dictionary into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result
