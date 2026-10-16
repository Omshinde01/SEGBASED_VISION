"""
command_parser.py - Interprets natural language pick-and-place instructions.
Examples:
  "pick up the cup and place it near laptop"
  "grasp bottle and put it on table"
  "move the red can next to keyboard"
  "cup" (pure detection mode)
"""

import re
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class PickAndPlaceTask:
    """Structured pick and place task parsed from user prompt."""
    raw_prompt: str
    is_pick_and_place: bool
    source_object: str          # e.g. "cup"
    target_object: Optional[str]# e.g. "laptop"
    target_surface: str         # e.g. "table", "desk", or "surface"
    relation: str               # "near", "next to", "on", "beside"

    def get_grounding_prompts(self) -> List[str]:
        """Returns list of distinct object queries needed for Grounding DINO."""
        prompts = [self.source_object]
        if self.target_object and self.target_object not in prompts:
            prompts.append(self.target_object)
        return prompts

    def get_combined_dino_prompt(self) -> str:
        """Returns period-separated text prompt for Grounding DINO."""
        items = [self.source_object]
        if self.target_object and self.target_object != self.source_object:
            items.append(self.target_object)
        # Also append table/desk to segment supporting surface
        items.append("table")
        return " . ".join(items) + " ."


class CommandParser:
    """Parses natural voice or text manipulation requests."""

    # Patterns for: "pick up [A] and place/put/keep/move it near/on [B]"
    PATTERNS = [
        # pick up X and place it near/on/beside Y
        re.compile(r"(?:pick\s*up|grasp|take|grab)\s+(?:the\s+)?(?P<source>[\w\s]+?)\s+and\s+(?:place|put|keep|move|drop)\s+(?:it\s+)?(?P<rel>near|next\s*to|beside|on\s*top\s*of|on)\s+(?:the\s+)?(?P<target>[\w\s]+)", re.IGNORECASE),
        # move X near/next to/on Y
        re.compile(r"(?:move|transfer)\s+(?:the\s+)?(?P<source>[\w\s]+?)\s+(?:to\s+)?(?P<rel>near|next\s*to|beside|on\s*top\s*of|on)\s+(?:the\s+)?(?P<target>[\w\s]+)", re.IGNORECASE),
        # put X near Y
        re.compile(r"(?:put|place|keep)\s+(?:the\s+)?(?P<source>[\w\s]+?)\s+(?P<rel>near|next\s*to|beside|on\s*top\s*of|on)\s+(?:the\s+)?(?P<target>[\w\s]+)", re.IGNORECASE),
    ]

    # Surface synonyms
    SURFACES = {"table", "desk", "counter", "tray", "ground", "bench", "stand", "surface"}

    def parse(self, text: str) -> PickAndPlaceTask:
        cleaned = text.strip()

        for pattern in self.PATTERNS:
            match = pattern.search(cleaned)
            if match:
                groups = match.groupdict()
                src = groups.get("source", "").strip().lower()
                target = groups.get("target", "").strip().lower()
                rel = groups.get("rel", "near").strip().lower()

                # Clean up extraneous words ("a ", "the ", "my ")
                src = re.sub(r"^(?:a|an|the|my|this)\s+", "", src)
                target = re.sub(r"^(?:a|an|the|my|this)\s+", "", target)

                # Determine if target is directly a surface or an object on a surface
                if target in self.SURFACES:
                    surf = target
                    obj_target = None
                else:
                    surf = "table"
                    obj_target = target

                return PickAndPlaceTask(
                    raw_prompt=cleaned,
                    is_pick_and_place=True,
                    source_object=src,
                    target_object=obj_target,
                    target_surface=surf,
                    relation=rel,
                )

        # Fallback: single object detection query (e.g. "cup" or "laptop")
        clean_single = re.sub(r"^(?:a|an|the|this)\s+", "", cleaned.lower())
        return PickAndPlaceTask(
            raw_prompt=cleaned,
            is_pick_and_place=False,
            source_object=clean_single,
            target_object=None,
            target_surface="table",
            relation="none",
        )
