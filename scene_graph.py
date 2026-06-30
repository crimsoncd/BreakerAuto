"""
Scene Graph — the central data contract shared across all pipeline stages.

Every stage reads and writes this ONE shared object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ElementStatus(Enum):
    PLANNED = "planned"
    EXTRACTING = "extracting"
    DONE = "done"
    FAILED = "failed"


class BackgroundStatus(Enum):
    PLANNED = "planned"
    GENERATING = "generating"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Element:
    """One element/layer extracted from the illustration."""

    id: str                                 # stable unique handle, e.g. "girl_01"
    name: str                               # object-level label
    bbox: list[int]                         # [xmin, ymin, xmax, ymax] normalized to 0-1000
    depth_rank: int                         # 1 = frontmost
    overlaps: list[str] = field(default_factory=list)  # names of elements this one overlaps
    isolation_prompt: Optional[str] = None
    layer_path: Optional[str] = None       # final RGBA cutout
    status: ElementStatus = ElementStatus.PLANNED
    attempts: int = 0
    defects: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "bbox": self.bbox,
            "depth_rank": self.depth_rank,
            "overlaps": self.overlaps,
            "isolation_prompt": self.isolation_prompt,
            "layer_path": self.layer_path,
            "status": self.status.value,
            "attempts": self.attempts,
            "defects": self.defects,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Element":
        return cls(
            id=d["id"],
            name=d["name"],
            bbox=d["bbox"],
            depth_rank=d["depth_rank"],
            overlaps=d.get("overlaps", []),
            isolation_prompt=d.get("isolation_prompt"),
            layer_path=d.get("layer_path"),
            status=ElementStatus(d.get("status", "planned")),
            attempts=d.get("attempts", 0),
            defects=d.get("defects", []),
        )


@dataclass
class Background:
    """Background layer information."""

    prompt: Optional[str] = None
    image_path: Optional[str] = None
    status: BackgroundStatus = BackgroundStatus.PLANNED
    attempts: int = 0
    defects: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "image_path": self.image_path,
            "status": self.status.value,
            "attempts": self.attempts,
            "defects": self.defects,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Background":
        return cls(
            prompt=d.get("prompt"),
            image_path=d.get("image_path"),
            status=BackgroundStatus(d.get("status", "planned")),
            attempts=d.get("attempts", 0),
            defects=d.get("defects", []),
        )


@dataclass
class SceneGraph:
    """The central shared data object for the entire pipeline."""

    image_path: str
    image_size: tuple[int, int]             # (W, H)
    background: Background = field(default_factory=Background)
    elements: list[Element] = field(default_factory=list)
    global_attempts: int = 0
    enum_reopenings: int = 0                # count of Stage-1 reopenings

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "image_size": list(self.image_size),
            "background": self.background.to_dict(),
            "elements": [e.to_dict() for e in self.elements],
            "global_attempts": self.global_attempts,
            "enum_reopenings": self.enum_reopenings,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SceneGraph":
        bg = Background.from_dict(d.get("background", {}))
        elements = [Element.from_dict(e) for e in d.get("elements", [])]
        return cls(
            image_path=d["image_path"],
            image_size=tuple(d["image_size"]),
            background=bg,
            elements=elements,
            global_attempts=d.get("global_attempts", 0),
            enum_reopenings=d.get("enum_reopenings", 0),
        )

    def save(self, path: str | Path) -> None:
        """Save SceneGraph as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "SceneGraph":
        """Load SceneGraph from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

    def get_element_by_name(self, name: str) -> Optional[Element]:
        """Find element by name."""
        for el in self.elements:
            if el.name == name:
                return el
        return None

    def get_element_by_id(self, elem_id: str) -> Optional[Element]:
        """Find element by id."""
        for el in self.elements:
            if el.id == elem_id:
                return el
        return None

    def sorted_elements(self) -> list[Element]:
        """Return elements sorted by depth_rank (1 = frontmost first, then deeper)."""
        return sorted(self.elements, key=lambda e: e.depth_rank)

    def next_available_id(self) -> str:
        """Generate a unique element id."""
        max_idx = 0
        for el in self.elements:
            # Parse numeric suffix from id like "girl_01"
            parts = el.id.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                max_idx = max(max_idx, int(parts[1]))
        return f"element_{max_idx + 1:02d}"