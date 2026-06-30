"""
Logger module — writes every intermediate result to a run folder for traceability.

Non-negotiable per spec: every VLM prompt+response, every JoyAI input+output,
every intermediate matte, and every reconstruction must be logged.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from PIL import Image


class RunLogger:
    """Per-run logger that writes to a timestamped folder."""

    def __init__(self, output_dir: str | Path = "runs",
                 image_prefix: str = "") -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if image_prefix:
            self.run_dir = Path(output_dir) / f"{image_prefix}_{timestamp}"
        else:
            self.run_dir = Path(output_dir) / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._step_counter = 0

    def _next_step(self, prefix: str = "step") -> str:
        self._step_counter += 1
        return f"{self._step_counter:04d}_{prefix}"

    def save_scene_graph(self, graph, label: str = "scene_graph") -> Path:
        """Save the current SceneGraph as JSON."""
        path = self.run_dir / f"{self._next_step(label)}.json"
        graph.save(path)
        return path

    def save_image(self, image: Image.Image, label: str = "image") -> Path:
        """Save a PIL Image to the run folder."""
        path = self.run_dir / f"{self._next_step(label)}.png"
        image.save(path)
        return path

    def log_vlm_call(self, role: str, system_prompt: str, user_text: str,
                     response: str, image_label: str = "") -> Path:
        """Log a VLM call: prompt, response, metadata."""
        path = self.run_dir / f"{self._next_step(f'vlm_{role}')}.json"
        data = {
            "role": role,
            "image_label": image_label,
            "system_prompt": system_prompt,
            "user_text": user_text,
            "response": response,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    def log_image_pair(self, ref: Image.Image, result: Image.Image,
                       label: str = "compare") -> tuple[Path, Path]:
        """Log a reference-result image pair for verification."""
        ref_path = self.run_dir / f"{self._next_step(f'{label}_ref')}.png"
        res_path = self.run_dir / f"{self._next_step(f'{label}_result')}.png"
        ref.save(ref_path)
        result.save(res_path)
        return ref_path, res_path

    def log_text(self, text: str, label: str = "note") -> Path:
        """Log arbitrary text."""
        path = self.run_dir / f"{self._next_step(label)}.txt"
        Path(path).write_text(text, encoding="utf-8")
        return path

    @property
    def output_dir(self) -> Path:
        return self.run_dir