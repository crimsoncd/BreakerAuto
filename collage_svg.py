#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recursively reconstruct SVG collages from layerwise metadata.json files.

Expected directory structure example:

input_root/
└── B/
    └── B008/
        ├── metadata.json
        ├── B008.png
        ├── B008_background.png
        ├── B008_background.svg
        ├── B008_astronaut.png
        ├── B008_astronaut.svg
        └── ...

Example usage:

    python reconstruct_svg_collage.py \
        --input_root /path/to/dataset

Specify a custom output filename:

    python reconstruct_svg_collage.py \
        --input_root /path/to/dataset \
        --output_name reconstructed.svg

Overwrite existing outputs:

    python reconstruct_svg_collage.py \
        --input_root /path/to/dataset \
        --overwrite
"""

import argparse
import copy
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def svg_tag(name: str) -> str:
    """Return an SVG-namespaced XML tag."""
    return f"{{{SVG_NS}}}{name}"


def parse_length(value: Optional[str]) -> Optional[float]:
    """
    Parse an SVG length such as:
        512
        512px
        10.5pt

    Only the numeric portion is used. Percentage values are not accepted.
    """
    if value is None:
        return None

    value = value.strip()

    if not value or value.endswith("%"):
        return None

    match = re.match(
        r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
        value,
    )
    if match is None:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_viewbox(root: ET.Element) -> Tuple[float, float, float, float]:
    """
    Read the source SVG viewBox.

    If no valid viewBox exists, infer it from width and height.
    """
    viewbox = root.get("viewBox")

    if viewbox:
        values = re.split(r"[,\s]+", viewbox.strip())

        if len(values) == 4:
            try:
                min_x, min_y, width, height = map(float, values)

                if width > 0 and height > 0:
                    return min_x, min_y, width, height
            except ValueError:
                pass

    width = parse_length(root.get("width"))
    height = parse_length(root.get("height"))

    if width is None or width <= 0:
        width = 1.0

    if height is None or height <= 0:
        height = 1.0

    return 0.0, 0.0, width, height


def strip_xml_namespace(tag: str) -> str:
    """Return the local part of an XML tag."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def replace_id_reference(value: str, id_map: Dict[str, str]) -> str:
    """
    Replace SVG ID references, including:

        url(#gradient1)
        #clipPath1
        href="#symbol1"
    """
    if not value:
        return value

    def replace_url(match: re.Match) -> str:
        old_id = match.group(1)
        return f"url(#{id_map.get(old_id, old_id)})"

    value = re.sub(
        r"url\(\s*['\"]?#([^)'\"\s]+)['\"]?\s*\)",
        replace_url,
        value,
    )

    if value.startswith("#"):
        old_id = value[1:]
        value = "#" + id_map.get(old_id, old_id)

    return value


def prefix_svg_ids(root: ET.Element, prefix: str) -> None:
    """
    Prefix IDs in one imported SVG to avoid collisions between layers.

    References such as clip-path="url(#clip0)" and href="#shape0"
    are updated accordingly.
    """
    id_map: Dict[str, str] = {}

    for element in root.iter():
        old_id = element.get("id")

        if old_id:
            new_id = f"{prefix}_{old_id}"
            id_map[old_id] = new_id
            element.set("id", new_id)

    if not id_map:
        return

    reference_attributes = {
        "clip-path",
        "fill",
        "filter",
        "marker-end",
        "marker-mid",
        "marker-start",
        "mask",
        "stroke",
        "style",
        "href",
        f"{{{XLINK_NS}}}href",
    }

    for element in root.iter():
        for attribute_name, attribute_value in list(element.attrib.items()):
            local_name = strip_xml_namespace(attribute_name)

            if (
                attribute_name in reference_attributes
                or local_name in reference_attributes
                or "url(#" in attribute_value
                or attribute_value.startswith("#")
            ):
                element.set(
                    attribute_name,
                    replace_id_reference(attribute_value, id_map),
                )

        if element.text and "url(#" in element.text:
            element.text = replace_id_reference(element.text, id_map)


def resolve_element_svg_path(
    input_root: Path,
    metadata_path: Path,
    relative_png_path: str,
) -> Path:
    """
    Resolve the SVG corresponding to a PNG path.

    Resolution order:
    1. Relative to input_root.
    2. Relative to the directory containing metadata.json.
    """
    relative_path = Path(relative_png_path).with_suffix(".svg")

    candidates = [
        input_root / relative_path,
        metadata_path.parent / relative_path,
        metadata_path.parent / relative_path.name,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    candidate_text = "\n".join(f"  - {path}" for path in candidates)

    raise FileNotFoundError(
        f"Could not find the SVG corresponding to:\n"
        f"  {relative_png_path}\n"
        f"Checked:\n{candidate_text}"
    )


def normalized_bbox_to_pixels(
    bbox: Iterable[float],
    canvas_width: float,
    canvas_height: float,
) -> Tuple[float, float, float, float]:
    """
    Convert a normalized [0,1000] bbox into pixel coordinates.

    bbox format:
        [x1, y1, x2, y2]
    """
    values = list(bbox)

    if len(values) != 4:
        raise ValueError(
            f"bbox must contain exactly four values, got: {values}"
        )

    x1, y1, x2, y2 = map(float, values)

    x = x1 / 1000.0 * canvas_width
    y = y1 / 1000.0 * canvas_height
    width = (x2 - x1) / 1000.0 * canvas_width
    height = (y2 - y1) / 1000.0 * canvas_height

    if width < 0 or height < 0:
        raise ValueError(f"Invalid bbox with negative size: {values}")

    return x, y, width, height


def load_svg(svg_path: Path) -> ET.Element:
    """Parse one SVG file and return a copied root element."""
    try:
        tree = ET.parse(svg_path)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid SVG XML: {svg_path}\n{exc}") from exc

    root = tree.getroot()

    if strip_xml_namespace(root.tag).lower() != "svg":
        raise ValueError(f"Root element is not <svg>: {svg_path}")

    return copy.deepcopy(root)


def append_svg_layer(
    output_root: ET.Element,
    source_svg_path: Path,
    bbox_pixels: Tuple[float, float, float, float],
    element_name: str,
    layer_index: int,
    unique_index: int,
    preserve_aspect_ratio: str,
) -> None:
    """
    Inline one source SVG as a nested SVG layer in the output document.
    """
    source_root = load_svg(source_svg_path)

    unique_prefix = f"layer_{layer_index}_{unique_index}"
    prefix_svg_ids(source_root, unique_prefix)

    source_min_x, source_min_y, source_width, source_height = parse_viewbox(
        source_root
    )

    x, y, width, height = bbox_pixels

    nested_svg = ET.SubElement(
        output_root,
        svg_tag("svg"),
        {
            "id": unique_prefix,
            "x": format_number(x),
            "y": format_number(y),
            "width": format_number(width),
            "height": format_number(height),
            "viewBox": (
                f"{format_number(source_min_x)} "
                f"{format_number(source_min_y)} "
                f"{format_number(source_width)} "
                f"{format_number(source_height)}"
            ),
            "preserveAspectRatio": preserve_aspect_ratio,
            "overflow": "visible",
            "data-name": element_name,
            "data-layer": str(layer_index),
            "data-source": str(source_svg_path),
        },
    )

    # Copy most root-level presentation attributes from the source SVG.
    skipped_attributes = {
        "id",
        "x",
        "y",
        "width",
        "height",
        "viewBox",
        "preserveAspectRatio",
        "version",
        "xmlns",
    }

    for attribute_name, attribute_value in source_root.attrib.items():
        local_name = strip_xml_namespace(attribute_name)

        if local_name not in skipped_attributes:
            nested_svg.set(attribute_name, attribute_value)

    for child in list(source_root):
        nested_svg.append(copy.deepcopy(child))


def format_number(value: float) -> str:
    """Format coordinates without unnecessary trailing zeros."""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))

    return f"{value:.6f}".rstrip("0").rstrip(".")


def indent_xml(element: ET.Element, level: int = 0) -> None:
    """Pretty-print XML for Python versions without ET.indent."""
    indentation = "\n" + level * "  "

    if len(element):
        if not element.text or not element.text.strip():
            element.text = indentation + "  "

        for child in element:
            indent_xml(child, level + 1)

        if not child.tail or not child.tail.strip():
            child.tail = indentation
    else:
        if level and (not element.tail or not element.tail.strip()):
            element.tail = indentation


def reconstruct_one_metadata(
    metadata_path: Path,
    input_root: Path,
    output_name: str,
    overwrite: bool,
    preserve_aspect_ratio: str,
    strict: bool,
) -> Optional[Path]:
    """Build one reconstructed SVG from one metadata.json."""
    output_path = metadata_path.parent / output_name

    if output_path.exists() and not overwrite:
        print(f"[SKIP] Output already exists: {output_path}")
        return None

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    image_size = metadata.get("image_size")

    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
    ):
        raise ValueError(
            f"'image_size' must be [width, height] in {metadata_path}"
        )

    canvas_width = float(image_size[0])
    canvas_height = float(image_size[1])

    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError(
            f"Invalid image_size in {metadata_path}: {image_size}"
        )

    elements = metadata.get("elements")

    if not isinstance(elements, list):
        raise ValueError(
            f"'elements' must be a list in {metadata_path}"
        )

    output_root = ET.Element(
        svg_tag("svg"),
        {
            "width": format_number(canvas_width),
            "height": format_number(canvas_height),
            "viewBox": (
                f"0 0 "
                f"{format_number(canvas_width)} "
                f"{format_number(canvas_height)}"
            ),
            "version": "1.1",
        },
    )

    title = ET.SubElement(output_root, svg_tag("title"))
    title.text = f"SVG reconstruction from {metadata_path.name}"

    # Python sorting is stable, so elements with the same layer preserve
    # their original order in metadata.json.
    sorted_elements = sorted(
        enumerate(elements),
        key=lambda item: (
            float(item[1].get("layer", 0)),
            item[0],
        ),
    )

    imported_count = 0

    for original_index, element in sorted_elements:
        try:
            png_path = element["path"]
            bbox = element["bbox"]
            layer = int(element.get("layer", 0))
            name = str(element.get("name", f"element_{original_index}"))

            source_svg_path = resolve_element_svg_path(
                input_root=input_root,
                metadata_path=metadata_path,
                relative_png_path=png_path,
            )

            bbox_pixels = normalized_bbox_to_pixels(
                bbox=bbox,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )

            append_svg_layer(
                output_root=output_root,
                source_svg_path=source_svg_path,
                bbox_pixels=bbox_pixels,
                element_name=name,
                layer_index=layer,
                unique_index=original_index,
                preserve_aspect_ratio=preserve_aspect_ratio,
            )

            imported_count += 1

        except Exception as exc:
            message = (
                f"[WARNING] Failed to import an element from "
                f"{metadata_path}:\n"
                f"  element index: {original_index}\n"
                f"  element: {element}\n"
                f"  reason: {exc}"
            )

            if strict:
                raise RuntimeError(message) from exc

            print(message, file=sys.stderr)

    indent_xml(output_root)

    tree = ET.ElementTree(output_root)
    tree.write(
        output_path,
        encoding="utf-8",
        xml_declaration=True,
    )

    print(
        f"[OK] {metadata_path} -> {output_path} "
        f"({imported_count}/{len(elements)} elements)"
    )

    return output_path


def find_metadata_files(input_root: Path) -> List[Path]:
    """Recursively find all metadata.json files."""
    return sorted(
        path
        for path in input_root.rglob("metadata.json")
        if path.is_file()
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively collage element SVG files according to "
            "layerwise metadata.json files."
        )
    )

    parser.add_argument(
        "--input_root",
        type=Path,
        required=True,
        help=(
            "Dataset root. Paths stored in metadata.json are resolved "
            "relative to this directory."
        ),
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="reconstructed.svg",
        help=(
            "Output filename placed beside each metadata.json. "
            "Default: reconstructed.svg"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output SVG.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Stop immediately when an SVG file is missing or invalid. "
            "Without this option, invalid elements are skipped."
        ),
    )

    parser.add_argument(
        "--preserve_aspect_ratio",
        choices=["none", "meet", "slice"],
        default="none",
        help=(
            "How each source SVG is fitted into its bbox. "
            "'none' stretches it exactly to the bbox; "
            "'meet' preserves aspect ratio and fits inside; "
            "'slice' preserves aspect ratio and covers the bbox. "
            "Default: none"
        ),
    )

    return parser


def get_preserve_aspect_ratio(mode: str) -> str:
    if mode == "none":
        return "none"

    if mode == "meet":
        return "xMidYMid meet"

    if mode == "slice":
        return "xMidYMid slice"

    raise ValueError(f"Unsupported preserve-aspect-ratio mode: {mode}")


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()

    if not input_root.is_dir():
        parser.error(f"Input root is not a directory: {input_root}")

    if not args.output_name.lower().endswith(".svg"):
        parser.error("--output_name must end with .svg")

    metadata_files = find_metadata_files(input_root)

    if not metadata_files:
        print(
            f"No metadata.json files found under: {input_root}",
            file=sys.stderr,
        )
        return 1

    preserve_aspect_ratio = get_preserve_aspect_ratio(
        args.preserve_aspect_ratio
    )

    success_count = 0
    failed_count = 0
    skipped_count = 0

    for metadata_path in metadata_files:
        try:
            result = reconstruct_one_metadata(
                metadata_path=metadata_path,
                input_root=input_root,
                output_name=args.output_name,
                overwrite=args.overwrite,
                preserve_aspect_ratio=preserve_aspect_ratio,
                strict=args.strict,
            )

            if result is None:
                skipped_count += 1
            else:
                success_count += 1

        except Exception as exc:
            failed_count += 1
            print(
                f"[ERROR] Failed to process {metadata_path}:\n{exc}",
                file=sys.stderr,
            )

            if args.strict:
                return 1

    print(
        "\nFinished:"
        f"\n  generated: {success_count}"
        f"\n  skipped:   {skipped_count}"
        f"\n  failed:    {failed_count}"
    )

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())