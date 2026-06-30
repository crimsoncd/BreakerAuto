#!/usr/bin/env python3
"""
Layer Decomposition Pipeline — CLI entry point.

Usage:
    # Run on a single image with real models (requires 4×A100 GPUs)
    python main.py --image images/009.png

    # Run in fake/stub mode for testing (no GPUs needed)
    python main.py --image images/009.png --fake

    # Batch process all images in a directory
    python main.py --dir images/ --output runs

    # Batch with fake mode
    python main.py --dir images/ --fake
"""

import argparse
import sys
import time
from pathlib import Path

from pipeline import run_pipeline



def process_single(image_path: Path, output_dir: str, use_fake: bool,
                   skip_verify: bool, skip_global: bool) -> dict:
    """Process one image through the decomposition pipeline."""
    print("\n" + "#" * 70)
    print(f"# IMAGE: {image_path.name}")
    print("#" * 70)

    result = run_pipeline(
        image_path=str(image_path),
        output_dir=output_dir,
        use_fake=use_fake,
        skip_verify=skip_verify,
        skip_global=skip_global
    )

    print(f"\n  Run dir:        {result['run_dir']}")
    print(f"  Reconstruction: {result['reconstruction']}")
    print(f"  Elements:       {len(result['elements'])}")
    done = sum(1 for e in result["elements"] if e["status"] == "done")
    failed = sum(1 for e in result["elements"] if e["status"] == "failed")
    print(f"  Done: {done}, Failed: {failed}")
    return result


def process_batch(image_dir: Path, output_dir: str, use_fake: bool,
                  skip_verify: bool, skip_global: bool) -> dict:
    """Process all images in a directory."""
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])

    if not image_files:
        print(f"No image files found in {image_dir}")
        return {"processed": 0, "results": []}

    print(f"\n{'=' * 70}")
    print(f"BATCH MODE: {len(image_files)} images in {image_dir}")
    print(f"{'=' * 70}")

    results = []
    start_time = time.time()
    for i, img_path in enumerate(image_files):
        print(f"\n[{i + 1}/{len(image_files)}]")
        try:
            result = process_single(img_path, output_dir, use_fake, skip_verify, skip_global)
            results.append({"image": img_path.name, "status": "ok", "result": result})
        except Exception as e:
            print(f"  ERROR processing {img_path.name}: {e}")
            results.append({"image": img_path.name, "status": "error", "error": str(e)})

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"BATCH COMPLETE: {len(results)} images in {elapsed:.0f}s")
    print(f"{'=' * 70}")
    for r in results:
        status_icon = "✓" if r["status"] == "ok" else "✗"
        print(f"  {status_icon} {r['image']}: {r['status']}")

    return {"processed": len(results), "results": results}


def main():
    parser = argparse.ArgumentParser(
        description="Decompose an illustration into background + element layers"
    )
    # Single image mode
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a single illustration image"
    )
    # Batch mode
    parser.add_argument(
        "--dir", type=str, default=None,
        help="Directory containing multiple images to process"
    )
    parser.add_argument(
        "--output", type=str, default="runs",
        help="Output directory for run folders (default: runs)"
    )
    parser.add_argument(
        "--fake", action="store_true",
        help="Run in fake/stub mode without real model calls"
    )
    parser.add_argument(
        "--skip_verify", action="store_true",
        help="Skip middle verify of cropped and cleaned elements."
    )
    parser.add_argument(
        "--skip_global", action="store_true",
        help="Skip middle verify of cropped and cleaned elements."
    )
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error("Either --image or --dir must be specified")

    # Single image mode
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"ERROR: Image not found: {image_path}")
            sys.exit(1)
        process_single(image_path, args.output, use_fake=args.fake,
                       skip_verify=args.skip_verify, skip_global=args.skip_global)
        print("\nDone.")
        return

    # Batch mode
    if args.dir:
        image_dir = Path(args.dir)
        if not image_dir.is_dir():
            print(f"ERROR: Directory not found: {image_dir}")
            sys.exit(1)
        process_batch(image_dir, args.output, use_fake=args.fake,
                       skip_verify=args.skip_verify, skip_global=args.skip_global)
        print("\nDone.")
        return


if __name__ == "__main__":
    main()