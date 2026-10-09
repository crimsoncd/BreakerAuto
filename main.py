#!/usr/bin/env python3
"""
Layer Decomposition Pipeline — CLI entry point.

Usage:
    # Run on a single image with real models (requires a multi-GPU box)
    python main.py --image images/009.png

    # Run in fake/stub mode for testing (no GPUs needed)
    python main.py --image images/009.png --fake

    # Enable the per-element VLM verification loop
    python main.py --image images/009.png --use_verify

    # Batch process all images in a directory
    python main.py --dir images/ --output runs

    # Batch with fake mode
    python main.py --dir images/ --fake
"""

import argparse
import sys
import time
from pathlib import Path

from config import DEFAULT_OUTPUT_DIR, BACKGROUND_METHOD
from pipeline import run_pipeline


def process_single(image_path: Path, output_dir: str, use_fake: bool,
                   use_verify: bool, use_global: bool,
                   bg_method: str = None) -> dict:
    """Process one image through the decomposition pipeline."""
    print("\n" + "#" * 70)
    print(f"# IMAGE: {image_path.name}")
    print("#" * 70)

    t0 = time.time()
    result = run_pipeline(
        image_path=str(image_path),
        output_dir=output_dir,
        use_fake=use_fake,
        use_verify=use_verify,
        use_global=use_global,
        bg_method=bg_method,
    )
    elapsed = time.time() - t0

    print(f"\n  Run dir:        {result['run_dir']}")
    if result.get("package_dir"):
        print(f"  Package:        {result['package_dir']}")
    print(f"  Reconstruction: {result['reconstruction']}")
    print(f"  Elements:       {len(result['elements'])}")
    done = sum(1 for e in result["elements"] if e["status"] == "done")
    failed = sum(1 for e in result["elements"] if e["status"] == "failed")
    print(f"  Done: {done}, Failed: {failed}")
    print(f"  Total time:     {elapsed:.1f}s")
    return result


def process_batch(image_dir: Path, output_dir: str, use_fake: bool,
                  use_verify: bool, use_global: bool,
                  bg_method: str = None) -> dict:
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
            result = process_single(img_path, output_dir, use_fake, use_verify, use_global, bg_method)
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
        "--output", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for run folders (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--fake", action="store_true",
        help="Run in fake/stub mode without real model calls"
    )
    parser.add_argument(
        "--use_verify", action="store_true",
        help="Enable VLM verification of each extracted element (default: skipped for speed)."
    )
    parser.add_argument(
        "--use_global", action="store_true",
        help="Enable the final global reconstruction verification loop (default: skipped for speed)."
    )
    parser.add_argument(
        "--bg_method", type=str, default=BACKGROUND_METHOD,
        choices=["classic", "model"],
        help=("Stage-3 background extraction method: 'classic' (default) = "
              "classical color fill when the background is verifiably flat, "
              "with model fallback; 'model' = VLM-written color-anchored "
              "prompt + Qwen edit generation.")
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
                       use_verify=args.use_verify, use_global=args.use_global,
                       bg_method=args.bg_method)
        print("\nDone.")
        return

    # Batch mode
    if args.dir:
        image_dir = Path(args.dir)
        if not image_dir.is_dir():
            print(f"ERROR: Directory not found: {image_dir}")
            sys.exit(1)
        process_batch(image_dir, args.output, use_fake=args.fake,
                      use_verify=args.use_verify, use_global=args.use_global,
                      bg_method=args.bg_method)
        print("\nDone.")
        return


if __name__ == "__main__":
    main()
