"""
Main pipeline orchestrator — ties all 4 stages together.

Control flow:
  graph = plan(image)                          # Stage 1
  for el in sort_by_depth(graph.elements):     # Stage 2
      extract(el, graph)
  graph.background = extract_background(graph)  # Stage 3
  while global_attempts < BUDGET:              # Stage 4
      recon = reassemble(graph)
      verdict = global_verify(recon, original)
      if verdict.ok: break
      apply_route(verdict, graph)
  ship(graph)
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Optional

from PIL import Image

from scene_graph import (
    SceneGraph, Element, Background,
    ElementStatus, BackgroundStatus,
)
from config import (
    ELEMENT_RETRIES, BACKGROUND_RETRIES, GLOBAL_ATTEMPTS,
    MAX_ENUM_REOPENINGS, MAX_ELEMENTS, BBOX_NORM,
    JOYAI_BASE_SEED,
    VLM_MAX_TOKENS_PLANNER, VLM_MAX_TOKENS_CHECKER, VLM_MAX_TOKENS_PROMPT_WRITER,
)
from prompts import (
    PLANNER_PROMPT,
    OCCUPANCY_CHECKER_PROMPT,
    ISOLATION_PROMPT_WRITER_PROMPT,
    ELEMENT_VERIFIER_PROMPT, ELEMENT_VERIFIER_TEXT,
    BACKGROUND_PROMPT_WRITER_PROMPT,
    BACKGROUND_VERIFIER_PROMPT, BACKGROUND_VERIFIER_TEXT,
    GLOBAL_VERIFIER_PROMPT, GLOBAL_VERIFIER_TEXT,
)
from logger import RunLogger
import pipeline_tools
from pipeline_tools import (
    vlm, joyai, crop, matte_to_alpha, resize, composite,
    vlm_json, parse_json_relaxed,
    resize_layer_to_bbox, denorm_bbox_pixels,
    _denorm_bbox,
)


# ---------------------------------------------------------------------------
# Stage 1 — Planning
# ---------------------------------------------------------------------------
def plan(image_path: str | Path, logger: RunLogger) -> SceneGraph:
    """Stage 1: Call VLM Planner → populate SceneGraph with element list."""
    print("\n" + "=" * 60)
    print("STAGE 1 — PLANNING")
    print("=" * 60)

    img = Image.open(image_path)
    w, h = img.size
    graph = SceneGraph(image_path=str(image_path), image_size=(w, h))

    user_text = "Analyze this illustration and list all separable objects."

    if pipeline_tools.FAKE_MODE:
        response_text = pipeline_tools._fake_vlm(PLANNER_PROMPT, img, user_text)
        result = parse_json_relaxed(response_text)
    else:
        result = vlm_json(
            system_prompt=PLANNER_PROMPT,
            image_input=img,
            user_text=user_text,
            max_new_tokens=VLM_MAX_TOKENS_PLANNER,
            role="planner",
            logger=logger,
        )

    raw_elements = result.get("elements", [])
    print(f"[Stage 1] VLM identified {len(raw_elements)} elements")

    for el_data in raw_elements[:MAX_ELEMENTS]:
        name = el_data.get("name", "unknown")
        # Create stable id
        el_id = f"{name}_01"
        element = Element(
            id=el_id,
            name=name,
            bbox=el_data.get("bbox", [0, 0, 100, 100]),
            depth_rank=el_data.get("depth_rank", len(graph.elements) + 1),
            overlaps=el_data.get("overlaps", []),
        )
        graph.elements.append(element)
        print(f"  Element: {element.name} id={element.id} "
              f"depth={element.depth_rank} bbox={element.bbox}")

    # Deduplicate ids: if two elements share the same id, append suffixes
    seen_ids = set()
    for el in graph.elements:
        if el.id in seen_ids:
            el.id = f"{el.name}_{graph.elements.index(el):02d}"
        seen_ids.add(el.id)

    logger.save_scene_graph(graph, "stage1_plan")
    return graph


# ---------------------------------------------------------------------------
# Stage 2 — Element extraction
# ---------------------------------------------------------------------------
def _format_neighbor_names(graph: SceneGraph, current_el: Element) -> str:
    """Format the list of other element names for the occupancy checker."""
    other_names = [e.name for e in graph.elements if e.name != current_el.name]
    if not other_names:
        return "none"
    return ", ".join(f'"{n}"' for n in other_names)


def _check_occupancy(graph: SceneGraph, element: Element, crop_img: Image.Image,
                     logger: RunLogger) -> dict:
    """Stage 2, step 1: Occupancy check."""
    neighbor_names = _format_neighbor_names(graph, element)
    system_prompt = OCCUPANCY_CHECKER_PROMPT.replace("{name}", element.name)
    system_prompt = system_prompt.replace("{neighbor_names}", neighbor_names)

    user_text = f"Check if other objects intrude into this crop of '{element.name}'."

    if pipeline_tools.FAKE_MODE:
        response_text = pipeline_tools._fake_vlm(system_prompt, crop_img, user_text)
        return parse_json_relaxed(response_text) or {"target_present": True, "contaminants": []}

    return vlm_json(
        system_prompt=system_prompt,
        image_input=crop_img,
        user_text=user_text,
        max_new_tokens=VLM_MAX_TOKENS_CHECKER,
        role=f"occupancy_{element.name}",
        logger=logger,
    )


def _write_isolation_prompt(element: Element, contaminants: list[str],
                            defects: list[str], logger: RunLogger) -> str:
    """Stage 2, step 2: Write isolation prompt."""
    system_prompt = ISOLATION_PROMPT_WRITER_PROMPT
    system_prompt = system_prompt.replace("{name}", element.name)
    system_prompt = system_prompt.replace(
        "{contaminants}", json.dumps(contaminants) if contaminants else "[]")
    system_prompt = system_prompt.replace(
        "{overlaps}", json.dumps(element.overlaps) if element.overlaps else "[]")
    system_prompt = system_prompt.replace(
        "{defects}", json.dumps(defects) if defects else "[]")

    user_text = f"Write an isolation instruction for '{element.name}'."

    if pipeline_tools.FAKE_MODE:
        response_text = pipeline_tools._fake_vlm(system_prompt, None, user_text)
        result = parse_json_relaxed(response_text) or {}
        prompt_text = result.get("prompt", f"isolate {element.name} on plain background")
        logger.log_text(prompt_text, label=f"prompt_{element.name}")
        return prompt_text

    result = vlm_json(
        system_prompt=system_prompt,
        image_input=None,
        user_text=user_text,
        max_new_tokens=VLM_MAX_TOKENS_PROMPT_WRITER,
        role=f"prompt_writer_{element.name}",
        logger=logger,
    )
    prompt_text = result.get("prompt", f"isolate the {element.name} on plain background")
    # Log the generated prompt as standalone text
    logger.log_text(prompt_text, label=f"prompt_{element.name}")
    return prompt_text


def _verify_element(element: Element, original_crop: Image.Image,
                    result_cutout: Image.Image, logger: RunLogger) -> dict:
    """Stage 2, step 5: Verify element cutout."""
    system_prompt = ELEMENT_VERIFIER_PROMPT.replace("{name}", element.name)
    user_text = ELEMENT_VERIFIER_TEXT

    if pipeline_tools.FAKE_MODE:
        response_text = pipeline_tools._fake_vlm(system_prompt,
                                  [original_crop, result_cutout],
                                  user_text)
        return parse_json_relaxed(response_text) or {"ok": True, "defects": [], "notes": ""}

    return vlm_json(
        system_prompt=system_prompt,
        image_input=[original_crop, result_cutout],
        user_text=user_text,
        max_new_tokens=VLM_MAX_TOKENS_CHECKER,
        role=f"verifier_{element.name}",
        logger=logger,
    )


def extract_element(element: Element, graph: SceneGraph, logger: RunLogger, skip_verify: bool) -> None:
    """Stage 2: Full element extraction pipeline with retry loop."""
    print(f"\n  --- Extracting: {element.name} (id={element.id}) ---")
    element.status = ElementStatus.EXTRACTING

    original_img = Image.open(graph.image_path)
    w, h = graph.image_size

    # Step 1: Crop the bbox with padding
    crop_img = crop(original_img, element.bbox, pad=0.1)
    crop_path = logger.save_image(crop_img, f"crop_{element.name}")
    print(f"  Crop saved: {crop_path}")

    # Step 2: Occupancy check
    occupancy = _check_occupancy(graph, element, crop_img, logger)
    target_present = occupancy.get("target_present", True)
    contaminants = occupancy.get("contaminants", [])
    print(f"  Occupancy: present={target_present}, contaminants={contaminants}")

    if not target_present:
        print(f"  WARNING: target '{element.name}' not found in crop! Continuing anyway.")

    # Retry loop
    best_result = None
    best_defects = None
    best_attempt = 0

    for attempt in range(1, ELEMENT_RETRIES + 1):
        print(f"  Attempt {attempt}/{ELEMENT_RETRIES}")

        # Step 3: Write isolation prompt
        defects_for_prompt = []
        if best_defects:
            defects_for_prompt = best_defects
        isolation_prompt = _write_isolation_prompt(
            element, contaminants, defects_for_prompt, logger)
        element.isolation_prompt = isolation_prompt
        print(f"  Isolation prompt: {isolation_prompt[:100]}...")

        # Step 4: Generate with JoyAI
        seed = JOYAI_BASE_SEED + attempt
        gen_out_path = logger.run_dir / f"joyai_{element.name}_attempt{attempt}.png"
        if pipeline_tools.FAKE_MODE:
            generated = pipeline_tools._fake_joyai(crop_img, isolation_prompt, gen_out_path, seed)
        else:
            generated = joyai(crop_img, isolation_prompt, gen_out_path, seed)
            if generated:
                logger.save_image(generated, f"gen_{element.name}_att{attempt}")

        if generated is None:
            print(f"  JoyAI generation failed for {element.name}")
            continue

        # Step 5: Matte to alpha
        rgba_layer = matte_to_alpha(generated)
        matte_path = logger.save_image(rgba_layer, f"matte_{element.name}_att{attempt}")

        # Resize to bbox dimensions
        rgba_resized = resize_layer_to_bbox(rgba_layer, element.bbox, w, h)

        # Step 6: Verify
        if skip_verify:
            print("Skipping verifcation...")
            verification = {"ok": True, "defects":None, "notes":None}
        else:
            verification = _verify_element(element, crop_img, rgba_resized, logger)
        ok = verification.get("ok", False)
        defects = verification.get("defects", [])
        notes = verification.get("notes", "")

        print(f"  Verification: ok={ok}, defects={defects}, notes={notes}")

        # Track best
        if best_result is None or ok:
            best_result = rgba_resized
            best_defects = defects
            best_attempt = attempt
            element.layer_path = str(matte_path)
            element.defects = defects

        if ok:
            break

    # After retry loop
    element.attempts = best_attempt
    if best_result is not None:
        # Use best_defects to determine if the final result is clean
        has_remaining_defects = bool(best_defects)
        if has_remaining_defects:
            element.status = ElementStatus.FAILED
            element.defects = best_defects
            print(f"  >>> Element {element.name} FAILED after {best_attempt} attempts (defects: {best_defects})")
        else:
            element.status = ElementStatus.DONE
            element.defects = []
            print(f"  >>> Element {element.name} DONE")
        # Save final layer
        final_path = logger.run_dir / f"layer_{element.name}.png"
        best_result.save(final_path)
        element.layer_path = str(final_path)
    else:
        element.status = ElementStatus.FAILED
        element.defects = ["generation_failed"]
        print(f"  >>> Element {element.name} FAILED — all JoyAI attempts exhausted")

    logger.save_scene_graph(graph, f"stage2_after_{element.name}")


def run_stage2(graph: SceneGraph, logger: RunLogger, skip_verify: bool) -> None:
    """Stage 2: Process all elements front-to-back by depth_rank."""
    print("\n" + "=" * 60)
    print("STAGE 2 — ELEMENT EXTRACTION")
    print("=" * 60)

    # Process in depth order (frontmost=1 first, then deeper)
    sorted_els = graph.sorted_elements()
    for i, element in enumerate(sorted_els):
        print(f"\n[{i + 1}/{len(sorted_els)}] Element: {element.name} (depth={element.depth_rank})")
        extract_element(element, graph, logger, skip_verify)


# ---------------------------------------------------------------------------
# Stage 3 — Background extraction
# ---------------------------------------------------------------------------
def _write_background_prompt(graph: SceneGraph, defects: list[str],
                             logger: RunLogger) -> str:
    """Stage 3: Write background removal prompt."""
    element_names = [e.name for e in graph.elements]
    system_prompt = BACKGROUND_PROMPT_WRITER_PROMPT
    system_prompt = system_prompt.replace(
        "{element_names}", json.dumps(element_names))
    system_prompt = system_prompt.replace(
        "{defects}", json.dumps(defects) if defects else "[]")

    user_text = "Write a background extraction instruction."

    if pipeline_tools.FAKE_MODE:
        response_text = pipeline_tools._fake_vlm(system_prompt, None, user_text)
        result = parse_json_relaxed(response_text) or {}
        prompt_text = result.get("prompt", "remove all foreground objects")
        logger.log_text(prompt_text, label="prompt_background")
        return prompt_text

    result = vlm_json(
        system_prompt=system_prompt,
        image_input=None,
        user_text=user_text,
        max_new_tokens=VLM_MAX_TOKENS_PROMPT_WRITER,
        role="bg_prompt_writer",
        logger=logger,
    )
    prompt_text = result.get("prompt", "remove all foreground objects, fill the background")
    # Log the generated prompt as standalone text
    logger.log_text(prompt_text, label="prompt_background")
    return prompt_text


def _verify_background(graph: SceneGraph, original_img: Image.Image,
                       bg_result: Image.Image, logger: RunLogger) -> dict:
    """Stage 3: Verify background."""
    element_names = [e.name for e in graph.elements]
    system_prompt = BACKGROUND_VERIFIER_PROMPT.replace(
        "{element_names}", json.dumps(element_names))
    user_text = BACKGROUND_VERIFIER_TEXT

    if pipeline_tools.FAKE_MODE:
        response_text = pipeline_tools._fake_vlm(system_prompt,
                                  [original_img, bg_result],
                                  user_text)
        return parse_json_relaxed(response_text) or {"ok": True, "defects": [], "notes": ""}

    return vlm_json(
        system_prompt=system_prompt,
        image_input=[original_img, bg_result],
        user_text=user_text,
        max_new_tokens=VLM_MAX_TOKENS_CHECKER,
        role="bg_verifier",
        logger=logger,
    )


def extract_background(graph: SceneGraph, logger: RunLogger) -> None:
    """Stage 3: Extract background with retry loop."""
    print("\n" + "=" * 60)
    print("STAGE 3 — BACKGROUND EXTRACTION")
    print("=" * 60)

    graph.background.status = BackgroundStatus.GENERATING
    original_img = Image.open(graph.image_path)

    best_bg = None
    best_defects = None

    for attempt in range(1, BACKGROUND_RETRIES + 1):
        print(f"\n  Background attempt {attempt}/{BACKGROUND_RETRIES}")

        # Write prompt
        defects_for_prompt = best_defects if best_defects else []
        bg_prompt = _write_background_prompt(graph, defects_for_prompt, logger)
        graph.background.prompt = bg_prompt
        print(f"  Background prompt: {bg_prompt[:120]}...")

        # Generate
        seed = JOYAI_BASE_SEED + attempt * 100
        bg_out_path = logger.run_dir / f"background_attempt{attempt}.png"
        if pipeline_tools.FAKE_MODE:
            generated = pipeline_tools._fake_joyai(original_img, bg_prompt, bg_out_path, seed)
        else:
            generated = joyai(original_img, bg_prompt, bg_out_path, seed)
            if generated:
                logger.save_image(generated, f"bg_gen_att{attempt}")

        if generated is None:
            print(f"  Background generation failed on attempt {attempt}")
            continue

        # Verify
        verification = _verify_background(graph, original_img, generated, logger)
        ok = verification.get("ok", False)
        defects = verification.get("defects", [])
        notes = verification.get("notes", "")
        print(f"  Background verification: ok={ok}, defects={defects}, notes={notes}")

        best_bg = generated
        best_defects = defects

        if ok:
            break

    graph.background.attempts = attempt

    if best_bg is not None:
        bg_path = logger.run_dir / "background_final.png"
        best_bg.save(bg_path)
        graph.background.image_path = str(bg_path)
        graph.background.defects = best_defects or []
        if best_defects:
            graph.background.status = BackgroundStatus.FAILED
            print("  >>> Background FAILED after retries")
        else:
            graph.background.status = BackgroundStatus.DONE
            print("  >>> Background DONE")
    else:
        graph.background.status = BackgroundStatus.FAILED
        graph.background.defects = ["generation_failed"]
        # Fallback: use a flat color
        fallback = Image.new('RGB', graph.image_size, (128, 128, 128))
        bg_path = logger.run_dir / "background_fallback.png"
        fallback.save(bg_path)
        graph.background.image_path = str(bg_path)
        print("  >>> Background FAILED — using fallback gray")

    logger.save_scene_graph(graph, "stage3_background")


# ---------------------------------------------------------------------------
# Stage 4 — Reassembly + Global Verification
# ---------------------------------------------------------------------------
def reassemble(graph: SceneGraph, logger: RunLogger) -> Image.Image:
    """Stage 4, step 1: Composite all layers over background."""
    print("\n" + "=" * 60)
    print("STAGE 4 — REASSEMBLY")
    print("=" * 60)

    bg_path = graph.background.image_path
    if not bg_path or not Path(bg_path).exists():
        print("  WARNING: No background image, using black fallback")
        bg_img = Image.new('RGB', graph.image_size, (0, 0, 0))
        bg_path = logger.run_dir / "fallback_bg.png"
        bg_img.save(bg_path)
        bg_path = str(bg_path)

    # Build layer stack: back-to-front (higher depth_rank = further back, drawn first)
    sorted_els = sorted(graph.elements, key=lambda e: e.depth_rank, reverse=True)
    layers = []
    for el in sorted_els:
        if el.layer_path and Path(el.layer_path).exists() and el.status != ElementStatus.FAILED:
            layer_img = Image.open(el.layer_path)
            layers.append((layer_img, el.bbox))
            print(f"  Layer: {el.name} (depth={el.depth_rank})")
        elif el.status == ElementStatus.FAILED:
            layer_img = Image.open(el.layer_path)
            layers.append((layer_img, el.bbox))
            print(f"  Notice: Using failed element: Layer: {el.name} (depth={el.depth_rank})")

    reconstruction = composite(bg_path, layers)
    recon_path = logger.save_image(reconstruction, "reconstruction")
    print(f"  Reconstruction saved: {recon_path}")
    logger.save_scene_graph(graph, "stage4_reassembly")
    return reconstruction


def global_verify(graph: SceneGraph, reconstruction: Image.Image,
                  logger: RunLogger, skip_global: bool) -> dict:
    """Stage 4, step 2: Global verification."""

    if skip_global:
        print("Skipping global check...")
        skipped_dict = {
                        "ok": True,
                        "missing":None,
                        "bad_layers":None,
                        "reorder":None,
                        "notes":None
                        }
        return skipped_dict

    element_summaries = []
    for el in graph.elements:
        element_summaries.append({
            "name": el.name,
            "bbox": el.bbox,
            "depth_rank": el.depth_rank,
        })
    system_prompt = GLOBAL_VERIFIER_PROMPT.replace(
        "{element_summaries}", json.dumps(element_summaries))
    user_text = GLOBAL_VERIFIER_TEXT

    original_img = Image.open(graph.image_path)

    if pipeline_tools.FAKE_MODE:
        response_text = pipeline_tools._fake_vlm(system_prompt,
                                  [original_img, reconstruction],
                                  user_text)
        return parse_json_relaxed(response_text) or {"ok": True, "missing": [], "bad_layers": [], "reorder": [], "notes": ""}

    return vlm_json(
        system_prompt=system_prompt,
        image_input=[original_img, reconstruction],
        user_text=user_text,
        max_new_tokens=VLM_MAX_TOKENS_CHECKER,
        role="global_verifier",
        logger=logger,
    )


def apply_routing(verdict: dict, graph: SceneGraph, logger: RunLogger) -> bool:
    """Stage 4, step 3: Apply the verifier's routing decisions.

    Returns True if any action was taken (pipeline should loop), False otherwise.
    """
    action_taken = False

    # 1. Handle missing elements — reopen Stage 1
    missing = verdict.get("missing", [])
    if missing and graph.enum_reopenings < MAX_ENUM_REOPENINGS:
        print(f"  Global verifier found {len(missing)} MISSING elements: {missing}")
        for m in missing:
            name = m.get("name", "unknown")
            bbox = m.get("bbox", [0, 0, 100, 100])
            new_el = Element(
                id=graph.next_available_id(),
                name=name,
                bbox=bbox,
                depth_rank=max([e.depth_rank for e in graph.elements] or [1]) + 1,
                overlaps=[],
            )
            graph.elements.append(new_el)
            graph.enum_reopenings += 1
            print(f"  Added missing element: {new_el.name} id={new_el.id}")
            # Extract it immediately
            extract_element(new_el, graph, logger)
            action_taken = True

    # 2. Handle bad layers — re-run that element's Stage 2
    bad_layers = verdict.get("bad_layers", [])
    for bad in bad_layers:
        name = bad.get("name", "")
        defects = bad.get("defects", [])
        el = graph.get_element_by_name(name)
        if el and el.attempts < ELEMENT_RETRIES:
            print(f"  Re-running bad layer: {name} defects={defects}")
            el.defects = defects
            el.status = ElementStatus.EXTRACTING
            extract_element(el, graph, logger)
            action_taken = True

    # 3. Handle z-order
    reorder = verdict.get("reorder", [])
    for r in reorder:
        front_name = r.get("front", "")
        behind_name = r.get("behind", "")
        front_el = graph.get_element_by_name(front_name)
        behind_el = graph.get_element_by_name(behind_name)
        if front_el and behind_el:
            # Ensure front_el has lower depth_rank (since 1 = frontmost)
            if front_el.depth_rank > behind_el.depth_rank:
                # Swap depth ranks
                print(f"  Fixing z-order: {front_name} should be in front of {behind_name}")
                front_el.depth_rank, behind_el.depth_rank = behind_el.depth_rank, front_el.depth_rank
                action_taken = True

    return action_taken


def run_stage4(graph: SceneGraph, logger: RunLogger, skip_global: bool) -> bool:
    """Stage 4: Reassembly + global verification master loop.

    Returns True if final result is acceptable, False if budget exhausted.
    """
    while graph.global_attempts < GLOBAL_ATTEMPTS:
        graph.global_attempts += 1
        print(f"\n  --- Global attempt {graph.global_attempts}/{GLOBAL_ATTEMPTS} ---")

        recon = reassemble(graph, logger)
        verdict = global_verify(graph, recon, logger, skip_global)

        ok = verdict.get("ok", False)
        notes = verdict.get("notes", "")
        print(f"  Global verification: ok={ok}, notes={notes}")
        print(f"  Missing: {verdict.get('missing', [])}")
        print(f"  Bad layers: {verdict.get('bad_layers', [])}")
        print(f"  Reorder: {verdict.get('reorder', [])}")

        if ok:
            print("\n  >>> RECONSTRUCTION ACCEPTED")
            logger.save_scene_graph(graph, "final_accepted")
            return True

        # Apply routing
        action_taken = apply_routing(verdict, graph, logger)
        if not action_taken:
            print("\n  >>> NO ROUTING ACTIONS — terminating loop")
            break

    # Budget exhausted
    print(f"\n  >>> BUDGET EXHAUSTED ({graph.global_attempts}/{GLOBAL_ATTEMPTS}) — shipping best partial")
    graph.save(logger.run_dir / "final_best_partial.json")
    return False


# ---------------------------------------------------------------------------
# Ship
# ---------------------------------------------------------------------------
def ship(graph: SceneGraph, logger: RunLogger) -> dict:
    """Package final outputs."""
    print("\n" + "=" * 60)
    print("SHIPPING")
    print("=" * 60)

    output = {
        "image_path": graph.image_path,
        "image_size": list(graph.image_size),
        "background": graph.background.to_dict(),
        "elements": [e.to_dict() for e in graph.elements],
        "reconstruction": str(logger.run_dir / "reconstruction.png"),
        "run_dir": str(logger.run_dir),
    }

    # Save final scene graph
    logger.save_scene_graph(graph, "final_shipped")
    print(f"  Output directory: {logger.run_dir}")
    print(f"  Elements: {len(graph.elements)}")
    done = sum(1 for e in graph.elements if e.status == ElementStatus.DONE)
    failed = sum(1 for e in graph.elements if e.status == ElementStatus.FAILED)
    print(f"  Done: {done}, Failed: {failed}")
    return output


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_pipeline(image_path: str | Path, output_dir: str = "runs",
                 use_fake: bool = False, skip_verify: bool = False, skip_global: bool = False) -> dict:
    """Run the full layer decomposition pipeline on one image.

    Args:
        image_path: path to the illustration image.
        output_dir: where to write the run folder.
        use_fake: if True, use stubs for VLM and JoyAI (testing only).

    Returns:
        dict with paths to all outputs.
    """
    pipeline_tools.set_fake_mode(use_fake)
    if use_fake:
        print("[PIPELINE] RUNNING IN FAKE/STUB MODE — no real model calls")

    # Auto-detect GPUs before any model loading (only in non-fake mode)
    if not use_fake:
        device1, device2 = pipeline_tools.auto_detect_gpu_devices()

    if skip_verify:
        print("Using skipping verification after cleaned cropped image.")

    if skip_global:
        print("Using skipping global recheck.")

    # Derive image prefix from basename (e.g. "009" from "009.png")
    image_stem = Path(image_path).stem
    logger = RunLogger(output_dir, image_prefix=image_stem)
    print(f"[PIPELINE] Run dir: {logger.run_dir}")

    try:
        # Stage 1
        graph = plan(image_path, logger)

        if not graph.elements:
            print("[PIPELINE] No elements found in plan. Skipping extraction.")
            graph.background.status = BackgroundStatus.DONE
            graph.background.image_path = image_path  # use original as bg fallback
        else:
            # Stage 2
            run_stage2(graph, logger, skip_verify)

            # Stage 3
            extract_background(graph, logger)

        # Stage 4
        run_stage4(graph, logger, skip_global)

        # Ship
        result = ship(graph, logger)
        print("\n[PIPELINE] COMPLETE")
        return result

    except Exception as e:
        print(f"\n[PIPELINE] ERROR: {e}")
        traceback.print_exc()
        logger.log_text(traceback.format_exc(), "pipeline_error")
        raise