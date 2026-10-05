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
import shutil
import time
import traceback
from pathlib import Path

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
    VLM_MAX_TOKENS_DESCRIBER,
    DEFAULT_OUTPUT_DIR,
)
from prompts import (
    PLANNER_PROMPT,
    OCCUPANCY_CHECKER_PROMPT,
    ISOLATION_PROMPT_WRITER_PROMPT,
    ELEMENT_VERIFIER_PROMPT, ELEMENT_VERIFIER_TEXT,
    BACKGROUND_PROMPT_WRITER_PROMPT,
    BACKGROUND_VERIFIER_PROMPT, BACKGROUND_VERIFIER_TEXT,
    GLOBAL_VERIFIER_PROMPT, GLOBAL_VERIFIER_TEXT,
    DESCRIBER_PROMPT,
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
# Progress / timing helpers
# ---------------------------------------------------------------------------
def _fmt_duration(seconds: float) -> str:
    """Format seconds as '65.3s' below 2 minutes, else '12m 05s'."""
    if seconds < 120:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s"


# ---------------------------------------------------------------------------
# Stage 1 — Planning
# ---------------------------------------------------------------------------
def _bboxes_overlap(a: list[int], b: list[int]) -> bool:
    """True if two 0-1000 normalized bboxes intersect."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _compute_overlaps(elements: list[Element]) -> None:
    """Fill each element's `overlaps` from bbox geometry.

    The planner no longer reports overlaps; derive them: an element is listed
    in another's `overlaps` when their bboxes intersect and it is drawn LATER
    (higher order), i.e. it may occlude the other.
    """
    for el in elements:
        el.overlaps = [
            other.name for other in elements
            if other is not el
            and other.order > el.order
            and _bboxes_overlap(el.bbox, other.bbox)
        ]


def plan(image_path: str | Path, logger: RunLogger) -> SceneGraph:
    """Stage 1: Call VLM Planner → populate SceneGraph with element list."""
    print("\n" + "=" * 60)
    print("STAGE 1 — PLANNING")
    print("=" * 60)

    img = Image.open(image_path)
    w, h = img.size
    graph = SceneGraph(image_path=str(image_path), image_size=(w, h))

    user_text = "Analyze this illustration and return the structured layout JSON."

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

    raw_layout = result.get("layout", [])
    print(f"[Stage 1] VLM identified {len(raw_layout)} layout items")

    # Split background (order 0) from objects; the planner may reuse an order
    # value for several items, so re-assign a compact 1..N sequence (stable
    # sort keeps the planner's z-order) and derive the one-and-only id from it.
    bg_item = None
    objects = []
    for item in raw_layout:
        if item.get("order", 1) == 0 and bg_item is None:
            bg_item = item
        else:
            objects.append(item)
    objects.sort(key=lambda it: it.get("order", 1))

    if bg_item is not None:
        graph.background.name = bg_item.get("name", "background")
        graph.background.description = bg_item.get("description")
        print(f"  Background: {graph.background.name} (order 0) bbox={bg_item.get('bbox')}")

    for item in objects[:MAX_ELEMENTS]:
        name = item.get("name", "unknown")
        order = len(graph.elements) + 1
        element = Element(
            id=f"element_{order:02d}",
            name=name,
            bbox=item.get("bbox", [0, 0, 1000, 1000]),
            order=order,
            description=item.get("description"),
        )
        graph.elements.append(element)
        print(f"  Element: {element.name} id={element.id} "
              f"order={element.order} (planned={item.get('order')}) bbox={element.bbox}")

    if len(objects) > MAX_ELEMENTS:
        print(f"  WARNING: MAX_ELEMENTS ({MAX_ELEMENTS}) reached — {len(objects) - MAX_ELEMENTS} items dropped")

    # Overlaps are no longer planned — derive them from bbox geometry.
    _compute_overlaps(graph.elements)

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
        role=f"occupancy_{element.id}",
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
        logger.log_text(prompt_text, label=f"prompt_{element.id}")
        return prompt_text

    result = vlm_json(
        system_prompt=system_prompt,
        image_input=None,
        user_text=user_text,
        max_new_tokens=VLM_MAX_TOKENS_PROMPT_WRITER,
        role=f"prompt_writer_{element.id}",
        logger=logger,
    )
    prompt_text = result.get("prompt", f"isolate the {element.name} on plain background")
    # Log the generated prompt as standalone text
    logger.log_text(prompt_text, label=f"prompt_{element.id}")
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
        role=f"verifier_{element.id}",
        logger=logger,
    )


def extract_element(element: Element, graph: SceneGraph, logger: RunLogger, use_verify: bool) -> None:
    """Stage 2: Full element extraction pipeline with retry loop."""
    print(f"\n  --- Extracting: {element.name} (id={element.id}) ---")
    print(f"  [el {element.id}] extraction started")
    t_el = time.time()
    element.status = ElementStatus.EXTRACTING

    original_img = Image.open(graph.image_path)
    w, h = graph.image_size

    # Step 1: Crop the bbox with padding
    crop_img = crop(original_img, element.bbox, pad=0.1)
    crop_path = logger.save_image(crop_img, f"crop_{element.id}")
    print(f"  [el {element.id}] Step 1/6 crop saved: {crop_path}")

    # Step 2: Occupancy check
    print(f"  [el {element.id}] Step 2/6 occupancy check...")
    occupancy = _check_occupancy(graph, element, crop_img, logger)
    target_present = occupancy.get("target_present", True)
    contaminants = occupancy.get("contaminants", [])
    print(f"  [el {element.id}] Step 2/6 occupancy done: present={target_present}, contaminants={contaminants}")

    if not target_present:
        print(f"  [el {element.id}] WARNING: target '{element.name}' not found in crop! Continuing anyway.")

    # Retry loop
    best_result = None
    best_defects = None
    best_attempt = 0

    for attempt in range(1, ELEMENT_RETRIES + 1):
        print(f"\n  [el {element.id}] Attempt {attempt}/{ELEMENT_RETRIES} started")
        t_att = time.time()

        # Step 3: Write isolation prompt
        print(f"  [el {element.id}] Step 3/6 writing isolation prompt (attempt {attempt}/{ELEMENT_RETRIES})...")
        defects_for_prompt = []
        if best_defects:
            defects_for_prompt = best_defects
        isolation_prompt = _write_isolation_prompt(
            element, contaminants, defects_for_prompt, logger)
        element.isolation_prompt = isolation_prompt
        print(f"  [el {element.id}] Step 3/6 isolation prompt: {isolation_prompt[:80]}...")

        # Step 4: Generate with JoyAI
        seed = JOYAI_BASE_SEED + attempt
        gen_out_path = logger.run_dir / f"joyai_{element.id}_attempt{attempt}.png"
        print(f"  [el {element.id}] Step 4/6 JoyAI generation (attempt {attempt}/{ELEMENT_RETRIES}, seed={seed})...")
        if pipeline_tools.FAKE_MODE:
            generated = pipeline_tools._fake_joyai(crop_img, isolation_prompt, gen_out_path, seed)
        else:
            generated = joyai(crop_img, isolation_prompt, gen_out_path, seed)
            if generated:
                logger.save_image(generated, f"gen_{element.id}_att{attempt}")

        if generated is None:
            print(f"  [el {element.id}] attempt {attempt} aborted after {_fmt_duration(time.time() - t_att)} — JoyAI generation failed")
            continue

        # Step 5: Matte to alpha
        print(f"  [el {element.id}] Step 5/6 matting to alpha + resize to bbox...")
        rgba_layer = matte_to_alpha(generated)
        matte_path = logger.save_image(rgba_layer, f"matte_{element.id}_att{attempt}")

        # Resize to bbox dimensions
        rgba_resized = resize_layer_to_bbox(rgba_layer, element.bbox, w, h)

        # Step 6: Verify
        if not use_verify:
            print(f"  [el {element.id}] Step 6/6 verification skipped (--use_verify not set)")
            verification = {"ok": True, "defects": None, "notes": None}
        else:
            print(f"  [el {element.id}] Step 6/6 verifying cutout against original crop...")
            verification = _verify_element(element, crop_img, rgba_resized, logger)
        ok = verification.get("ok", False)
        defects = verification.get("defects", [])
        notes = verification.get("notes", "")

        print(f"  [el {element.id}] Step 6/6 verification result: ok={ok}, defects={defects}, notes={(notes or '')[:80]}")

        # Track best
        if best_result is None or ok:
            best_result = rgba_resized
            best_defects = defects
            best_attempt = attempt
            element.layer_path = str(matte_path)
            element.defects = defects

        print(f"  [el {element.id}] attempt {attempt} finished in {_fmt_duration(time.time() - t_att)}")
        if ok:
            break

    # After retry loop
    print(f"  [el {element.id}] retry loop finished — best attempt {best_attempt}, total {_fmt_duration(time.time() - t_el)}")
    element.attempts = best_attempt
    if best_result is not None:
        # Use best_defects to determine if the final result is clean
        has_remaining_defects = bool(best_defects)
        if has_remaining_defects:
            element.status = ElementStatus.FAILED
            element.defects = best_defects
            print(f"  >>> Element {element.id} FAILED after {best_attempt} attempts (defects: {best_defects})")
        else:
            element.status = ElementStatus.DONE
            element.defects = []
            print(f"  >>> Element {element.id} DONE")
        # Save final layer
        final_path = logger.run_dir / f"layer_{element.id}.png"
        best_result.save(final_path)
        element.layer_path = str(final_path)
    else:
        element.status = ElementStatus.FAILED
        element.defects = ["generation_failed"]
        print(f"  >>> Element {element.id} FAILED — all JoyAI attempts exhausted")

    logger.save_scene_graph(graph, f"stage2_after_{element.id}")


def run_stage2(graph: SceneGraph, logger: RunLogger, use_verify: bool) -> None:
    """Stage 2: Process all elements front-to-back by order (highest first)."""
    print("\n" + "=" * 60)
    print("STAGE 2 — ELEMENT EXTRACTION")
    print("=" * 60)

    # Process in drawing order reversed (frontmost = highest order first, then deeper)
    sorted_els = graph.sorted_elements()
    for i, element in enumerate(sorted_els):
        print(f"\n[{i + 1}/{len(sorted_els)}] Element: {element.name} (id={element.id}, order={element.order})")
        extract_element(element, graph, logger, use_verify)


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
    t_bg = time.time()
    print("  [bg] background extraction started")

    best_bg = None
    best_defects = None

    for attempt in range(1, BACKGROUND_RETRIES + 1):
        print(f"\n  [bg] Attempt {attempt}/{BACKGROUND_RETRIES} started")
        t_att = time.time()

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
            print(f"  [bg] attempt {attempt} aborted after {_fmt_duration(time.time() - t_att)} — generation failed")
            continue

        # Verify
        verification = _verify_background(graph, original_img, generated, logger)
        ok = verification.get("ok", False)
        defects = verification.get("defects", [])
        notes = verification.get("notes", "")
        print(f"  Background verification: ok={ok}, defects={defects}, notes={notes}")

        best_bg = generated
        best_defects = defects

        print(f"  [bg] attempt {attempt} finished in {_fmt_duration(time.time() - t_att)}")
        if ok:
            break

    graph.background.attempts = attempt
    print(f"  [bg] retry loop finished — best attempt {attempt}, total {_fmt_duration(time.time() - t_bg)}")

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

    # Build layer stack: back-to-front (ascending order = drawn later on top)
    sorted_els = graph.back_to_front()
    layers = []
    for el in sorted_els:
        if el.layer_path and Path(el.layer_path).exists() and el.status != ElementStatus.FAILED:
            layer_img = Image.open(el.layer_path)
            layers.append((layer_img, el.bbox))
            print(f"  Layer: {el.id} ({el.name}, order={el.order})")
        elif el.status == ElementStatus.FAILED:
            layer_img = Image.open(el.layer_path)
            layers.append((layer_img, el.bbox))
            print(f"  Notice: Using failed element: Layer: {el.id} ({el.name}, order={el.order})")

    reconstruction = composite(bg_path, layers)
    recon_path = logger.save_image(reconstruction, "reconstruction")
    print(f"  Reconstruction saved: {recon_path}")
    logger.save_scene_graph(graph, "stage4_reassembly")
    return reconstruction


def global_verify(graph: SceneGraph, reconstruction: Image.Image,
                  logger: RunLogger, use_global: bool) -> dict:
    """Stage 4, step 2: Global verification."""

    if not use_global:
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
            "id": el.id,
            "name": el.name,
            "bbox": el.bbox,
            "order": el.order,
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


def apply_routing(verdict: dict, graph: SceneGraph, logger: RunLogger,
                  use_verify: bool) -> bool:
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
                order=graph.max_order() + 1,   # new items land on top
                overlaps=[],
            )
            graph.elements.append(new_el)
            graph.enum_reopenings += 1
            print(f"  Added missing element: {new_el.name} id={new_el.id}")
            # Extract it immediately
            extract_element(new_el, graph, logger, use_verify)
            action_taken = True

    # 2. Handle bad layers — re-run that element's Stage 2
    bad_layers = verdict.get("bad_layers", [])
    for bad in bad_layers:
        elem_id = bad.get("id", "")
        defects = bad.get("defects", [])
        el = graph.get_element_by_id(elem_id)
        if el and el.attempts < ELEMENT_RETRIES:
            print(f"  Re-running bad layer: {elem_id} ({el.name}) defects={defects}")
            el.defects = defects
            el.status = ElementStatus.EXTRACTING
            extract_element(el, graph, logger, use_verify)
            action_taken = True

    # 3. Handle z-order
    reorder = verdict.get("reorder", [])
    for r in reorder:
        front_id = r.get("front", "")
        behind_id = r.get("behind", "")
        front_el = graph.get_element_by_id(front_id)
        behind_el = graph.get_element_by_id(behind_id)
        if front_el and behind_el:
            # Ensure front_el has the HIGHER order (drawn later = in front)
            if front_el.order < behind_el.order:
                # Swap orders
                print(f"  Fixing z-order: {front_id} should be in front of {behind_id}")
                front_el.order, behind_el.order = behind_el.order, front_el.order
                action_taken = True

    return action_taken


def run_stage4(graph: SceneGraph, logger: RunLogger, use_global: bool,
               use_verify: bool = False) -> bool:
    """Stage 4: Reassembly + global verification master loop.

    Returns True if final result is acceptable, False if budget exhausted.
    """
    while graph.global_attempts < GLOBAL_ATTEMPTS:
        graph.global_attempts += 1
        print(f"\n  --- Global attempt {graph.global_attempts}/{GLOBAL_ATTEMPTS} started ---")
        t_g = time.time()

        recon = reassemble(graph, logger)
        verdict = global_verify(graph, recon, logger, use_global)

        ok = verdict.get("ok", False)
        notes = verdict.get("notes", "")
        print(f"  Global verification: ok={ok}, notes={notes}")
        print(f"  Missing: {verdict.get('missing', [])}")
        print(f"  Bad layers: {verdict.get('bad_layers', [])}")
        print(f"  Reorder: {verdict.get('reorder', [])}")

        if ok:
            print(f"\n  [global] attempt {graph.global_attempts} finished in {_fmt_duration(time.time() - t_g)} — accepted")
            print("\n  >>> RECONSTRUCTION ACCEPTED")
            logger.save_scene_graph(graph, "final_accepted")
            return True

        # Apply routing
        action_taken = apply_routing(verdict, graph, logger, use_verify)
        if not action_taken:
            print(f"\n  [global] attempt {graph.global_attempts} finished in {_fmt_duration(time.time() - t_g)} — no routing actions")
            print("\n  >>> NO ROUTING ACTIONS — terminating loop")
            break

        print(f"  [global] attempt {graph.global_attempts} finished in {_fmt_duration(time.time() - t_g)} — routing applied, looping")

    # Budget exhausted
    print(f"\n  >>> BUDGET EXHAUSTED ({graph.global_attempts}/{GLOBAL_ATTEMPTS}) — shipping best partial")
    graph.save(logger.run_dir / "final_best_partial.json")
    return False


# ---------------------------------------------------------------------------
# Dataset package — descriptions + order-labelled export
# ---------------------------------------------------------------------------
def compute_orders(graph: SceneGraph) -> list[tuple[int, Element]]:
    """Return (order, element) pairs for compositing/export.

    The planner's `order` is the one-and-only label: background is 0 (kept in
    graph.background), elements are 1..N drawn back-to-front. Elements whose
    layer file is missing on disk are skipped, leaving their order slot empty.
    """
    ready = [
        e for e in graph.elements
        if e.layer_path and Path(e.layer_path).exists()
    ]
    ready.sort(key=lambda e: e.order)
    return [(e.order, e) for e in ready]


def describe(graph: SceneGraph, logger: RunLogger) -> None:
    """One joint VLM call filling all dataset descriptions.

    Produces: overall image description, background name+description,
    global_style, and a short per-element description — the metadata of
    target.json. Matched to elements by exact name.
    """
    print("\n" + "=" * 60)
    print("DESCRIBING LAYOUT ITEMS")
    print("=" * 60)

    ordered = compute_orders(graph)
    element_summaries = [
        {"order": order, "id": el.id, "name": el.name, "bbox": el.bbox}
        for order, el in ordered
    ]
    system_prompt = DESCRIBER_PROMPT.replace(
        "{element_summaries}", json.dumps(element_summaries))
    user_text = ("FIRST image is the original illustration; SECOND image is the "
                 "extracted background layer. Describe this illustration and its "
                 "layout items.")

    original_img = Image.open(graph.image_path)
    # Second image = the extracted background, so its description matches the
    # layer actually exported (foregrounds already removed) instead of guessing.
    bg_file = graph.background.image_path
    if bg_file and Path(bg_file).exists() and Path(bg_file) != Path(graph.image_path):
        image_input = [original_img, Image.open(bg_file)]
    else:
        image_input = original_img

    if pipeline_tools.FAKE_MODE:
        response_text = pipeline_tools._fake_vlm(system_prompt, image_input, user_text)
        result = parse_json_relaxed(response_text) or {}
    else:
        result = vlm_json(
            system_prompt=system_prompt,
            image_input=image_input,
            user_text=user_text,
            max_new_tokens=VLM_MAX_TOKENS_DESCRIBER,
            role="describer",
            logger=logger,
        )

    graph.image_description = result.get("description") or ""

    bg_info = result.get("background") or {}
    graph.background.name = bg_info.get("name") or "background"
    graph.background.description = bg_info.get("description") or ""

    style = result.get("global_style") or {}
    graph.global_style = {
        "color_scheme": style.get("color_scheme") or "",
        "mood": style.get("mood") or "",
    }

    desc_by_id = {
        item.get("id"): item.get("description") or ""
        for item in result.get("elements", [])
        if isinstance(item, dict) and item.get("id")
    }
    missing = 0
    for _, el in ordered:
        el.description = desc_by_id.get(el.id) or el.description or ""
        if not el.description:
            missing += 1
            print(f"  WARNING: no description returned for '{el.id}'")

    if not graph.image_description:
        print("  WARNING: no overall image description returned")

    print(f"  Described {len(ordered) - missing}/{len(ordered)} elements")
    logger.log_text(json.dumps({
        "image_description": graph.image_description,
        "background_name": graph.background.name,
        "background_description": graph.background.description,
        "global_style": graph.global_style,
        "elements": {el.id: el.description for _, el in ordered},
    }, indent=2, ensure_ascii=False), label="descriptions")
    logger.save_scene_graph(graph, "described")


def export_dataset(graph: SceneGraph, logger: RunLogger) -> Path:
    """Write the finished per-image dataset package into the run folder.

    Package layout (<run_dir>/package/):
      <stem>.json      summary in target.json format (file_name, description,
                       layout[{order,name,bbox,description}], global_style)
      background.png   order 0
      element01.png    order 1..N, back-to-front — order is the only label
      <original file>  copy of the input illustration (what file_name points to)
      reconstruction.png  QA copy of the final composite
    """
    print("\n" + "=" * 60)
    print("EXPORTING DATASET PACKAGE")
    print("=" * 60)

    if graph.image_description is None:
        describe(graph, logger)

    pkg_dir = logger.run_dir / "package"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # 1. Original illustration — file_name points at this copy.
    src_image = Path(graph.image_path)
    file_name = src_image.name
    shutil.copy2(src_image, pkg_dir / file_name)

    # 2. Layout: order 0 = background.
    layout: list[dict] = []
    bg_path = graph.background.image_path
    if bg_path and Path(bg_path).exists():
        shutil.copy2(bg_path, pkg_dir / "background.png")
        layout.append({
            "order": 0,
            "name": graph.background.name or "background",
            "bbox": [0, 0, 1000, 1000],
            "description": graph.background.description or "",
        })
    else:
        print("  WARNING: background image missing — omitted from layout")

    # 3. Elements: element<order>.png (same label as the element id), back-to-front.
    for order, el in compute_orders(graph):
        dst_name = f"element{order:02d}.png"
        shutil.copy2(el.layer_path, pkg_dir / dst_name)
        layout.append({
            "order": order,
            "name": el.name,
            "bbox": list(el.bbox),
            "description": el.description or "",
        })
        print(f"  order {order:02d}: {el.id} ({el.name}) -> {dst_name}")

    # 4. Reconstruction copy for QA (not referenced by file_name).
    recon_candidates = sorted(logger.run_dir.glob("*reconstruction*.png"))
    if recon_candidates:
        shutil.copy2(recon_candidates[-1], pkg_dir / "reconstruction.png")

    # 5. The summary json, keys in target.json order.
    summary = {
        "file_name": file_name,
        "description": graph.image_description or "",
        "layout": layout,
        "global_style": graph.global_style or {"color_scheme": "", "mood": ""},
    }
    json_path = pkg_dir / f"{src_image.stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Package: {pkg_dir}")
    print(f"  Items: {len(layout)} (1 background + {len(layout) - 1} elements)")
    return pkg_dir


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
def run_pipeline(image_path: str | Path, output_dir: str = DEFAULT_OUTPUT_DIR,
                 use_fake: bool = False, use_verify: bool = False,
                 use_global: bool = False) -> dict:
    """Run the full layer decomposition pipeline on one image.

    Args:
        image_path: path to the illustration image.
        output_dir: where to write the run folder.
        use_fake: if True, use stubs for VLM and JoyAI (testing only).
        use_verify: if True, run the VLM verification loop on each extracted
            element cutout (slower, higher quality).
        use_global: if True, run the final global reconstruction verification
            and re-routing loop.

    Returns:
        dict with paths to all outputs.
    """
    pipeline_tools.set_fake_mode(use_fake)
    if use_fake:
        print("[PIPELINE] RUNNING IN FAKE/STUB MODE — no real model calls")

    t_run = time.time()
    print(f"[PIPELINE] Run started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Resolve JoyAI's GPU before any model loading (only in non-fake mode).
    # The VLM runs remotely via API, so a single local GPU suffices.
    if not use_fake:
        pipeline_tools.resolve_joyai_device()

    if use_verify:
        print("Element verification ENABLED — VLM checks each extracted cutout.")

    if use_global:
        print("Global verification ENABLED — final reconstruction is re-checked.")

    # Derive image prefix from basename (e.g. "009" from "009.png")
    image_stem = Path(image_path).stem
    logger = RunLogger(output_dir, image_prefix=image_stem)
    print(f"[PIPELINE] Run dir: {logger.run_dir}")

    try:
        # Stage 1
        t_stage = time.time()
        graph = plan(image_path, logger)
        print(f"\n[PIPELINE] Stage 1 (planning) finished in {_fmt_duration(time.time() - t_stage)} "
              f"(elapsed {_fmt_duration(time.time() - t_run)})")

        if not graph.elements:
            print("[PIPELINE] No elements found in plan. Skipping extraction.")
            graph.background.status = BackgroundStatus.DONE
            graph.background.image_path = image_path  # use original as bg fallback
        else:
            # Stage 2
            t_stage = time.time()
            run_stage2(graph, logger, use_verify)
            print(f"\n[PIPELINE] Stage 2 (element extraction) finished in {_fmt_duration(time.time() - t_stage)} "
                  f"(elapsed {_fmt_duration(time.time() - t_run)})")

            # Stage 3
            t_stage = time.time()
            extract_background(graph, logger)
            print(f"\n[PIPELINE] Stage 3 (background) finished in {_fmt_duration(time.time() - t_stage)} "
                  f"(elapsed {_fmt_duration(time.time() - t_run)})")

        # Stage 4
        t_stage = time.time()
        run_stage4(graph, logger, use_global, use_verify)
        print(f"\n[PIPELINE] Stage 4 (reassembly) finished in {_fmt_duration(time.time() - t_stage)} "
              f"(elapsed {_fmt_duration(time.time() - t_run)})")

        # Dataset package (descriptions + order-labelled export)
        t_stage = time.time()
        pkg_dir = export_dataset(graph, logger)
        print(f"\n[PIPELINE] Dataset export finished in {_fmt_duration(time.time() - t_stage)} "
              f"(elapsed {_fmt_duration(time.time() - t_run)})")

        # Ship
        result = ship(graph, logger)
        result["package_dir"] = str(pkg_dir)
        result["elapsed_seconds"] = time.time() - t_run
        print(f"\n[PIPELINE] COMPLETE — total time {_fmt_duration(time.time() - t_run)}")
        return result

    except Exception as e:
        print(f"\n[PIPELINE] ERROR after {_fmt_duration(time.time() - t_run)}: {e}")
        traceback.print_exc()
        logger.log_text(traceback.format_exc(), "pipeline_error")
        raise