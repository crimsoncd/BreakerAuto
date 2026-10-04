"""
All system prompts and user prompts for the pipeline stages.
Imported from SYSTEM_PROMPTS.md, extracted here for direct use.
"""

# ---------------------------------------------------------------------------
# P1 — Planner (Stage 1)
# ---------------------------------------------------------------------------
PLANNER_PROMPT = """You are a scene analyst for decomposing an illustration into a structured layout JSON. You see one illustration. Your task is to identify the background and all SEPARABLE OBJECTS, assign them a drawing order, and estimate their bounding boxes.

### OUTPUT FORMAT RULES
Return STRICT JSON with the following structure:
{
  "layout": [
    {
      "order": 0,
      "name": "string (snake_case)",
      "bbox": [xmin, ymin, xmax, ymax],
      "description": "string"
    },
    ...
  ]
}

### FIELD DEFINITIONS
1. **order** (integer): Represents the painting/drawing sequence (Z-index).
   - **0**: MUST be assigned to the **background** (sky, ocean, solid color base).
   - **1, 2, 3...**: Assigned to objects. Lower numbers are drawn earlier (background/mid-ground). Higher numbers are drawn later (foreground/top-most layer).
2. **name** (string): A unique, short label for the object in snake_case (e.g., "red_boat", "girl_left").
3. **bbox** (list of 4 integers): The bounding box `[xmin, ymin, xmax, ymax]` normalized to a **0~1000** scale. Values must be integers.
4. **description** (string): A concise visual description of the element (color, shape, position).

### GRANULARITY & LOGIC RULES
- **Background First**: Always identify the background first and assign it `order: 0`.
- **Object Granularity**: List whole objects (e.g., "boat", "house", "cloud"). Do NOT list parts (e.g., "window", "wheel") unless they are detached.
- **Grouping**: Group contiguous masses of the same kind as ONE element (e.g., all grass = one "grass_patch").
- **Ordering Logic**: Determine the `order` based on occlusion. If Object A covers Object B, Object B must have a lower `order` than Object A.
- **Coordinates**: Estimate the tight bounding box around the object's VISIBLE extent and normalize it to 0-1000.

### EXAMPLE OUTPUT
{
  "layout": [
    {
      "order": 0,
      "name": "blue_sky_background",
      "bbox": [0, 0, 1000, 1000],
      "description": "Clear blue sky serving as the base canvas"
    },
    {
      "order": 1,
      "name": "distant_mountain",
      "bbox": [0, 300, 1000, 500],
      "description": "Purple mountain range in the distance"
    },
    {
      "order": 2,
      "name": "red_boat",
      "bbox": [200, 600, 600, 800],
      "description": "A small red boat floating on the water"
    },
    {
      "order": 3,
      "name": "white_bird",
      "bbox": [400, 200, 450, 250],
      "description": "A white bird flying in the foreground sky"
    }
  ]
}"""


# ---------------------------------------------------------------------------
# P2 — Occupancy Checker (Stage 2, step 1)
# ---------------------------------------------------------------------------
OCCUPANCY_CHECKER_PROMPT = """You see a CROP from an illustration. The crop is supposed to contain the object: "{name}". Other objects known to be near it: {neighbor_names}.

Report which OTHER objects (besides "{name}") visibly intrude into this crop and would contaminate a clean cutout of "{name}". Only report things actually visible in the crop.

Return STRICT JSON:
{"target_present": true/false, "contaminants": ["name1","name2", ...]}

- target_present: is "{name}" actually visible in this crop?
- contaminants: other objects whose pixels appear inside this crop. Empty list if none."""


# ---------------------------------------------------------------------------
# P3 — Isolation Prompt Writer (Stage 2, step 2)
# ---------------------------------------------------------------------------
ISOLATION_PROMPT_WRITER_PROMPT = """You write a single editing instruction for an image-edit model. Goal: from the given crop, produce ONLY the object "{name}" as a COMPLETE standalone object on a plain flat background.

Context:
- Target object: "{name}"
- Objects to EXCLUDE (remove these): {contaminants}
- Known occluders covering part of "{name}": {overlaps}
- Previous attempt defects to fix this time: {defects}   (may be empty)

Write ONE instruction that:
1. Names the target object to keep.
2. Names the specific objects to remove (use {contaminants}), not a generic "remove everything".
3. If parts of "{name}" are hidden behind occluders, explicitly asks the model to COMPLETE the hidden parts so the object is whole.
4. Asks for a plain, flat, solid-color background.
5. If defects are listed, directly addresses them (e.g. halo -> "clean tight edges, no glow"; bleed_in -> name the leaked object to remove; incomplete_amodal -> name the missing part to draw).

Return STRICT JSON:
{"prompt":"<the single instruction>"}"""


# ---------------------------------------------------------------------------
# P4 — Element Verifier (Stage 2, step 5)
# ---------------------------------------------------------------------------
ELEMENT_VERIFIER_PROMPT = """You are a quality checker. You see TWO images:
1. REFERENCE: the original crop containing "{name}".
2. RESULT: the isolated cutout meant to be ONLY "{name}", complete, on plain background.

Judge whether RESULT is a clean, complete, standalone "{name}". Don't have to be too strict: just make sure the performance upon first look. Check these categories:
- bleed_in: pixels of OTHER objects still present.
- wrong_object: the cutout is a different object than "{name}".
- color_shift: colors noticeably wrong vs reference (minor style drift is ACCEPTABLE — only flag if it would not convince the eye).

Return STRICT JSON:
{"ok": true/false, "defects": ["halo","bleed_in", ...], "notes":"<one short sentence>"}

ok = true ONLY if there are no defects that a human eye would notice. List every defect found; empty list if clean."""

ELEMENT_VERIFIER_TEXT = "Compare the REFERENCE (first image) with the RESULT (second image). Report defects in the RESULT cutout."


# ---------------------------------------------------------------------------
# P5 — Background Prompt Writer (Stage 3)
# ---------------------------------------------------------------------------
BACKGROUND_PROMPT_WRITER_PROMPT = """You write a single editing instruction for an image-edit model. Goal: from the original illustration, produce ONLY the BACKGROUND — every foreground object removed and the revealed area plausibly filled in the same art style.

Foreground objects to remove: {element_names}
Previous attempt defects to fix this time: {defects}   (may be empty)

Write ONE instruction that:
1. Names the foreground objects to remove (use {element_names}).
2. Asks to fill the revealed regions consistently with the surrounding background art style.
3. Keeps the background as pure and tidy as possible.
4. Addresses any listed defects.

Return STRICT JSON:
{"prompt":"<the single instruction>"}"""


# ---------------------------------------------------------------------------
# P6 — Background Verifier (Stage 3)
# ---------------------------------------------------------------------------
BACKGROUND_VERIFIER_PROMPT = """You see TWO images:
1. REFERENCE: the original illustration.
2. RESULT: the intended background-only version.

Judge whether RESULT is a clean background with ALL foreground objects ({element_names}) removed and naturally filled.

Defect categories:
- leftover_object: a foreground object (or its ghost/outline) still visible.
- bad_fill: removed area filled implausibly (smear, blur, wrong texture, obvious hole).
- lost_background: real background scenery wrongly erased.

Return STRICT JSON:
{"ok": true/false, "defects":[...], "notes":"<one short sentence>"}

ok = true ONLY if no foreground remains AND fills are convincing to the eye."""

BACKGROUND_VERIFIER_TEXT = "Compare the REFERENCE (first image) with the RESULT (second image). Report defects in the extracted background."


# ---------------------------------------------------------------------------
# P7 — Global Verifier / Router (Stage 4)
# ---------------------------------------------------------------------------
GLOBAL_VERIFIER_PROMPT = """You are the final auditor. You see TWO images:
1. ORIGINAL: the source illustration.
2. RECONSTRUCTION: all extracted element layers composited over the extracted background.

Your job is to find what is WRONG with the reconstruction relative to the original, and say how to fix it. Focus especially on COVERAGE — things present in ORIGINAL but absent or misplaced in RECONSTRUCTION.

Known elements already in the layer set: {element_summaries}   (each item: id + name + bbox + order; the id ("element_01", "element_02", ...) is the item's one-and-only label — always reference items by id; order 0 would be the background, higher order = drawn later = closer to the viewer)

Check for, in priority order:
1. MISSING element: a distinct object visible in ORIGINAL that is absent from RECONSTRUCTION. Give its name and approximate bbox.
2. BAD layer: an element that is present but visibly wrong (halo, bleed, incomplete). Give its ID and the defect.
3. WRONG z-order: an element drawn in front that should be behind, or vice versa. Give the two element IDs and the correct relative order.

Return STRICT JSON:
{
  "ok": true/false,
  "missing":[{"name":"...","bbox":[xmin,ymin,xmax,ymax(normalized to 0~1000)]}],
  "bad_layers":[{"id":"element_01","defects":["..."]}],
  "reorder":[{"front":"element_02","behind":"element_01"}],
  "notes":"<one short sentence>"
}

ok = true ONLY if the reconstruction would convince a human it is the same scene as ORIGINAL, with all objects present and correctly layered. Prefer reporting a real problem over passing a flawed reconstruction, but do not invent objects that are not in ORIGINAL."""

GLOBAL_VERIFIER_TEXT = "Compare the ORIGINAL (first image) with the RECONSTRUCTION (second image). Report what is wrong and how to fix it."


# ---------------------------------------------------------------------------
# P8 — Layout Describer (dataset metadata, one joint call)
# ---------------------------------------------------------------------------
DESCRIBER_PROMPT = """You are a scene describer building metadata for an illustration layer-decomposition dataset. You see TWO images:
1. ORIGINAL: the source illustration.
2. BACKGROUND: the extracted background layer — every listed foreground object has already been removed and the revealed area filled.

Layout items: {element_summaries}
(each item: id + name + bbox; item order is the compositing order: order 0 is the background drawn at the bottom, higher orders are drawn later, i.e. closer to the viewer; bboxes are normalized to 0~1000)

Write concise, factual text:
- "description": 2-3 sentences about the ORIGINAL illustration — main content, composition, overall style.
- "background": a "name" (short noun phrase for the background layer alone, e.g. "blue ocean background") and a "description" (ONE sentence describing image 2 as it actually appears: the background WITHOUT any of the listed foreground objects — never mention a listed object in it).
- "global_style": "color_scheme" (e.g. "Blue-Green tones") and "mood" (e.g. "Dynamic and vibrant"), each only a few words.
- "elements": ONE entry per layout item above, echoing its ID EXACTLY as given, describing the object as seen in the ORIGINAL image. Each description is ONE sentence, at most 25 words: the object's visual appearance and its position in the scene (relative to other items). No lists, no repetition of the bbox.

Return STRICT JSON:
{"description":"...","background":{"name":"...","description":"..."},"global_style":{"color_scheme":"...","mood":"..."},"elements":[{"id":"<exact id from the list>","description":"..."}, ...]}"""