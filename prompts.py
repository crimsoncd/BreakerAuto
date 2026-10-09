"""
All system prompts and user prompts for the pipeline stages.
Imported from SYSTEM_PROMPTS.md, extracted here for direct use.
"""

# ---------------------------------------------------------------------------
# P1 — Planner (Stage 1)
# ---------------------------------------------------------------------------
PLANNER_PROMPT = """You are a scene analyst decomposing an illustration into a structured layout JSON that a painter will use to repaint the image layer by layer. You see one illustration. Your task: identify the background and every drawable element unit, assign a drawing order, and estimate bounding boxes.

### CORE MENTAL MODEL — THE PAINTER PROTOCOL
Imagine a painter repainting this image on an empty canvas strictly in `order` sequence: order 0 first, then 1, 2, ... When the painter starts element k, the canvas already contains exactly elements 0..k-1 and nothing else. All rules below follow from this model; keep it in mind while writing every field.

### OUTPUT FORMAT
Return ONLY a STRICT JSON object with this structure:
{
  "layout": [
    {
      "order": 0,
      "name": "string",
      "bbox": [xmin, ymin, xmax, ymax],
      "description": "string"
    },
    ...
  ]
}

### FIELD DEFINITIONS
1. **order** (integer): drawing sequence (Z-index). `0` MUST be the background. `1, 2, 3...`: elements; lower = painted earlier (behind), higher = painted later (on top). Use consecutive integers 0..N-1, each exactly once.
2. **name** (string): a short, unique, plain-language label for the element (e.g., "crab", "girl holding seedlings"). No casing or formatting constraint; spaces are allowed.
3. **bbox** (list of 4 integers): tight bounding box `[xmin, ymin, xmax, ymax]` of the element's full visible extent, normalized to a 0-1000 scale. All four values MUST be integers within [0, 1000]. If held/worn items are merged into this element (see POSSESSION RULE), the bbox MUST include them.
4. **description** (string): a self-contained painting instruction for this layer only (see DESCRIPTION RULES).

### ELEMENT UNITS — WHAT COUNTS AS ONE ELEMENT
- **Background first**: the canvas base (solid color, sky, ocean) is always one element with `order: 0`.
- **Whole objects, not parts**: list whole objects ("boat", "house"); do NOT list attached parts ("wheel", "window") separately.
- **POSSESSION RULE — holder and held are ONE element**: anything gripped, held, worn, or carried by a character or animal (a bundle in the hands, a tool, a hat, a bag) is NOT a separate element. Merge it into the holder: mention it in the holder's name/description ("girl holding a bundle of rice seedlings in both hands") and include it in the holder's bbox. List an object separately only if no hand or body part grips it and it stands on its own in the scene (e.g., a plant rooted in the ground that nobody touches).
- **Interaction without possession**: if two independent elements merely touch or overlap (neither holds the other), keep them separate and resolve their order by occlusion.
- **Grouping**: a contiguous mass of the same kind with no holder = ONE element (e.g., one grass patch = "grass patch"). The same kind of object held by two different holders merges into each holder respectively, never into one shared entry.

### ORDER RULES (Z-INDEX)
- **Occlusion**: if element A visibly covers element B, then B.order < A.order.
- Held/worn items share their holder's order (they are the same element).
- If occlusion gives no clue, use depth intuition: farther/behind = lower order.

### DESCRIPTION RULES — CAUSALITY (NO FORWARD REFERENCES)
Each description is the note handed to the painter at the moment this layer is painted. The painter has only seen layers with lower order. Therefore:
1. **Self-contained**: identify the element by its own appearance (color, shape, size, pose, clothing) and by ABSOLUTE canvas position ("left edge", "lower-right quadrant", "center"). Never define an element through another element that is not on the canvas yet.
2. **Backward references allowed**: you MAY mention another element only if its order is strictly LOWER (already painted), typically to state occlusion ("painted over the yellow circle", "partly covering the mountain").
3. **Forward references forbidden**: NEVER mention, name, or presuppose any element with an equal or higher order (not yet painted). If you feel tempted to write "held by the girl on the right", then either the object is held by her -> merge it into her element (POSSESSION RULE), or replace the reference with absolute position ("on the right side of the canvas").
4. **Mention merged content**: a holder's description MUST state what it holds/wears/carries, because those pixels are painted in this same layer.
5. **One layer, one subject**: describe only this element; do not restate the whole scene.

### SELF-CHECK (run silently before outputting; fix any violation)
1. Is any listed element gripped/worn/carried by another listed element? -> merge it into the holder, delete the separate entry, and expand the holder's bbox and description.
2. For each element in ascending order: does its description mention another element? If yes, is that element's order strictly lower? -> if not, rewrite with absolute position or merge.
3. Occlusion consistency: for every visible overlap, does the covering element have the higher order?
4. bbox: four integers, all within 0-1000, tight around the visible extent including merged held items?
5. orders: consecutive 0..N-1 with background = 0? names unique?

### COMMON MISTAKES (DO NOT)
- Listing "seedlings in the girl's hands" as an element separate from the girl. -> Merge: "girl holding seedlings".
- Writing in an early layer's description "...held by the girl on the right" when the girl is a later layer. -> Forward reference; merge or use absolute position.
- Coordinates outside 0-1000.
- Describing relations to not-yet-painted layers ("under the bird added later").

### EXAMPLE OUTPUT
{
  "layout": [
    {
      "order": 0,
      "name": "blue sky",
      "bbox": [0, 0, 1000, 1000],
      "description": "Flat clear blue sky filling the entire canvas as the base layer"
    },
    {
      "order": 1,
      "name": "distant mountain range",
      "bbox": [0, 300, 1000, 500],
      "description": "Purple mountain range spanning the middle of the canvas, painted over the sky"
    },
    {
      "order": 2,
      "name": "fisherman in red boat holding a net",
      "bbox": [200, 560, 600, 800],
      "description": "Small red boat with a fisherman holding a brown fishing net in his hands, floating on the water in front of the mountain range"
    },
    {
      "order": 3,
      "name": "white bird",
      "bbox": [400, 200, 450, 250],
      "description": "White bird with spread wings flying in the upper-center sky, above the mountain range"
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
# P5 — Background Prompt Writer (Stage 3, model route)
# ---------------------------------------------------------------------------
BACKGROUND_PROMPT_WRITER_PROMPT = """You write a single editing instruction for an image-edit model. Goal: from the original illustration, produce ONLY the BACKGROUND — every foreground object removed and the revealed area filled.

Foreground objects to remove: {element_names}
Background analysis: {bg_type} ("flat" = a single solid background color; "textured" = real scenery/gradient)
Measured background color: {bg_color} (median color of the image outside the element boxes; "unknown" if not measurable)
Previous attempt defects to fix this time: {defects}   (may be empty)

Write ONE instruction that:
1. Names the foreground objects to remove (use {element_names}).
2. ANCHORING (critical — the model tends to re-invent the background otherwise):
   - If the background is flat: state the EXACT measured color (use {bg_color}) and demand the entire output be that flat solid color — no objects, no shadows, no outlines, no color drift.
   - If the background is textured: demand that the already-visible background areas stay EXACTLY unchanged (same colors, texture, lighting and style) and that ONLY the removed object regions are filled, by extending the surrounding background naturally.
3. Addresses any listed defects directly (e.g. leftover_object -> "ensure absolutely no remnant or ghost outline of X remains").

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