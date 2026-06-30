# Build Instructions — Agentic Layer Decomposition for Illustrations

> Purpose of this doc: enough specification for a coding agent to build a **runnable v0** of the pipeline, then iterate. Optimize for *getting end-to-end first*, not for completeness. Stub the hard loops, wire the happy path, then close loops one at a time.

---

## 1. What we are building

An agent that takes a single artistic illustration and decomposes it into:
- one **background** layer (foreground objects removed, amodally filled), and
- N **element** layers, each a *complete standalone object* (object-level granularity: "girl", "grass", "river" — never "hair", "hat"), amodally completed where occluded, on transparent alpha.

Quality bar: **convinces the human eye**, not pixel-perfect. We accept that a generated/completed element may differ slightly from the original.

Reconstruction test: compositing all elements front-to-back over the background should visually reproduce the original.

## 2. Models / runtime

- **VLM**: Qwen3VL-32B — all perception and judgment (planning, checks, prompt-writing). One model, many roles, differentiated only by system prompt (see `SYSTEM_PROMPTS.md`).
- **Edit model**: JoyAI — all generative isolation, amodal completion, and background fill.
- **DIP**: classical CV (OpenCV/PIL) — plain-bg → alpha matting, cropping, resizing, compositing. No ML.
- **Hardware**: 4× A100 80G. Models can be held resident; calls are local. Treat both model calls as expensive though — budget them. See `MODELS_AND_RESOURCES.md` for detail.

## 3. The central data contract — the Scene Graph

Every stage reads and writes ONE shared object. Define it first; everything else depends on it.

```
SceneGraph
  image_path: str
  image_size: (W, H)
  background: { prompt: str|null, image_path: str|null, status: enum, attempts: int }
  elements: [ Element ]
  global_attempts: int

Element
  id: str                      # stable unique handle, e.g. "girl_01" — assigned once, never reused
  name: str                    # object-level label
  bbox: [xmin, ymin, xmax, ymax]           # from planner
  depth_rank: int              # 1 = frontmost
  overlaps: [element_id, ...]  # which other elements this one overlaps
  isolation_prompt: str|null
  layer_path: str|null         # final RGBA cutout
  status: enum {planned, extracting, done, failed}
  attempts: int
  defects: [str]               # last verifier findings
```

**`id` is sacred.** It is the single handle that keeps one object coherent across planning → prompting → generation → checking. Never let a stage silently re-resolve "the girl" by name; always carry the `id`.

Always remember bbox is in [xmin, ymin, xmax, ymax] format, the numbers normalized to 0~1000. So you should resize the bbox whenever use it to an image.

## 4. Pipeline stages

### Stage 1 — Planning
- Input: original image.
- Call VLM (Planner prompt) → returns the full element list with `name`, `bbox`, `depth_rank`, `overlaps`.
- Output: populated SceneGraph.
- Enforce object-level granularity via the prompt's few-shot examples.
- This stage is **reopenable** by Stage 4 (to add a missed element).

### Stage 2 — Element extraction (loop, process front-to-back by depth_rank)
For each element, in depth order:
1. **Occupancy check** (VLM): crop bbox (×1.1), ask whether other named elements intrude into this crop. Output: list of contaminant names. *(This replaces the bbox-tightness check — we trust the box, we check for contamination.)*
2. **Write isolation prompt** (VLM): given the element name, the crop, and the contaminant list, produce a JoyAI instruction that (a) isolates THIS object, (b) names the specific things to exclude, (c) requests amodal completion of any occluded parts.
3. **Generate** (JoyAI): run the isolation prompt on the crop → isolated object on plain bg.
4. **Matte** (DIP): plain-bg → alpha; resize to bbox.
5. **Verify** (VLM): compare result against the ORIGINAL crop as reference; report defects by category {halo, bleed_in, missing_part, wrong_object, color_shift, incomplete_amodal}. If clean → `done`. If defective → feed defects back into step 2 and retry. Max 3 attempts. On exhaustion → keep best attempt, mark `failed`, **do not drop**.

### Stage 3 — Background (runs AFTER all elements)
- Conditioned on the known element list: prompt JoyAI to remove *named* foregrounds ("remove the girl, the boat, the reeds") and fill, rather than a generic "remove all foreground".
- Same verify/retry loop (max 3), reference = original image.

### Stage 4 — Reassembly + global verification (master loop)
1. Composite: background at bottom, then elements by depth_rank (back-to-front when drawing).
2. **Global verify** (VLM): compare reconstruction to original. Route:
   - region in original but missing from reconstruction → **reopen Stage 1**: add the missed element (new `id`), run it through Stage 2.
   - a specific layer reads wrong → re-run that element's Stage 2 loop with the noted defect.
   - z-order wrong → adjust depth_rank and recomposite only.
3. Terminate when reconstruction acceptable OR `global_attempts` budget hit. On budget exhaustion, ship best partial + flag unresolved elements.

## 5. Tool interface (build these as plain functions; no agent framework needed for v0)

```
vlm(system_prompt, image(s), user_text) -> str        # the only VLM entrypoint
joyai(image, prompt) -> image                          # the only edit entrypoint
crop(image, bbox, pad=0.1) -> image
matte_to_alpha(image_on_plain_bg) -> rgba              # DIP
resize(image, size) -> image
composite(background, [layers_in_z_order]) -> image
```

Parse VLM JSON outputs strictly; on parse failure, one reformat-retry, then treat as empty/clean default and log loudly.

## 6. Control flow

```
graph = plan(image)                          # Stage 1
for el in sort_by_depth(graph.elements):     # Stage 2
    extract(el, graph)                        #   with internal 3x retry
graph.background = extract_background(graph)  # Stage 3
while global_attempts < BUDGET:              # Stage 4
    recon = reassemble(graph)
    verdict = global_verify(recon, original)
    if verdict.ok: break
    apply_route(verdict, graph)              # may reopen Stage 1 / re-run an element / reorder
ship(graph)
```

## 7. Budgets (make these config constants from day 1)

- `ELEMENT_RETRIES = 3`
- `BACKGROUND_RETRIES = 3`
- `GLOBAL_ATTEMPTS = 3`
- `MAX_ENUM_REOPENINGS = 3`  *(hard cap so planner↔checker can't oscillate forever)*
- `MAX_ELEMENTS = 20` *(sanity guard against runaway enumeration)*

## 8. Logging (non-negotiable, build it in v0)

Because the pipeline is non-deterministic, every result must be traceable. For each image, write a run folder containing: the SceneGraph after each stage (JSON), every VLM prompt+response, every JoyAI input crop + output, every intermediate matte, and every reconstruction. A bad output must be traceable to the stage and call that caused it. Set seeds where the models allow.

## 9. BUILD ORDER (do it in this sequence — runnable first)

1. **Scene graph + tool stubs.** Define the data structures and the 6 functions, with `joyai` and `vlm` returning canned outputs. Get the control flow running end-to-end on fakes.
2. **Stage 1 real.** Wire the real Planner VLM call; eyeball the scene graph on 3–5 test images.
3. **Stage 2 happy path, NO retry.** One pass: occupancy → prompt → generate → matte → resize. Skip verification. Just produce layers.
4. **Stage 3 happy path.** Background generation, no retry.
5. **Stage 4 composite, NO global verify.** Just reassemble and save. *Now you have a runnable v0 end-to-end.* Compare collages by eye.
6. **Add verification loops** one at a time: element verify (3.→5.), then background, then global. Each loop is independent — add, test, keep.
7. **Add Stage-4 routing** (reopen enumeration / reorder) last — it's the trickiest and benefits from everything else being stable.

Stop after step 5 and look at real outputs before building the loops. The loops should be designed against the *actual* failures you see, not the ones we predicted.

## 10. Known pitfalls to watch during testing (where this tends to break)

- **Coverage / missed elements** is the worst failure and the least observable — it only surfaces at Stage 4. Watch whether the global checker actually *notices absences*. This is the #1 thing to validate early.
- **Identity drift across stages** — confirm the same `id` resolves to the same object in plan, prompt, generation, and check. Duplicates ("two girls") are the stress test.
- **Local-vs-global threshold conflict** — an element can pass its local check but still make the global reconstruction wrong. If global keeps failing on "passed" elements, the local checker is too lenient. Tune them to agree.
- **Amodal completion on heavy occlusion** — invented occluded pixels have no ground truth; expect a low-quality tail no retry fixes. Consider flagging rather than infinitely retrying.
- **Self-eval optimism** — the VLM grading its own pipeline skews lenient on faint halos/bleed. Always pass the original as reference; always ask for defect categories, never good/bad.
- **Loop convergence / cost** — nested retries × 32B calls can blow up latency and may not converge. Budgets above are the guardrail; honor them and ship best-partial.
