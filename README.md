# Layer Decomposition Pipeline

An agentic pipeline that takes a single artistic illustration and decomposes it into a **background** layer plus N **element** layers (object-level cutouts), then reassembles them into a reconstruction of the original.

- Perception & judgment (planning, prompt-writing, verification) → **qwen3.7-flash** via the Bailian (DashScope) OpenAI-compatible API — no local GPU needed
- Generative isolation / amodal completion / background fill → **JoyAI** (image-edit model, local GPU)
- Matting, resizing, compositing → classical CV (Pillow / rembg), no ML

For the full design spec (scene-graph data contract, stage breakdown, budgets, known pitfalls), see [`AGENT_BUILD_INSTRUCTIONS.md`](AGENT_BUILD_INSTRUCTIONS.md). The system prompts for every agent role live in [`prompts.py`](prompts.py) (originally [`SYSTEM_PROMPTS.md`](SYSTEM_PROMPTS.md)).

---

## Pipeline at a glance

```
Stage 1  Planning        VLM returns layout -> SceneGraph (id=element_XX, name, bbox, order, overlaps)
Stage 2  Extraction      per element: occupancy check -> isolation prompt -> JoyAI generate
                         -> matte to alpha -> resize -> [optional VLM verify + retry, 3x]
Stage 3  Background      JoyAI removes named foregrounds and fills, [optional verify + retry, 3x]
Stage 4  Reassembly      composite layers over background -> [optional global verify + routing loop, 3x]
```

Every stage reads and writes one shared [`SceneGraph`](scene_graph.py) object; every intermediate image, prompt, and response is logged to a timestamped run folder (see [`logger.py`](logger.py)).

---

## Requirements

- **Hardware**: one GPU with enough VRAM for JoyAI (~80G-class). The VLM runs remotely through the Bailian API, so it needs no local GPU and no co-residence constraint. (The reference box was 4× A100 80G; spare cards can host a second JoyAI worker to parallelize element extraction.)
- **API key**: a Bailian API key in the git-ignored `.env` file as `BAILIAN_API_KEY` (loaded at import time; see `config.py`).
- **Model / code**, pointed at from `config.py`:
  - The **JoyAI-Image** release checkpoint (`JOYAI_CKPT_ROOT`) plus a local clone of the **JoyAI-Image source repo** (`JOYAI_SRC_DIR`, loaded via its `infer_runtime` / `modules` code — the original deployment; the diffusers `JoyImageEditPipeline` route was tried and impaired quality, so it was reverted).
  - That source repo additionally needs `flash_attn` installed in the environment.
- **Python 3.10+** with the packages in [`requirements.txt`](requirements.txt) (or install this repo as a package via `pip install -e .`).
- **Reference environment**: the conda env `JoyNew` (`/remote-home/Zhangkaile/miniconda3/envs/JoyNew/bin/python`) — it ships the `openai` client and the torch/diffusers stack the wrappers rely on (see [`MODELS_AND_RESOURCES.md`](MODELS_AND_RESOURCES.md)).

---

## Setup

### 1. Install this repo

```bash
git clone https://github.com/crimsoncd/BreakerAuto.git
cd BreakerAuto
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt       # or: pip install -e .   (builds the package, adds the `decompose` CLI)
```

### 2. Configure your local resources

All machine-specific settings live in [`config.py`](config.py). **Either** set
the environment variables **or** edit `config.py` directly — don't do both:

```bash
# .env (git-ignored) — Bailian API key for the VLM:
#   BAILIAN_API_KEY=sk-...
export JOYAI_CKPT_ROOT=/path/to/JoyAI-Image-Edit          # checkpoint root
export JOYAI_SRC_DIR=/path/to/JoyAI-Image                 # clone of the JoyAI-Image repo
export JOYAI_DEVICE=cuda:2                          # optional, else auto-detected
```

Editing `config.py` instead looks like this:

```python
JOYAI_CKPT_ROOT = "/path/to/JoyAI-Image-Edit"
JOYAI_SRC_DIR = "/path/to/JoyAI-Image"
```

   > Only JoyAI runs locally, so a single GPU suffices. If you don't set `JOYAI_DEVICE`, the pipeline picks the card with the most free memory at startup.

---

## Usage

> If you installed this repo as a package (`pip install -e .`), the `decompose`
> command is equivalent to `python main.py` below.

### Single image

```bash
python main.py --image images/009.png
```

### Batch

```bash
python main.py --dir images/ --output runs
```

### Fake / stub mode (no GPUs, no models)

Useful for wiring the control flow and inspecting the logging:

```bash
python main.py --image images/009.png --fake
```

### Verification flags

The two VLM verification loops are **off by default** for speed; turn them on only when you need the quality gate:

| Flag | Effect |
|------|--------|
| `--use_verify` | VLM checks each extracted element cutout against the original crop; retries (up to `ELEMENT_RETRIES`) with defects fed back into the next isolation prompt. |
| `--use_global` | After reassembly, the VLM compares the reconstruction to the original and routes: re-extract a bad layer, add a missing element, or fix z-order. Retries up to `GLOBAL_ATTEMPTS`. |

```bash
python main.py --image images/009.png --use_verify --use_global
```

> These flags replace the old `--skip_verify` / `--skip_global` switches: behavior is inverted (default = not verified), and `run_pipeline(...)` now takes `use_verify` / `use_global`.

### Full CLI reference

```
--image PATH      single illustration to process
--dir PATH        batch: process every image in a directory
--output PATH     run-folder root (default: runs)
--fake            stub the VLM and JoyAI calls (testing only)
--use_verify      enable per-element VLM verification
--use_global      enable final global reconstruction verification
```

---

## Configuration reference

Everything below lives in [`config.py`](config.py). Env vars (if set) override the file values.

| Key | Env var | Default | Meaning |
|-----|---------|---------|---------|
| `VLM_BASE_URL` | `VLM_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI-compatible endpoint for the VLM |
| `VLM_API_KEY` | `BAILIAN_API_KEY` | — (from `.env`) | API key for the VLM endpoint |
| `VLM_MODEL_NAME` | `VLM_MODEL_NAME` | `qwen3.7-flash` | Model name served by the endpoint |
| `JOYAI_CKPT_ROOT` | `JOYAI_CKPT_ROOT` | `/remote-home/Zhangkaile/models/JoyAI-Image-Edit/` | JoyAI checkpoint dir (loaded via the JoyAI-Image repo code) |
| `JOYAI_SRC_DIR` | `JOYAI_SRC_DIR` | `/remote-home/Zhangkaile/dev/JoyAI-Image` | Root of the JoyAI-Image source repo (its `src` goes on `sys.path`) |
| `JOYAI_DEVICE` | `JOYAI_DEVICE` | auto-detect | GPU card for JoyAI (the only local model) |
| `JOYAI_BASE_SEED` | — | `42` | base seed; bumped on retries |
| `ELEMENT_RETRIES` | — | `3` | max per-element generation attempts |
| `BACKGROUND_RETRIES` | — | `3` | max background attempts |
| `GLOBAL_ATTEMPTS` | — | `3` | max global-verification rounds |
| `MAX_ELEMENTS` | — | `20` | sanity cap on elements per image |
| `DEFAULT_OUTPUT_DIR` | — | `runs` | default `--output` |

---

## Output layout

Each image produces a timestamped run folder, e.g. `runs/009_20260731_153000/`:

```
009_20260731_153000/
├── 0001_stage1_plan.json              # scene graph after planning
├── 0002_crop_element_01.png             # cropped input per element (labelled by id)
├── 0003_vlm_occupancy_element_01.json   # every VLM call logged (prompt + response)
├── ... joyai_*.png, gen_*, matte_*, layer_*.png
├── background_final.png               # Stage 3 result
├── reconstruction.png                 # Stage 4 composite
└── 00XX_final_shipped.json            # final scene graph
```

Every VLM prompt+response, every JoyAI input/output, and every intermediate matte is saved so any bad output is traceable to the stage and call that produced it.

---

## Utility scripts

- [`call_bailian_vlm.py`](call_bailian_vlm.py) — the VLM wrapper (Bailian API, qwen3.7-flash); also a small `__main__` sanity CLI: `python call_bailian_vlm.py --image in.png --prompt "..."`.
- [`call_JoyAI.py`](call_JoyAI.py) — importable `JoyEdit` / `JoyEditBatch` wrappers around the original JoyAI-Image repo-code deployment (`infer_runtime` / `modules`; also a small `__main__` sanity CLI).
- [`cleaner.py`](cleaner.py) — sort finished run folders into a flat, metadata-annotated dataset: `python cleaner.py --input runsreal/<collection> --output cleaned/ --code C`.
- [`gather_reconstruction.py`](gather_reconstruction.py) — copy the final reconstruction from many run folders into one directory.
- [`collage_svg.py`](collage_svg.py) — rebuild SVG collages from layerwise `metadata.json` files.

---

## Notes & caveats

- Only JoyAI loads locally and stays resident; run the whole batch in a single invocation rather than one process per image (the VLM is a remote API call).
- On JoyAI retries the seed is bumped automatically — the same prompt + seed reproduces the same output.
- The VLM grading its own pipeline tends to be lenient; the verifier prompts are tuned to report *defect categories*, not a yes/no, and always compare against the original as reference.
- **The verification loops are expensive**: in testing, each JoyAI generation takes ~2 min and the verifier flags nearly every first attempt (`bleed_in`/`halo`), so `--use_verify` can push a single image to ~75 min (8 elements × 3 retries). Some flagged `halo` is introduced by the rembg matting step itself, which no generation retry can fix. Run without `--use_verify`/`--use_global` for fast passes; enable them only when you need the quality gate.
- See [`AGENT_BUILD_INSTRUCTIONS.md`](AGENT_BUILD_INSTRUCTIONS.md) §10 for the known failure modes (missed elements, identity drift, local-vs-global threshold conflicts, heavy occlusion).
