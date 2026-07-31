# Layer Decomposition Pipeline

An agentic pipeline that takes a single artistic illustration and decomposes it into a **background** layer plus N **element** layers (object-level cutouts), then reassembles them into a reconstruction of the original.

- Perception & judgment (planning, prompt-writing, verification) → **Qwen3-VL** (VLM)
- Generative isolation / amodal completion / background fill → **JoyAI** (image-edit model)
- Matting, resizing, compositing → classical CV (Pillow / rembg), no ML

For the full design spec (scene-graph data contract, stage breakdown, budgets, known pitfalls), see [`AGENT_BUILD_INSTRUCTIONS.md`](AGENT_BUILD_INSTRUCTIONS.md). The system prompts for every agent role live in [`prompts.py`](prompts.py) (originally [`SYSTEM_PROMPTS.md`](SYSTEM_PROMPTS.md)).

---

## Pipeline at a glance

```
Stage 1  Planning        VLM lists separable objects -> SceneGraph (name, bbox, depth_rank, overlaps)
Stage 2  Extraction      per element: occupancy check -> isolation prompt -> JoyAI generate
                         -> matte to alpha -> resize -> [optional VLM verify + retry, 3x]
Stage 3  Background      JoyAI removes named foregrounds and fills, [optional verify + retry, 3x]
Stage 4  Reassembly      composite layers over background -> [optional global verify + routing loop, 3x]
```

Every stage reads and writes one shared [`SceneGraph`](scene_graph.py) object; every intermediate image, prompt, and response is logged to a timestamped run folder (see [`logger.py`](logger.py)).

---

## Requirements

- **Hardware**: a machine with enough VRAM for both models at once. The reference box is 4× A100 80G — the VLM (~64GB bf16) and JoyAI (~80G-class) each get their own card. A single card will not hold both.
- **Models / code**, pointed at from `config.py`:
  - Qwen3-VL instruction model (local directory or a HuggingFace repo id, e.g. `Qwen/Qwen3-VL-32B-Instruct`).
  - The **JoyAI-Image** release: a **checkpoint directory** plus the **JoyAI-Image source code**. The source is a private, non-PyPI package (module name `joyai-image-release`) that provides `infer_runtime` / `modules`; you must either build & install it, or expose it through `JOYAI_SRC_DIR` (see [Setup](#setup)).
- **Python 3.10+** with the packages in [`requirements.txt`](requirements.txt) (or install this repo as a package via `pip install -e .`).

---

## Setup

### 1. Make the JoyAI-Image package available (prerequisite)

The JoyAI-Image source is not on PyPI, so install it first. Either way, its
heavy dependencies (`torch==2.8.0`, `diffusers==0.36.0`,
`transformers>=4.57.0,<4.58.0`, …) get resolved consistently with the pins in
[`requirements.txt`](requirements.txt) / `pyproject.toml`.

**Option A — build & install it (recommended):**

```bash
pip install /path/to/JoyAI-Image       # builds & installs joyai-image-release
```

> Note: JoyAI declares a couple of private packages (e.g. `kernels`). If the
> build fails on one, build/install it from the JoyAI-Image repo first, then
> retry.

**Option B — use a clone directly (no build):** skip this step and set
`JOYAI_SRC_DIR` in step 3 to your JoyAI-Image clone; the pipeline adds its
`src/` folder to `sys.path` itself.

### 2. Install this repo

```bash
git clone https://github.com/crimsoncd/BreakerAuto.git
cd BreakerAuto
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt       # or: pip install -e .   (builds the package, adds the `decompose` CLI)
```

### 3. Configure your local resources

All machine-specific settings live in [`config.py`](config.py). **Either** set
the environment variables **or** edit `config.py` directly — don't do both:

```bash
export QWEN_MODEL_ID=/path/to/Qwen3-VL-32B-Instruct
export JOYAI_CKPT_ROOT=/path/to/JoyAI-Image-Edit
export JOYAI_SRC_DIR=/path/to/JoyAI-Image          # repo root; its src/ is used
export QWEN_DEVICE=cuda:0                           # optional, else auto-detected
export JOYAI_DEVICE=cuda:1                          # optional, else auto-detected
```

Editing `config.py` instead looks like this:

```python
QWEN_MODEL_ID   = "/path/to/Qwen3-VL-32B-Instruct"
JOYAI_CKPT_ROOT = "/path/to/JoyAI-Image-Edit"
JOYAI_SRC_DIR   = "/path/to/JoyAI-Image"
```

   > The two models should live on **different** cards. If you don't set the device variables, the pipeline picks the two cards with the most free memory at startup.

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
| `QWEN_MODEL_ID` | `QWEN_MODEL_ID` | `Qwen/Qwen3-VL-32B-Instruct` | VLM checkpoint dir or HF repo id |
| `QWEN_DEVICE` | `QWEN_DEVICE` | auto-detect | GPU card for the VLM |
| `JOYAI_CKPT_ROOT` | `JOYAI_CKPT_ROOT` | `models/JoyAI-Image-Edit/` | JoyAI checkpoint root |
| `JOYAI_SRC_DIR` | `JOYAI_SRC_DIR` | `path/to/JoyAI-Image` | JoyAI-Image source repo root (`src/` is added to `sys.path`) |
| `JOYAI_DEVICE` | `JOYAI_DEVICE` | auto-detect | GPU card for JoyAI (must differ from the VLM's card) |
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
├── 0002_crop_girl.png                 # cropped input per element
├── 0003_vlm_occupancy_girl.json       # every VLM call logged (prompt + response)
├── ... joyai_*.png, gen_*, matte_*, layer_*.png
├── background_final.png               # Stage 3 result
├── reconstruction.png                 # Stage 4 composite
└── 00XX_final_shipped.json            # final scene graph
```

Every VLM prompt+response, every JoyAI input/output, and every intermediate matte is saved so any bad output is traceable to the stage and call that produced it.

---

## Utility scripts

- [`call_Qwen3VL.py`](call_Qwen3VL.py) — standalone VLM CLI: `python call_Qwen3VL.py --image in.png --prompt "..."`.
- [`call_JoyAI.py`](call_JoyAI.py) — importable `JoyEdit` / `JoyEditBatch` wrappers around the JoyAI-Image CLI (also a small `__main__` sanity CLI).
- [`cleaner.py`](cleaner.py) — sort finished run folders into a flat, metadata-annotated dataset: `python cleaner.py --input runsreal/<collection> --output cleaned/ --code C`.
- [`gather_reconstruction.py`](gather_reconstruction.py) — copy the final reconstruction from many run folders into one directory.
- [`collage_svg.py`](collage_svg.py) — rebuild SVG collages from layerwise `metadata.json` files.

---

## Notes & caveats

- Model loading is heavy; in one process both models stay resident (cached), so run the whole batch in a single invocation rather than one process per image.
- On JoyAI retries the seed is bumped automatically — the same prompt + seed reproduces the same output.
- The VLM grading its own pipeline tends to be lenient; the verifier prompts are tuned to report *defect categories*, not a yes/no, and always compare against the original as reference.
- See [`AGENT_BUILD_INSTRUCTIONS.md`](AGENT_BUILD_INSTRUCTIONS.md) §10 for the known failure modes (missed elements, identity drift, local-vs-global threshold conflicts, heavy occlusion).
