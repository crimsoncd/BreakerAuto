# Models & Resources

How the pipeline uses its two models: a **remote VLM** (Bailian API) and a
**local JoyAI** image-edit model, and how the wrappers must be called.

## Device map — only JoyAI needs a GPU

The VLM runs remotely through the Bailian (DashScope) OpenAI-compatible API
(default model: `qwen3.7-flash`), so it occupies no local GPU and adds no
co-residence constraint. **A single GPU is enough** to run the whole
pipeline; the free cards can host a second JoyAI worker to parallelize
element extraction (elements within one depth tier are independent, and the
JoyAI generate is the bottleneck).

JoyAI's device is resolved by `pipeline_tools.resolve_joyai_device()`:
explicit `JOYAI_DEVICE` config/env value first, otherwise the card with the
most free memory is auto-detected at startup.

The API key is read from the git-ignored `.env` file (`BAILIAN_API_KEY`),
loaded into the environment by `config._load_env_file()` at import time.
Existing environment variables take precedence over the `.env` file.

## Calling conventions

VLM (qwen3.7-flash via Bailian API), single image (planner):
```python
from call_bailian_vlm import BailianVLM_inference
out = BailianVLM_inference(image, user_text, system_prompt=PLANNER_PROMPT, max_new_tokens=1500)
```

VLM, two images (any verifier — ORIGINAL first, RESULT second):
```python
out = BailianVLM_inference([original_crop, result_cutout], VERIFY_TEXT,
                           system_prompt=ELEMENT_VERIFIER_PROMPT, max_new_tokens=256)
```

VLM, text-only (prompt-writer roles):
```python
out = BailianVLM_inference(None, WRITER_TEXT, system_prompt=WRITER_PROMPT, max_new_tokens=512)
```

- Images (local paths, PIL objects, or lists of either) are sent as base64
  data URLs; `None` means a text-only call.
- `deterministic=True` (default) maps to `temperature=0` for reproducible
  structured output; `max_new_tokens` maps to the API's `max_tokens`.
- API/network failures are retried 3× with a 5s delay before raising.

JoyAI edit via the original JoyAI-Image repo-code deployment (single free
GPU, e.g. cuda:2). The model is built from `config.JOYAI_CKPT_ROOT` using
the `infer_runtime` / `modules` code of `config.JOYAI_SRC_DIR` (its `src`
folder is put on `sys.path` by `call_JoyAI.py`) — **not** through diffusers:
```python
from call_JoyAI import JoyEdit
res = JoyEdit(crop, isolation_prompt, out_path, device="cuda:2", seed=attempt_seed)
img = res.image
```

Note: the JoyAI-Image source code hard-imports `flash_attn`, so the
environment must have it installed (a matching wheel ships inside the
checkpoint dir / source repo).

## Retry rule that needs no code change

On any JoyAI retry, **bump the seed** (e.g. `seed = 42 + attempt`). Same prompt + same seed reproduces the same bad image and wastes the attempt. The param is already there — just vary it.

## Cost note (measured on the first API-based run)

- JoyAI generation: ~2 min per image (`JOYAI_STEPS=30`) — the bottleneck.
- VLM API calls: ~10–30s each — negligible.
- With `--use_verify` on and a strict verifier, most elements burn all 3
  retries → ~75 min per image (8 elements). Run without the verification
  flags for fast passes. A second JoyAI worker on a spare GPU is the
  obvious next speedup.

## Phase-2 note — warm model servers (no longer needed for the VLM)

The old motivation (avoid reloading the 32B VLM on every orchestrator
restart) is gone: the VLM is an API call and restarts are cheap. A thin
long-lived JoyAI server (FastAPI around `JoyEdit`) is still worth it if you
restart the orchestrator often while iterating on prompts, and it would
isolate JoyAI's `sys.path` surgery from the main process. Not needed now.

## Python Environment

Use Python under the conda env `JoyNew`
(`/remote-home/Zhangkaile/miniconda3/envs/JoyNew/bin/python`). Note: this
conda environment contains most packages you'll ever need (the `openai`
client, …), so do not pip install anything unless something is genuinely
missing — e.g. `flash_attn` for the JoyAI-Image repo code (see the note
above). You can inform me whenever unexpected developing issues happen.
Plus, if you would run a process that consumes a large amount of time, you
can use `nohup` and save the log (use `python -u` so the log is not
buffered).
