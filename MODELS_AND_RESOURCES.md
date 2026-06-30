# Models & Resources

How the two models share the 4× A100 80G box, and how the wrappers must be called.

## Device map — dedicate, do not swap unless necessary

Each model is ~80G-class; two cannot share one card without OOM. But you have **4 cards**, so co-residence is the answer, not swapping. Swapping would reload a 32B model from disk inside the per-element loop — minutes per swap — and dominate runtime. Do not swap unless there is only one card.

Both models stay loaded for the whole run. The two spare cards are reserved for the obvious phase-2 win: a **second JoyAI worker** to parallelize element extraction (elements within one depth tier are independent, and the JoyAI generate is the bottleneck). The VLM does not need a second copy.

The wrappers now pin this by default: `QWEN_DEVICE = "cuda:0"` in the Qwen wrapper, `DEFAULT_DEVICE = "cuda:1"` in the JoyAI wrapper. Whenever before running a process, check the state of two GPU. You can use another available GPUs and you can modify your codes according to it.


## Calling conventions

VLM, single image (planner):
```python
from call_Qwen3VL import Qwen3VL_inference
out = Qwen3VL_inference(image, user_text, system_prompt=PLANNER_PROMPT, max_new_tokens=1500)
```

VLM, two images (any verifier — ORIGINAL first, RESULT second):
```python
out = Qwen3VL_inference([original_crop, result_cutout], VERIFY_TEXT,
                        system_prompt=ELEMENT_VERIFIER_PROMPT, max_new_tokens=256)
```

JoyAI edit (stays on cuda:1):
```python
from call_JoyAI import JoyEdit
res = JoyEdit(crop, isolation_prompt, out_path, device="cuda:1", seed=attempt_seed)
img = res.image
```

## Retry rule that needs no code change

On any JoyAI retry, **bump the seed** (e.g. `seed = 42 + attempt`). Same prompt + same seed reproduces the same bad image and wastes the attempt. The param is already there — just vary it.


## Phase-2 note — warm model servers (optional but high-value)

During tuning you'll restart the orchestrator constantly. In one process, every restart reloads the 32B VLM (minutes). Wrap each model in a thin long-lived local server (e.g. FastAPI around the exact functions above) so the orchestrator restarts in seconds while models stay hot. Secondary benefit: it isolates JoyAI's `sys.path` surgery and custom `src` imports from the transformers process, avoiding import/CUDA-context clashes. Not needed for v0; worth it once you're iterating on prompts and thresholds.

## Python Environment

Use Python under the path `/remote-home/Zhangkaile/miniconda3/envs/JoyZ/bin/python3.10`. Note: this conda environment contains every package you'll ever need, so do not pip install anything. You can inform me whenever unexpected devolping issues happen. Plus, if you would run a process consumes large time, you can use `nohup` and save the log.