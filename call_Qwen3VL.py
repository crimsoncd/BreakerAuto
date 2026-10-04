import argparse
import re

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

import config

# Global variables to cache model and processor
_model = None
_processor = None


def _default_device() -> str:
    """Resolve the default GPU card for the VLM from config."""
    return config.QWEN_DEVICE or "cuda:0"


def get_model_and_processor(device=None):
    """Initializes and caches the Qwen3.8 VLM and processor."""
    global _model, _processor

    if _model is None or _processor is None:
        device = device or _default_device()
        model_id = config.QWEN_MODEL_ID
        print(f"Loading {model_id} on {device}...")

        # Always use bfloat16 to fit the 27B model in GPU memory.
        _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        _model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map=device,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
        ).eval()

        print("--- Qwen3.8 Device ---")
        print(next(_model.parameters()).device)

    return _model, _processor


def clear_qwen_cache():
    """Clear the cached VLM and free GPU memory."""
    global _model, _processor
    _model = None
    _processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _normalize_images(image_input):
    """Accept a single image (str/path/PIL) or a list of them; return a list."""
    if image_input is None:
        return []
    if isinstance(image_input, (list, tuple)):
        return list(image_input)
    return [image_input]


def _load_pil_images(images):
    """Coerce every entry (path / PIL) into an RGB PIL image."""
    from PIL import Image

    loaded = []
    for img in images:
        if isinstance(img, Image.Image):
            loaded.append(img.convert("RGB"))
        else:
            loaded.append(Image.open(str(img)).convert("RGB"))
    return loaded


def _strip_think(text: str) -> str:
    """Remove reasoning/think blocks the model may emit before the answer."""
    return re.sub(
        r"(<think>.*?</think>|^.*?</think>|<think>.*$)",
        "",
        text,
        flags=re.DOTALL,
    ).strip()


def Qwen3VL_inference(
    image_input,
    prompt,
    system_prompt=None,
    use_flash_attn=False,
    max_new_tokens=65536,
    deterministic=True,
    device=None,
):
    """
    Runs inference on the Qwen3.8 VLM with one or MORE images and a text prompt.

    Args:
        image_input: a single image (URL / local path / PIL.Image) OR a list of
            images. Multiple images are required for the verifier roles, which
            see the ORIGINAL as a reference alongside the RESULT. When a list is
            passed, images appear in the prompt in the given order; refer to them
            in the text as "the first image", "the second image", etc.
        prompt (str): the user text / question.
        system_prompt (str|None): optional system role text. Our seven agent
            roles are written as system prompts; pass them here.
        use_flash_attn (bool): kept for API compatibility; attention backend is
            chosen by the model config.
        max_new_tokens (int): max tokens to generate. Use ~1500 for the planner
            (long JSON), ~256 for the checkers. Default 512.
        deterministic (bool): if True, greedy decoding (do_sample=False) for
            reproducible, parseable structured output. If False, samples with
            temperature=0.7 / top_p=0.8.

    Returns:
        str: the generated text response (think blocks stripped).
    """
    model, processor = get_model_and_processor(device=device)

    images = _load_pil_images(_normalize_images(image_input))
    # prompt = prompt + " /no_think"

    # Build the user content: one image block per image, then the text.
    user_content = [{"type": "image"} for _ in images]
    user_content.append({"type": "text", "text": prompt})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": user_content})

    # Apply the chat template, then build tensors with the processor.
    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text_input],
        images=images if images else None,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    # Inference
    print("Generating response...")
    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if deterministic:
        gen_kwargs["do_sample"] = False  # greedy -> reproducible JSON
    else:
        gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.8)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    # Trim the input tokens out of the generated response tokens
    input_len = inputs.input_ids.shape[1]
    generated_ids_trimmed = [out_ids[input_len:] for out_ids in generated_ids]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return _strip_think(output_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the Qwen3.8 VLM via CLI.")

    # Required arguments
    parser.add_argument("--image", type=str, required=True, nargs="+",
                        help="One or more image paths/URLs (space separated).")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt for the image(s).")

    # Optional arguments
    parser.add_argument("--system", type=str, default=None, help="Optional system prompt.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum new tokens to generate.")
    parser.add_argument("--flash_attn", action="store_true", help="Kept for compatibility (unused).")
    parser.add_argument("--sample", action="store_true", help="Enable sampling (default is greedy).")
    parser.add_argument("--device", type=str, default=None,
                        help="GPU card, e.g. cuda:0 (default: from config or cuda:0).")

    args = parser.parse_args()

    # If a single image was given, unwrap the list for convenience.
    image_arg = args.image if len(args.image) > 1 else args.image[0]

    response = Qwen3VL_inference(
        image_input=image_arg,
        prompt=args.prompt,
        system_prompt=args.system,
        use_flash_attn=args.flash_attn,
        max_new_tokens=args.max_tokens,
        deterministic=not args.sample,
        device=args.device,
    )

    print("\n--- Model Response ---")
    print(response)
