import argparse
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Global variables to cache model and processor
_model = None
_processor = None

# Pin the VLM to its own card. JoyAI lives on a DIFFERENT card (see Models & Resources).
# QWEN_DEVICE = "cuda:1"


def get_model_and_processor(use_flash_attn=False, device="cuda:1"):
    """Initializes and caches the Qwen3-VL model and processor."""
    global _model, _processor

    if _model is None or _processor is None:
        model_id = "/remote-home/Zhangkaile/models/Qwen3-VL-32B-Instruct/"
        print(f"Loading {model_id} on {device}...")

        # Always use bfloat16 to fit the 32B model in GPU memory (~64GB).
        # float32 would require ~128GB and cause CPU offloading / extreme slowdown.
        kwargs = {
            "device_map": device,
            "torch_dtype": torch.bfloat16,
        }
        if use_flash_attn:
            kwargs["attn_implementation"] = "flash_attention_2"
            print("Flash Attention 2 enabled.")

        _model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        _processor = AutoProcessor.from_pretrained(model_id)

        print("--- Qwen3VL Device Map ---")
        print(_model.hf_device_map)

    return _model, _processor


def clear_qwen_cache():
    """Clear the cached Qwen3VL model and free GPU memory."""
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


def Qwen3VL_inference(
    image_input,
    prompt,
    system_prompt=None,
    use_flash_attn=False,
    max_new_tokens=512,
    deterministic=True,
    device="cuda:2",
):
    """
    Runs inference on Qwen3-VL with one or MORE images and a text prompt.

    Args:
        image_input: a single image (URL / local path / PIL.Image) OR a list of
            images. Multiple images are required for the verifier roles, which
            see the ORIGINAL as a reference alongside the RESULT. When a list is
            passed, images appear in the prompt in the given order; refer to them
            in the text as "the first image", "the second image", etc.
        prompt (str): the user text / question.
        system_prompt (str|None): optional system role text. Our seven agent
            roles are written as system prompts; pass them here.
        use_flash_attn (bool): whether to use Flash Attention 2.
        max_new_tokens (int): max tokens to generate. Use ~1500 for the planner
            (long JSON), ~256 for the checkers. Default 512.
        deterministic (bool): if True, greedy decoding (do_sample=False) for
            reproducible, parseable structured output. Set False only if you
            deliberately want sampling.

    Returns:
        str: the generated text response.
    """
    model, processor = get_model_and_processor(use_flash_attn=use_flash_attn, device=device)

    images = _normalize_images(image_input)

    # Build the user content: one image block per image, then the text.
    user_content = [{"type": "image", "image": img} for img in images]
    user_content.append({"type": "text", "text": prompt})

    messages = []
    if system_prompt:
        # The processor expects system message content as a list of text blocks too
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": user_content})

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference
    print("Generating response...")
    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if deterministic:
        gen_kwargs["do_sample"] = False  # greedy -> reproducible JSON

    generated_ids = model.generate(**inputs, **gen_kwargs)

    # Trim the input tokens out of the generated response tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Qwen3-VL-32B-Instruct via CLI.")

    # Required arguments
    parser.add_argument("--image", type=str, required=True, nargs="+",
                        help="One or more image paths/URLs (space separated).")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt for the image(s).")

    # Optional arguments
    parser.add_argument("--system", type=str, default=None, help="Optional system prompt.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum new tokens to generate.")
    parser.add_argument("--flash_attn", action="store_true", help="Enable Flash Attention 2.")
    parser.add_argument("--sample", action="store_true", help="Enable sampling (default is greedy).")

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
    )

    print("\n--- Model Response ---")
    print(response)
