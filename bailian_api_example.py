import os
import base64
import shutil
import time
import argparse
from openai import OpenAI

# --- DEFAULTS ---
DEFAULT_API_KEY = "sk-e4e886d9a87948b0a23b17bb33a025b2"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL_NAME = "qwen3-vl-flash"
DEFAULT_TASK = "layout"
MAX_RETRIES = 3
RETRY_DELAY = 5  # Seconds to wait before retrying

DEFAULT_PROMPT = """You are an expert visual analyst specializing in illustration composition.
 
Analyze the provided illustration and decompose it into logical visual layers and their constituent elements.
 
**Output rules (STRICT):**
- Return ONLY a valid JSON object — no markdown fences, no explanatory text before or after.
- Maximum 3 layers (layer0 = furthest background, higher numbers = closer to viewer).
- Maximum 6 elements total across all layers.
- Each element is a JSON array of exactly 3 strings:
    [ "element_id", "short name", "detailed description (≤20 words)" ]
  where element_id follows the pattern "element_01", "element_02", … (zero-padded, globally unique).
- Layer keys must be "layer0", "layer1", … in order.
- Descriptions must be specific, visual, and concise — focus on color, shape, position, and texture.
 
**Required JSON structure:**
{
  "layout": {
    "layer0": [
      ["element_01", "short name", "description ≤20 words"]
    ],
    "layer1": [
      ["element_02", "short name", "description ≤20 words"],
      ["element_03", "short name", "description ≤20 words"]
    ],
    ...
  }
}
 
Analyze the illustration now and return the JSON object."""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch image captioning using Qwen-VL via OpenAI-compatible API."
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Path to the source directory containing images.",
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help=(
            "Path to the target directory for output files. "
            "Defaults to source directory (only .txt files are saved, images are not copied)."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Model name to use for captioning. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help=f"Task label used as suffix in output .txt filenames. Default: {DEFAULT_TASK}",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        metavar="PROMPT_FILE",
        help=(
            "Path to a .txt file containing the prompt. "
            "If not provided, the built-in default prompt is used."
        ),
    )
    parser.add_argument(
        "--token",
        default=DEFAULT_API_KEY,
        help="API key for authentication. Defaults to the key embedded in the script.",
    )
    parser.add_argument(
        "--max-num",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Stop after successfully captioning this many images. "
            "If not set, all images are processed."
        ),
    )
    return parser.parse_args()


def load_prompt(prompt_path):
    """Load prompt text from a file, or return the default prompt."""
    if prompt_path is None:
        return DEFAULT_PROMPT
    if not os.path.isfile(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def encode_image(image_path):
    """Encodes a local image to base64 for the API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_caption_safe(client, model, prompt, image_path):
    """Calls the API with retry logic and error handling."""
    base64_image = encode_image(image_path)

    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
            )
            return completion.choices[0].message.content.strip()

        except Exception as e:
            print(f"  [Attempt {attempt + 1}] Error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return None


def main():
    args = parse_args()

    source_dir = args.source_dir
    # When target_dir is not specified, save outputs alongside source images (no copy)
    target_dir = args.target_dir
    same_dir_mode = target_dir is None
    if same_dir_mode:
        target_dir = source_dir

    prompt = load_prompt(args.prompt)

    client = OpenAI(api_key=args.token, base_url=DEFAULT_BASE_URL)

    if not os.path.exists(source_dir):
        raise NotADirectoryError(f"Source directory does not exist: {source_dir}")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    images = [
        f
        for f in os.listdir(source_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    print(f"\n>>> Source : {source_dir}")
    print(f">>> Target : {target_dir}  ({'in-place, no image copy' if same_dir_mode else 'copy images + save .txt'})")
    print(f">>> Model  : {args.model}")
    print(f">>> Task   : {args.task}")
    print(f">>> Images : {len(images)} found")
    if args.max_num is not None:
        print(f">>> Limit  : stop after {args.max_num} successful caption(s)")
    print()

    completed = 0

    for img_name in images:
        # Honour --max-num limit
        if args.max_num is not None and completed >= args.max_num:
            print(f"\n[!] Reached --max-num limit of {args.max_num}. Stopping.")
            break

        base_name = os.path.splitext(img_name)[0]
        src_path = os.path.join(source_dir, img_name)
        tgt_img_path = os.path.join(target_dir, img_name)
        tgt_txt_path = os.path.join(target_dir, f"{base_name}-{args.task}.txt")

        # Resumability: skip already-captioned images
        if os.path.exists(tgt_txt_path):
            print(f"  [-] {img_name} already captioned. Skipping.")
            continue

        print(f"  [+] Captioning {img_name}...")
        caption = get_caption_safe(client, args.model, prompt, src_path)

        if caption:
            # Copy image only when target differs from source
            if not same_dir_mode:
                shutil.copy2(src_path, tgt_img_path)

            with open(tgt_txt_path, "w", encoding="utf-8") as f:
                f.write(caption)

            completed += 1
            suffix = f" ({completed}/{args.max_num})" if args.max_num is not None else ""
            print(f"      Success: Saved {base_name}-{args.task}.txt{suffix}")
        else:
            print(f"      FAILED: Could not process {img_name}")

    print(f"\nAll tasks completed! {completed} image(s) captioned.")


if __name__ == "__main__":
    main()