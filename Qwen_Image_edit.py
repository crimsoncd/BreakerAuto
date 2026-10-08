"""
Edit an arbitrary-size image with Qwen-Image-2.1 and return the result at the
original size.

QwenImage21Pipeline already resizes any input image internally to
multiple-of-32 dimensions with area ~= output_resolution^2 (aspect preserved,
default 1024), so no size parameters are needed. The only extra step is
resizing the model output back to the exact original size.
"""

import torch
from PIL import Image
from diffusers import QwenImage21Pipeline

SRC = "/remote-home/Zhangkaile/dev/Normal2/QwenImage2.1/edit_target.png"
DST = "/remote-home/Zhangkaile/dev/Normal2/QwenImage2.1/edit_target_edited.png"

PROMPT = (
    "Isolate the slit_lamp by removing the blue_oval_backdrop, patient_chair, "
    "doctor, and patient. Complete any partially visible sections of the "
    "slit_lamp so it is fully whole. Place the isolated object on a plain, "
    "flat, solid-color background."
)


if __name__ == "__main__":
    # ---- 1. Load the input image (any size works) ----
    input_image = Image.open(SRC)
    if input_image.mode != "RGB":
        input_image = input_image.convert("RGB")
    orig_size = input_image.size  # (w, h)
    # print(f"original size : {orig_size[0]}x{orig_size[1]}")

    # ---- 2. Load pipeline and run edit ----
    pipe = QwenImage21Pipeline.from_pretrained(
        "/remote-home/Zhangkaile/models/Qwen-Image-2.1/", torch_dtype=torch.bfloat16
    ).to("cuda")

    image = pipe(
        prompt=PROMPT,
        image=input_image,
        num_inference_steps=40,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]

    # print(f"raw output    : {image.size[0]}x{image.size[1]}")

    # ---- 3. Restore exact original size ----
    if image.size != orig_size:
        image = image.resize(orig_size, Image.LANCZOS)
    image.save(DST)
    # print(f"saved         : {DST} ({image.size[0]}x{image.size[1]})")
