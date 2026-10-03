import torch
from PIL import Image

from diffusers import JoyImageEditPipeline

pipeline = JoyImageEditPipeline.from_pretrained("/remote-home/Zhangkaile/models/JoyAI-Image-Edit-Diffusers/")
pipeline.to(torch.bfloat16)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)
print("pipeline loaded")

img_path = "/remote-home/Zhangkaile/dev/AutoTask/AutoAgentRaw/images/009.png"
prompt = "Remove the girl."
image = Image.open(img_path).convert("RGB")

inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "num_inference_steps": 40,
    "guidance_scale": 4.0,
}

print("run pipeline...")

with torch.inference_mode():
    output = pipeline(**inputs)
    image = output.images[0]
    image.save("joyai_image_edit_output.png")
    print("image saved.")
