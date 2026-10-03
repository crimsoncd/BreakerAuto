import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# 1. 设置模型路径
MODEL_ID = "/remote-home/Zhangkaile/models/Qwen3.8-27B"

def load_qwen_vl_model(model_path: str):
    """
    加载多模态模型和 Processor
    """
    print(f"Loading model from {model_path}...")
    
    # Processor 负责处理文本分词与图像预处理
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    
    # 使用 AutoModelForImageTextToText 加载视觉语言模型
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    ).eval()
    
    print("Model loaded successfully!")
    return model, processor


def generate_response(
    model, 
    processor, 
    prompt: str, 
    image_paths: list[str] = None, 
    max_new_tokens: int = 2048
) -> str:
    """
    调用模型生成回答
    """
    images = []
    if image_paths:
        for img_path in image_paths:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            images.append(Image.open(img_path).convert("RGB"))

    # 构造标准的多模态 Messages 结构
    content = []
    if images:
        for _ in images:
            content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "user", "content": content}
    ]

    # 使用 Processor 应用 Chat Template 并构建张量
    text_input = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = processor(
        text=text_input,
        images=images if images else None,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    # 生成回答
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8
        )

    # 截取生成的新 Token 并解码
    input_len = inputs.input_ids.shape[1]
    generated_ids_trimmed = [
        out_ids[input_len:] for out_ids in generated_ids
    ]
    
    response = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]

    return response


if __name__ == "__main__":
    model, processor = load_qwen_vl_model(MODEL_ID)

    # 1. 纯文本测试
    # print("\n--- Testing Text Prompt ---")
    # text_prompt = "Hello, how does Qwen work?"
    # answer_text = generate_response(
    #     model=model,
    #     processor=processor,
    #     prompt=text_prompt,
    #     max_new_tokens=512
    # )
    # print("Model Output:\n", answer_text)

    # 2. 带图测试（可取消注释使用）
    image_file = "/remote-home/Zhangkaile/dev/AutoTask/AutoAgentRaw/images/009.png"
    prompt_with_img = "请描述这张图片的内容。"
    answer_img = generate_response(
        model=model,
        processor=processor,
        prompt=prompt_with_img,
        image_paths=[image_file]
    )
    print("Image Output:\n", answer_img)

