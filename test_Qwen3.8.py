from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "/remote-home/Zhangkaile/models/Qwen3.8-27B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype="auto"
)

text = "Hello, how does Qwen3.8 work?"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=65536)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

