from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"  # 或 Qwen/Qwen2-7B-Chat
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 设置填充令牌

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # 自动分配GPU
    trust_remote_code=True
)