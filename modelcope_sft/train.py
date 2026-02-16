from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# 1. 加载模型与分词器 (7B标准版)
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,  # 关键：4090使用BF16
    trust_remote_code=True
)

# 2. 配置LoRA (参数精简有效)
lora_config = LoraConfig(
    r=8,  # 秩
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 应只占原模型约0.1%

# 3. 加载并预处理数据
dataset = load_dataset("json", data_files="data.json")
def preprocess(example):
    text = f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput: {example['output']}"
    return tokenizer(text, truncation=True, max_length=512, padding="max_length")
tokenized_dataset = dataset.map(preprocess, batched=True)

# 4. 配置训练参数 (针对4090优化)
training_args = TrainingArguments(
    output_dir="./qwen-lora-output",
    per_device_train_batch_size=4,  # 可调整，4090上7B模型可尝试8
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,  # 关键：开启BF16，大幅加速
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    remove_unused_columns=False
)

# 5. 开始训练
from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
)
trainer.train()