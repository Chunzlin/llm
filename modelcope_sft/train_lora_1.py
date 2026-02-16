"""
Qwen3-1.7B LoRA微调完整脚本
环境要求: torch, transformers, peft, datasets, swanlab, pandas
作者: AI助手 | 针对Ubuntu 22.04 + RTX 4090 24GB优化
"""

import json
import pandas as pd
import torch
import os
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import (
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,  # 用于QLoRA量化（可选）
    pipeline
)
from peft import (
    LoraConfig, 
    TaskType, 
    get_peft_model, 
    PeftModel,
    prepare_model_for_kbit_training  # 用于量化训练准备
)
import swanlab
from typing import List, Dict, Optional

# ==================== 配置区域 ====================
class TrainingConfig:
    """训练配置类，集中管理所有超参数"""
    
    # 模型配置
    MODEL_NAME = "Qwen/Qwen3-1.7B"
    MODEL_CACHE_DIR = "./autodl-tmp/"  # 模型缓存目录
    
    # LoRA配置
    LORA_R = 8  # LoRA秩，控制参数数量。建议值：4, 8, 16, 32（越大能力越强但参数越多）
    LORA_ALPHA = 32  # 缩放参数，通常设为LORA_R的2-4倍
    LORA_DROPOUT = 0.1  # Dropout率，防止过拟合
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # 注入LoRA的模块
    
    # 训练参数
    BATCH_SIZE = 2  # 每设备批次大小，4090上可尝试4-8
    GRADIENT_ACCUMULATION_STEPS = 8  # 梯度累积步数，有效批次=BATCH_SIZE*此值=16较好 推荐
    NUM_EPOCHS = 3  # 训练轮数
    LEARNING_RATE = 2e-4  # 学习率，LoRA常用1e-4到5e-4
    MAX_LENGTH = 2048  # 最大序列长度
    
    # 提示词模板
    SYSTEM_PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
    
    # 路径配置
    OUTPUT_DIR = "./autodl-tmp/output/Qwen3-1.7B-LoRA"
    TRAIN_DATA_PATH = "train.jsonl"
    VAL_DATA_PATH = "val.jsonl"
    
    # 是否使用4-bit量化（QLoRA）进一步节省显存
    USE_4BIT_QUANTIZATION = False  # 设为True可使用QLoRA（需要bitsandbytes库）
    
    # SwanLab配置
    SWANLAB_PROJECT = "qwen3-sft-medical-lora"
    SWANLAB_RUN_NAME = "qwen3-1.7B-lora-experiment"

# 应用SwanLab配置
os.environ["SWANLAB_PROJECT"] = TrainingConfig.SWANLAB_PROJECT
swanlab.config.update({
    "model": TrainingConfig.MODEL_NAME,
    "lora_r": TrainingConfig.LORA_R,
    "lora_alpha": TrainingConfig.LORA_ALPHA,
    "lora_target_modules": str(TrainingConfig.LORA_TARGET_MODULES),
    "batch_size": TrainingConfig.BATCH_SIZE,
    "learning_rate": TrainingConfig.LEARNING_RATE,
    "max_length": TrainingConfig.MAX_LENGTH,
    "use_4bit": TrainingConfig.USE_4BIT_QUANTIZATION,
})

# ==================== 数据处理函数 ====================
def format_dataset_jsonl(input_path: str, output_path: str) -> None:
    """
    将原始JSONL数据集格式化为标准指令微调格式
    
    参数:
        input_path: 原始数据路径
        output_path: 格式化后数据路径
    """
    formatted_data = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                # 构建标准指令格式
                formatted_example = {
                    "instruction": TrainingConfig.SYSTEM_PROMPT,
                    "input": data.get("question", ""),
                    "output": f"<think>{data.get('think', '')}</think>\n{data.get('answer', '')}"
                }
                formatted_data.append(formatted_example)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误（行被跳过）: {e}")
                continue
    
    # 保存格式化后的数据
    with open(output_path, "w", encoding="utf-8") as f:
        for example in formatted_data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"数据集已格式化并保存到: {output_path}，共 {len(formatted_data)} 条样本")

def preprocess_function(example: Dict, tokenizer) -> Dict:
    """
    预处理函数：将文本数据转换为模型训练所需的token IDs
    
    参数:
        example: 单条数据样本
        tokenizer: 分词器对象
    
    返回:
        包含input_ids, attention_mask, labels的字典
    """
    # 构建对话模板（遵循Qwen官方格式）
    conversation = [
        {"role": "system", "content": TrainingConfig.SYSTEM_PROMPT},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]}
    ]
    
    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False  # 训练时不添加生成提示
    )
    
    # 对文本进行分词
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=TrainingConfig.MAX_LENGTH,
        padding=False  # 在DataCollator中统一处理padding
    )
    
    # 为训练准备标签（将输入部分标记为-100，模型只会学习预测输出部分）
    # 这里简化处理：整个序列都用于计算损失
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# ==================== 模型加载与LoRA配置 ====================
def load_model_and_tokenizer():
    """
    加载模型和分词器，并配置LoRA
    
    返回:
        model: 配置了LoRA的模型
        tokenizer: 分词器
    """
    print("=" * 50)
    print("开始加载模型和分词器...")
    
    # 1. 下载或加载模型（如果已缓存则直接使用）
    model_dir = snapshot_download(
        TrainingConfig.MODEL_NAME, 
        cache_dir=TrainingConfig.MODEL_CACHE_DIR,
        revision="master"
    )
    print(f"模型位置: {model_dir}")
    
    # 2. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=False,
        trust_remote_code=True
    )
    
    # 设置特殊token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # 填充侧设置为右侧
    
    # 3. 配置量化（如果启用QLoRA）
    bnb_config = None
    torch_dtype = torch.bfloat16  # 4090支持bfloat16，效果好
    
    if TrainingConfig.USE_4BIT_QUANTIZATION:
        print("使用4-bit量化（QLoRA）...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        torch_dtype = torch.float16
    
    # 4. 加载基础模型
    print("加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",  # 自动分配到可用设备
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        use_cache=False,  # 训练时关闭缓存以节省显存
    )
    
    # 5. 为k-bit训练准备模型（如果使用量化）
    if TrainingConfig.USE_4BIT_QUANTIZATION:
        model = prepare_model_for_kbit_training(model)
    
    # 6. 配置LoRA
    print("配置LoRA适配器...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=TrainingConfig.LORA_R,
        lora_alpha=TrainingConfig.LORA_ALPHA,
        lora_dropout=TrainingConfig.LORA_DROPOUT,
        target_modules=TrainingConfig.LORA_TARGET_MODULES,
        bias="none",
        modules_to_save=None,  # 可指定需要全参数训练的模块
    )
    
    # 7. 应用LoRA到模型
    model = get_peft_model(model, lora_config)
    
    # 8. 启用梯度检查点（进一步节省显存）
    model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # 9. 打印可训练参数信息
    model.print_trainable_parameters()
    
    print("模型和分词器加载完成！")
    print("=" * 50)
    
    return model, tokenizer

# ==================== 训练函数 ====================
def train_model(model, tokenizer, train_dataset, eval_dataset):
    """
    训练LoRA适配器
    
    参数:
        model: LoRA模型
        tokenizer: 分词器
        train_dataset: 训练数据集
        eval_dataset: 验证数据集
    
    返回:
        trainer: 训练器对象
    """
    print("开始配置训练参数...")
    
    # 训练参数配置
    # 训练参数配置
    training_args = TrainingArguments(
        # 输出目录
        output_dir=TrainingConfig.OUTPUT_DIR,
        
        # 训练参数
        per_device_train_batch_size=TrainingConfig.BATCH_SIZE,
        per_device_eval_batch_size=TrainingConfig.BATCH_SIZE,
        gradient_accumulation_steps=TrainingConfig.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=TrainingConfig.NUM_EPOCHS,
        learning_rate=TrainingConfig.LEARNING_RATE,
        
        # 优化器设置
        optim="adamw_8bit" if TrainingConfig.USE_4BIT_QUANTIZATION else "adamw_torch",
        weight_decay=0.01,
        warmup_ratio=0.03,  # 预热比例
        
        # 评估与保存策略 - 关键修改处！
        eval_strategy="steps",          # 原来是 evaluation_strategy
        eval_steps=100,
        save_strategy="steps",          # 保持不变
        save_steps=400,
        save_total_limit=3,             # 最多保存3个检查点
        load_best_model_at_end=True,    # 训练结束后加载最佳模型
        metric_for_best_model="eval_loss",  # 根据验证损失选择最佳模型
        
        # 日志与报告
        logging_strategy="steps",       # 保持不变
        logging_steps=10,
        report_to="swanlab",
        run_name=TrainingConfig.SWANLAB_RUN_NAME,
        
        # 精度与硬件优化
        bf16=torch.cuda.is_bf16_supported(),  # 4090支持bf16
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,    # 梯度检查点节省显存
        
        # 其他
        dataloader_num_workers=4,
        remove_unused_columns=True,
        group_by_length=True,           # 按长度分组提高效率
        dataloader_pin_memory=True,
        
        # 新版本可能需要添加的额外参数
        logging_first_step=True,        # 记录第一步的日志
        greater_is_better=False,        # eval_loss越小越好
    )
    
    # 数据收集器（动态填充）
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,  # 对齐到8的倍数，GPU效率更高
        return_tensors="pt"
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型（LoRA适配器）
    print("保存LoRA适配器...")
    model.save_pretrained(os.path.join(TrainingConfig.OUTPUT_DIR, "final_lora_adapter"))
    tokenizer.save_pretrained(os.path.join(TrainingConfig.OUTPUT_DIR, "final_lora_adapter"))
    
    # 保存训练历史
    trainer.save_model(TrainingConfig.OUTPUT_DIR)
    trainer.save_state()
    
    print(f"训练完成！模型保存在: {TrainingConfig.OUTPUT_DIR}")
    
    return trainer

# ==================== 推理函数 ====================
class QwenLoraInference:
    """LoRA微调后的推理类"""
    
    def __init__(self, base_model_name: str = None, lora_adapter_path: str = None):
        """
        初始化推理模型
        
        参数:
            base_model_name: 基础模型名称或路径
            lora_adapter_path: LoRA适配器路径
        """
        if base_model_name is None:
            base_model_name = TrainingConfig.MODEL_CACHE_DIR
            base_model_name = './autodl-tmp/Qwen/Qwen3-1.7B'
            
        if lora_adapter_path is None:
            lora_adapter_path = os.path.join(TrainingConfig.OUTPUT_DIR, "final_lora_adapter")
        
        print(f"加载基础模型: {base_model_name}")
        print(f"加载LoRA适配器: {lora_adapter_path}")
        
        # 加载基础模型
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            use_fast=False,
            trust_remote_code=True
        )
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        # 加载LoRA适配器
        self.model = PeftModel.from_pretrained(
            self.base_model,
            lora_adapter_path,
            device_map="auto"
        )
        
        # 设置为评估模式
        self.model.eval()
        self.base_model.eval()
        print("推理模型加载完成！")
    
    def generate_response(self, 
                         question: str, 
                         model: str = 'ft',  # 使用原模型还是微调后的模型
                         max_new_tokens: int = 1324,
                         temperature: float = 0.7,
                         do_sample: bool = True) -> str:
        """
        生成回答
        
        参数:
            question: 用户问题
            max_new_tokens: 最大生成token数
            temperature: 温度参数（控制随机性）
            do_sample: 是否使用采样
        
        返回:
            模型生成的回答
        """
        model = self.model if model=='ft' else self.base_model
        # 构建对话
        messages = [
            {"role": "system", "content": TrainingConfig.SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        
        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        
        # 生成参数
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_config
            )
        
        # 解码并提取助手回复
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取助手部分（从"assistant"标签后开始）
        assistant_marker = "<|im_start|>assistant\n"
        if assistant_marker in full_response:
            response = full_response.split(assistant_marker)[-1]
        else:
            response = full_response
        
        return response.strip()
    
    def batch_predict(self, 
                     questions: List[str], 
                     max_new_tokens: int = 512) -> List[str]:
        """
        批量预测
        
        参数:
            questions: 问题列表
            max_new_tokens: 最大生成token数
        
        返回:
            回答列表
        """
        responses = []
        for i, question in enumerate(questions):
            print(f"处理问题 {i+1}/{len(questions)}: {question[:50]}...")
            response = self.generate_response(question, max_new_tokens)
            responses.append(response)
        return responses

# ==================== 主执行流程 ====================
def main():
    """主函数：执行完整训练和推理流程"""
    
    print("=" * 60)
    print("Qwen3-1.7B LoRA微调脚本")
    print(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print("=" * 60)
    
    # 步骤1: 检查并格式化数据集
    print("\n[步骤1/5] 准备数据集...")
    train_formatted_path = "train_formatted.jsonl"
    val_formatted_path = "val_formatted.jsonl"
    
    if not os.path.exists(train_formatted_path):
        format_dataset_jsonl(TrainingConfig.TRAIN_DATA_PATH, train_formatted_path)
    if not os.path.exists(val_formatted_path):
        format_dataset_jsonl(TrainingConfig.VAL_DATA_PATH, val_formatted_path)
    
    # 步骤2: 加载模型和分词器
    print("\n[步骤2/5] 加载模型和分词器...")
    model, tokenizer = load_model_and_tokenizer()
    
    # 步骤3: 加载并预处理数据集
    print("\n[步骤3/5] 加载并预处理数据集...")
    
    # 加载训练集
    train_df = pd.read_json(train_formatted_path, lines=True)
    train_ds = Dataset.from_pandas(train_df)
    
    # 加载验证集
    val_df = pd.read_json(val_formatted_path, lines=True)
    val_ds = Dataset.from_pandas(val_df)
    
    print(f"训练集大小: {len(train_ds)}")
    print(f"验证集大小: {len(val_ds)}")
    
    # 预处理数据集（使用lambda函数传递tokenizer）
    train_dataset = train_ds.map(
        lambda x: preprocess_function(x, tokenizer),
        remove_columns=train_ds.column_names,
        batched=False
    )
    
    eval_dataset = val_ds.map(
        lambda x: preprocess_function(x, tokenizer),
        remove_columns=val_ds.column_names,
        batched=False
    )
    
    # 步骤4: 训练模型
    print("\n[步骤4/5] 训练LoRA适配器...")
    trainer = train_model(model, tokenizer, train_dataset, eval_dataset)
    
    # 步骤5: 测试推理
    print("\n[步骤5/5] 测试推理...")
    
    # 创建推理实例
    inference_model = QwenLoraInference()
    
    # 测试问题
    test_questions = [
        "什么是糖尿病？",
        "感冒和流感有什么区别？",
        "如何预防高血压？"
    ]
    
    # 生成回答
    print("\n测试推理结果:")
    print("-" * 50)
    
    test_results = []
    for i, question in enumerate(test_questions):
        print(f"\n问题 {i+1}: {question}")
        response = inference_model.generate_response(question, max_new_tokens=300)
        print(f"回答: {response}")
        
        # 记录到SwanLab
        test_results.append({
            "question": question,
            "response": response
        })
    
    # 将测试结果保存到文件
    with open(os.path.join(TrainingConfig.OUTPUT_DIR, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n测试结果已保存到: {os.path.join(TrainingConfig.OUTPUT_DIR, 'test_results.json')}")
    
    # 保存SwanLab日志
    swanlab_text_logs = [swanlab.Text(f"Q: {r['question']}\nA: {r['response']}") for r in test_results]
    swanlab.log({"Test_Predictions": swanlab_text_logs})
    
    print("\n" + "=" * 60)
    print("LoRA微调流程全部完成！")
    print(f"1. 模型适配器保存在: {TrainingConfig.OUTPUT_DIR}/final_lora_adapter")
    print(f"2. 训练日志查看: SwanLab项目 '{TrainingConfig.SWANLAB_PROJECT}'")
    print(f"3. 使用 QwenLoraInference 类加载模型进行推理")
    print("=" * 60)

if __name__ == "__main__":
    # 执行主流程
    main()
    
    # 可选：训练后直接进行推理示例
    inference = QwenLoraInference()
    result = inference.generate_response("凝血块机化后会对呼吸功能造成损害吗？怎么影响的呢？？")
    print(result)

    print('\n---\n base ------------\n----------')
    result = inference.generate_response("凝血块机化后会对呼吸功能造成损害吗？怎么影响的呢？？", model='base')
    print(result)