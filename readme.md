## 这个代码主要是学习如何对大模型进行微调，适合新手
! 使用modelscope下载模型和数据集，速度快，无需特殊网络
* llm/modelcope_sft/train_lora_1.py 进行lora微调
* llm/modelcope_sft/train_asft.py 进行全参数微调  
说明：lora速度较快，全参数微调很慢  
lora微调0.1%的参数就够了，生成的模型文件25M左右  
本实验使用5060Ti（16G）显卡和4090D(24G)显卡，系统为ubuntu22.04  
低于16G显存的显卡需要调整训练参数，比如调低batch-size
