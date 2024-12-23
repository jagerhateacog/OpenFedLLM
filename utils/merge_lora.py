"""
Usage:
python merge_lora.py --base_model_path [BASE-MODEL-PATH] --lora_path [LORA-PATH]
"""
import argparse
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def merge_lora(base_model_name, lora_path):

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    peft_model = PeftModel.from_pretrained(base_model, lora_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    model = peft_model.merge_and_unload()
    target_model_path = lora_path.replace("checkpoint", "full")
    model.save_pretrained(target_model_path)
    tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args.base_model_path, args.lora_path)
"""
这段代码的功能是将LoRA（Low-Rank Adaptation）模型与一个基础语言模型合并，并保存为一个新的完整模型。具体步骤如下：

加载基础模型和LoRA模型：
使用AutoModelForCausalLM.from_pretrained()方法加载基础语言模型（例如GPT、BERT等），并通过PeftModel.from_pretrained()加载LoRA模型。LoRA模型是基于基础模型的低秩适配（通常用于在大模型上进行更高效的微调）。

合并和卸载LoRA：
通过peft_model.merge_and_unload()方法，LoRA适配层会被合并到基础模型中，从而生成一个包含LoRA调整的完整模型。merge_and_unload()方法会把LoRA的参数与基础模型的参数合并，并且卸载LoRA特有的适配层。

保存合并后的模型：
合并后的完整模型被保存到指定路径（target_model_path），这个路径是通过替换lora_path中的“checkpoint”字符串为“full”来构造的。模型和tokenizer都会被保存到这个新路径，以便后续加载使用。

命令行接口（CLI）：
该脚本使用argparse模块来解析命令行参数。用户需要提供--lora_path参数来指定LoRA模型的路径，而--base_model_path参数是可选的（但需要根据实际情况进行指定）。
"""
