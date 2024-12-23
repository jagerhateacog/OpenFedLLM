"""
To support TRL supervised fine-tuning. Right now, we need to manually set the template here.
这段代码的功能是为监督微调（Supervised Fine-Tuning）提供格式化的模板，支持不同类型的模板（如Alpaca和Vicuna），用于准备训练数据。
"""

alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response: {}{}"""

vicuna_template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT: {}{}"""

"""
TEMPLATE_DICT 是一个字典，存储了不同模板的映射：
'alpaca'：对应alpaca_template，以及一个用于标识响应部分的字符串'\n### Response:'。
'vicuna'：对应vicuna_template，以及一个用于标识响应部分的字符串' ASSISTANT:'。
"""

TEMPLATE_DICT = {
    'alpaca': (alpaca_template, '\n### Response:'),
    'vicuna': (vicuna_template, ' ASSISTANT:'),
}

"""
该函数接收两个参数：
template_name：选择模板的名称（例如，'alpaca' 或 'vicuna'）。
eos_token：结束标记（通常是"<|endoftext|>"或类似的字符，用于标记生成文本的结束）。
函数的工作流程：
从TEMPLATE_DICT中获取对应模板和响应部分的标识符。
返回一个格式化函数formatting_prompts_func，该函数接收一个example（包含instruction和response的字典），并格式化输出。
"""

def get_formatting_prompts_func(template_name, eos_token):
    overall_temp, response_temp = TEMPLATE_DICT[template_name]
    def formatting_prompts_func(example):    
        output_texts = []    
        for i in range(len(example['instruction'])):    
            text = overall_temp.format(example['instruction'][i], example['response'][i], eos_token)    
            output_texts.append(text)    
        return output_texts    
    
    return formatting_prompts_func, response_temp
