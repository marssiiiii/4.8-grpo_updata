import json
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
import sys
sys.path.append("/mnt/efs/jiuxing_li/five_plus_two_optimization/train_model_new/train_base_model")
from ernie_base_model_train import initialize_token_embedding

input_path_1="train_json_data/five_plus_two_train_jsonl_data/data_2.4/data_2.4_random_0.6_qwen_1.5b.jsonl"
input_path_2="train_json_data/five_plus_two_train_jsonl_data/design_3.10/initial_data_train_set_actionized.jsonl"
model_path="five_plus_two_optimization/train_model_new/GRPO/base_model/qwen1.5b"

MODEL_ID = "baidu/ERNIE-4.5-0.3B-PT"

if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID) #初始化tokenizer
    if tokenizer.pad_token is None: #如果tokenizer中没有pad_token,就用eos_token填充
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto", #自动把模型分配到gpu
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, #使用半精度节省显存，如gpu不支持则用32位浮点
        use_cache=False,  # 禁用KV缓存，与梯度检查点冲突
    )
    initialize_token_embedding(model,tokenizer)
    
    text="<inoutbox>(1351,1983)<END>"
    print(len(tokenizer(text).input_ids))
    
    '''
    list1,list2=[],[]
    with open(input_path_1, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            prompt=record['prompt']
            prompt_tokens = len(tokenizer(prompt).input_ids)
            list1.append(prompt_tokens)

    with open(input_path_2, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            prompt=record['prompt']
            prompt_tokens = len(tokenizer(prompt).input_ids)
            list2.append(prompt_tokens)

    list2 = np.array(list2)
    print(f"之前prompt平均token数：{np.mean(list1)},最小：{np.min(list1)},最大：{np.max(list1)}")
    print(f"现在prompt平均token数：{np.mean(list2)},最小：{np.min(list2)},最大：{np.max(list2)},\
        3000超过了{np.sum(list2 < 3000) / len(list2)}的数据")
    '''