import json
from datasets import Dataset
from transformers import AutoTokenizer
import statistics
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append("five_plus_two_optimization/train_model_new/train_base_model")
from ernie_base_model_train import initialize_token_embedding
#TRAIN_JSONL = "train_json_data/five_plus_two_train_jsonl_data/design_3.27/base_model_train/train_set_100_actionized_sort_floor_force.jsonl"
TRAIN_JSONL = "train_json_data/five_plus_two_train_jsonl_data/design_3.27/base_model_train/train_set_100_actionized_sort_floor_force_1_auged(3_times).jsonl"
#MODEL_ID = "baidu/ERNIE-4.5-0.3B-PT"
MODEL_ID = "google/gemma-3-1b-it"

def count_text_token(text):
    enc = tokenizer(text, add_special_tokens=False)
    token_count = len(enc["input_ids"])
    return token_count

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

    data_list = []
    with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                data_list.append(json.loads(line))

    dataset = Dataset.from_list(data_list) #dataset转化为Huggings face中的一种数据格式，dataset[0]={'prompt':,'completion':}
    print(f"Dataset size: {len(dataset)}") 

    #计算token_length
    token_lengths = []

    for example in data_list:
        prompt_response=example["prompt"]+example["response"]
        prompt_response_token_length = len(tokenizer(prompt_response).input_ids)
        token_lengths.append(prompt_response_token_length)

    avg_len = sum(token_lengths) / len(token_lengths)
    max_len = max(token_lengths)
    min_len = min(token_lengths)
    median_len = statistics.median(token_lengths)
    bound_len=6000
    ratio_less_bound = sum(l < bound_len for l in token_lengths) / len(token_lengths)

    print("==== Token Length Statistics ====")
    print(f"📌 Number of samples        : {len(token_lengths)}")
    print(f"📏 Avg token length         : {avg_len:.2f}")
    print(f"📏 Max token length         : {max_len}")
    print(f"📏 Min token length         : {min_len}")
    print(f"📏 Median token length         : {median_len}")
    print(f"📏 Ratio less than {bound_len}        : {ratio_less_bound}")
    print("=================================\n")
