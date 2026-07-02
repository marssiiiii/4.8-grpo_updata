MODEL_ID   = "baidu/ERNIE-4.5-0.3B-PT"
MODEL_PATH_1 = "five_plus_two_optimization/train_model_new/train_base_model/TRAIN_RESULTS/ernie_base_model_10"
DATA_PATH = "train_json_data/five_plus_two_train_jsonl_data/design_3.27/base_model_train/train_set_100_actionized_sort_floor_force_auged(3_times).jsonl"
import json
from peft import LoraConfig, get_peft_model
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForCausalLM, TrainerCallback
import gc
import numpy as np
import os
import sys
sys.path.append("/home/jiuxing_li")
from five_plus_two_optimization.train_model_new.train_base_model.ernie_base_model_train import initialize_token_embedding

def initialize_untrained_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        use_cache=False,
    )
    initialize_token_embedding(model,tokenizer)
    return tokenizer,model

def initialize_trained_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH_1,device_map="auto",torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_1)
    model.eval()
    return tokenizer,model

if __name__ == "__main__":
    ut_tokenizer,ut_model=initialize_untrained_model()
    tokenizer,model=initialize_trained_model()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            prompt = record["prompt"]
            response=record["response"]

            generate_prefix = tokenizer(prompt,return_tensors="pt").to("cuda").input_ids  # prefix
            response_pred=""
            
            #生成回答
            gen = model.generate(generate_prefix, max_new_tokens=200, do_sample=True,temperature=0.7)
            prompt_token_length=len(tokenizer(prompt).input_ids)
            generated_ids = gen[0,prompt_token_length:]
            generated_tokens = tokenizer.decode(generated_ids, skip_special_tokens=False)
            response_pred+=generated_tokens
