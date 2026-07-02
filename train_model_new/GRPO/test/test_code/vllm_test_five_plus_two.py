import json
import torch
import sys
import os
from peft import PeftModel,LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import time
from vllm import LLM, SamplingParams
import matplotlib.pyplot as plt

MODEL_PATH="/mnt/efs/jiuxing_li/five_plus_two_optimization/train_model/GRPO/base_model/qwen1.5b" #var
INPUT_JSON = "train_json_data/five_plus_two_train_jsonl_data/data_2.4/data_2.4_random_0.6_qwen_1.5b.jsonl"#var
MAX_LENGTH=5000


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    vllm_gen = LLM(model=MODEL_PATH, gpu_memory_utilization=0.5)
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            prompt = record["prompt"]
            ground_completion=record["completion"]
            house=record["house"]
            floor=record["floor"]
            tip_text = []
            tip_text.append(prompt)
            sampling_params = SamplingParams(n=10, temperature=0.5, max_tokens=200)
            voutputs = vllm_gen.generate(tip_text, sampling_params, use_tqdm=False)
            response_list, response_token_ids = [], []
            for v in voutputs: #将所有生成的answer都打平,返回answer+answer_token_id
                for z in v.outputs:
                    response_list.append(tokenizer.decode(z.token_ids))
                    response_token_ids.append(z.token_ids)
            print(response_list, response_token_ids)
            break

            '''
            tip_text = []
            text="<s>[INST]" + prompt + "[/INST]"
            prompt_ids = tokenizer(text, return_tensors="pt")["input_ids"]
            plen = prompt_ids.shape[1]
            tip_text.append(text)
            zz = vllm_gen.generate(tip_text, sampling_params=gen_logps_sp, use_tqdm=False)
            answer_logprobs = []
            for xx in zz:
                for out in xx.outputs:  # 0 表示第一个 prompt
                    print(tokenizer.decode(out.token_ids))
                    token_logprobs = out.logprobs  # dict: {token_id: TokenLogProb}
                    print(token_logprobs)
                    # 取 top-1 logprob
                    first_dict = token_logprobs[0]  # dict
                    first_logprob = list(first_dict.values())[0].logprob
                    answer_logprobs.append(first_logprob)

            gen_logps = torch.tensor([answer_logprobs])
            print(f"gen_logps: {gen_logps}")
            '''
            '''
            voutputs = vllm_gen.generate(tip_text, sampling_params, use_tqdm=False)
            answers, ans_token_ids = [], []
            for v in voutputs: #将所有生成的answer都打平,返回answer+answer_token_id
                for z in v.outputs:
                    answers.append(tokenizer.decode(z.token_ids))
                    ans_token_ids.append(z.token_ids)
            #print(answers, ans_token_ids)
            reward_correct(prompt,ground_completion,answers[0])
            visualize_segments_side_by_side(get_segments_info(prompt),get_segments_info(ground_completion)
            ,get_segments_info(answers[0]),"code/GRPO/vis1")
            '''
            #break