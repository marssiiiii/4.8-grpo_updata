import json
import torch
import sys
import os
from peft import PeftModel,LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
sys.path.append("/mnt/efs/jiuxing_li/five_plus_two_optimization/train_model_new/train_base_model")
from ernie_base_model_train import initialize_token_embedding
import time

# ------------------------
# 配置
# ------------------------
MODEL_PATH="five_plus_two_optimization/train_model_new/train_base_model/TRAIN_RESULTS/ernie_base_model_9" #var
mode_path = "baidu/ERNIE-4.5-0.3B-PT"
TRAIN_JSON = "five_plus_two_optimization/train_model_new/train_base_model/test/train_set_100_actionized_sort_floor_force_1_validation.jsonl"#var
TEST_JSON = "five_plus_two_optimization/train_model_new/train_base_model/test/test_set_100_actionized_sort_floor_force_1_validation.jsonl"
OUTPUT_PATH="five_plus_two_optimization/train_model_new/train_base_model/test/test_jsonl/base_model_9_train.jsonl" #var
MAX_LENGTH=10000
TRAIN_NUM,TEST_NUM=218,215
NUM_TOKENS=500
# ------------------------
# 加载模型与tokenizer
# ------------------------

def gen_sample(total_num,sample_points,data_json,num_tokens,output_path=None):
    for sample in range (total_num): #50个样本
        if sample not in sample_points:
            continue
        for repeat_times in range(1):
            print(f"当前生成{sample+1},重复第{repeat_times+1}次,{MODEL_PATH},{output_path}\n")
            #读取sample数据
            with open(data_json, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i == sample:   # 读取指定数据
                        record = json.loads(line)
                        house=record["house"]
                        floor=record["floor"]
                        design=record["design"]
                        bound=record["bound"]
                        context=record["context"]
                        completion_predict=record["completion_predict"]
                        response=record["response"]
                        prompt = record["prompt"]
                        response=record["response"]
                        pre_lineload_text=record["pre_lineload_text"]
                        pre_post_text=record["pre_post_text"]
                        break
            generate_prefix = tokenizer(prompt,return_tensors="pt").to("cuda").input_ids  # prefix
            response_pred=""
            
            #生成回答
            gen = model.generate(generate_prefix, max_new_tokens=num_tokens, do_sample=True,temperature=0.7)
            prompt_token_length=len(tokenizer(prompt).input_ids)
            generated_ids = gen[0,prompt_token_length:]
            generated_tokens = tokenizer.decode(generated_ids, skip_special_tokens=False)
            response_pred+=generated_tokens

            print(f"response:{response_pred}")
            if output_path!=None:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"house":house,"floor":floor,"design":design,"bound":bound,
                                        "context":context,"completion_predict":completion_predict,
                                        "prompt": prompt,"response": response,
                                        "response_pred":response_pred,
                                        "pre_lineload_text":pre_lineload_text,
                                        "pre_post_text":pre_post_text,
                                        }
                                        , ensure_ascii=False) + "\n")
            #break

if __name__ == "__main__":
    NEED_OUTPUT=True
    #直接加载模型
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,device_map="auto",torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()

    '''
    #无第一阶段训练的check-point加载
    #构建模型与tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mode_path) #自动加载Gemma3对应tokenizer
    if tokenizer.pad_token is None: #如果tokenizer中没有pad_token,就用eos_token填充
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        mode_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, #使用半精度节省显存，如gpu不支持则用32位浮点
        use_cache=False,  # 禁用KV缓存，与梯度检查点冲突
    )
    initialize_token_embedding(model,tokenizer)
    model = PeftModel.from_pretrained(model,MODEL_PATH)
    model.eval()
    emb = torch.load(f"{MODEL_PATH}/embedding.safetensors")
    target_dtype = model.get_input_embeddings().weight.dtype
    emb = emb.to(target_dtype)
    model.get_input_embeddings().weight.data.copy_(emb)
    print("LoRA + Embedding 恢复成功")
    print(model.get_input_embeddings().weight.dtype)
    print(emb.dtype)
    print(model.get_input_embeddings().weight.shape)
    print(emb.shape)
    '''
    
    # ------------------------
    #构建测试数据
    # ------------------------
    SAMPLE_NUM=200
    train_sample_points=np.linspace(0,TRAIN_NUM,SAMPLE_NUM,dtype=int)
    test_sample_points=np.linspace(0,TEST_NUM,SAMPLE_NUM,dtype=int)
    
    start_time = time.time()
    NEED_ASK_TRAIN,NEED_ASK_TEST=True,False

    if NEED_ASK_TRAIN:
        if NEED_OUTPUT:
            gen_sample(total_num=TRAIN_NUM,sample_points=train_sample_points,data_json=TRAIN_JSON,num_tokens=NUM_TOKENS,output_path=OUTPUT_PATH)
        else:
            gen_sample(total_num=TRAIN_NUM,sample_points=train_sample_points,data_json=TRAIN_JSON,num_tokens=NUM_TOKENS)
    if NEED_ASK_TEST:
        if NEED_OUTPUT:
            gen_sample(total_num=TEST_NUM,sample_points=test_sample_points,data_json=TEST_JSON,num_tokens=NUM_TOKENS,output_path=OUTPUT_PATH) 
        else:
            gen_sample(total_num=TEST_NUM,sample_points=test_sample_points,data_json=TEST_JSON,num_tokens=NUM_TOKENS)

    end_time = time.time()
    print(f"代码平均执行时间：{(end_time - start_time)/100:.4f} 秒")