import json
import torch
import sys
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/train_model_new/base_code")
from five_plus_two_optimization.train_model_new.base_code.reward_design_trunc import \
    process_completion_predict_based_on_response,get_design_score,process_completion_predict_filt
from five_plus_two_optimization.train_model_new.train_base_model.cmp_actionized_and_tokenized_api import \
construct_prompt
import os
from peft import PeftModel,LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import time
import re
import random
#思路更新12.3
#在推理过程中加入物理规则

# ------------------------
# 配置
# ------------------------
TRAIN_JSON = "train_json_data/five_plus_two_train_jsonl_data/design_3.27/grpo_phase_1_train_data/train_set_inbox_force_prompt_1.jsonl"#var
TEST_JSON = "train_json_data/five_plus_two_train_jsonl_data/design_3.27/grpo_phase_1_train_data/test_set_inbox_force_prompt_1.jsonl" 

TRAIN_NUM,TEST_NUM=13613,4583
ASK_ROUND=3
# ------------------------
# 加载模型与tokenizer
# ------------------------

def IS_IMPROVE(context,completion_predict,answer):
    result_before=get_design_score(context,completion_predict)
    ratio_before=result_before["area_ratio"]
    #得到设计后的房屋整体设计分数
    result_after=get_design_score(context,answer)
    #if result_after["OPT_ROUND"]>0:
    #    print("含有非法设计！")
    #   return -1
    ratio_after=result_after["area_ratio"]

    print(f"ratio_before:{ratio_before},ratio_after:{ratio_after}")

    if ratio_after>ratio_before:
        print("提升！")
        return 2,ratio_after
    elif ratio_after==ratio_before:
        print("持平！")
        return 1,ratio_after
    else:
        print("下降！")
        return 0,ratio_after

def generate_answer(input_path,sample_points,generate_token_num,output_path=None,ask_round=None):
    generate_num=0
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in sample_points:   # 读取指定数据
                generate_num=generate_num+1
                if generate_num<=34:
                    continue
                print(i,MODEL_PATH,OUTPUT_PATH)
                record = json.loads(line)
                context=record["context"]
                prompt = record["prompt"]
                house=record["house"]
                floor=record["floor"]
                completion_predict=record["completion_predict"]
                bound=record["bound"]
                #for _ in range(ask_round): #生成response
                response_processed_before,response_processed=completion_predict,completion_predict
                response_list=[]
                while 1==1:
                    prompt=construct_prompt(context=context,completion_predict=response_processed_before) #构造prompt

                    generate_prefix = tokenizer(prompt,return_tensors="pt").to("cuda").input_ids #依据生成response
                    response=""
                    gen = model.generate(generate_prefix, max_new_tokens=generate_token_num, do_sample=False)
                    prompt_token_length=len(tokenizer(prompt).input_ids)
                    generated_ids = gen[0,prompt_token_length:]
                    generated_tokens = tokenizer.decode(generated_ids, skip_special_tokens=False)
                    response+=generated_tokens
                    print(f"response:{response}")
                    response_list.append(response)

                    response_processed=process_completion_predict_based_on_response( #依据response生成新的completion_predict
                        completion_predict=response_processed_before,response=response)
                    print(f"response_processed:{response_processed}")

                    result_before,result_after=get_design_score(context,response_processed_before),\
                    get_design_score(context,response_processed) #分别为两种设计打分
                    ratio_before,ratio_after=result_before["area_ratio"],result_after["area_ratio"]
                    print(f"ratio_before:{ratio_before},ratio_after:{ratio_after}")
                    if ratio_after<=ratio_before:
                        print("没有提升，退出")
                        break

                    response_processed_before=response_processed

                if output_path!=None:
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"house":house,"floor":floor,"bound":bound,"context":context,
                                            "prompt": prompt,"completion_predict":completion_predict,
                                            "response":response_list,
                                            "answer":response_processed,}
                                            , ensure_ascii=False) + "\n")

def generate_answer_auged(input_path,sample_points,generate_token_num_1=100,generate_token_num_2=200,temperature=1,MAX_GENERATE_ROUND=40,output_path=None):
    generate_num=0
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in sample_points:   # 读取指定数据
                generate_num=generate_num+1
                print(i,MODEL_PATH,OUTPUT_PATH)
                record = json.loads(line)
                context=record["context"]
                prompt = record["prompt"]
                house=record["house"]
                floor=record["floor"]
                completion_predict=record["completion_predict"]
                bound=record["bound"]
                generate_round=0
                round_completion_predict=completion_predict
                response_list,improved_response_list=[],[]
                while generate_round<MAX_GENERATE_ROUND:
                    print(f"ASK ROUND:{generate_round}")
                    prompt=construct_prompt(context=context,completion_predict=round_completion_predict) #构造prompt
                    prompt_token_length=len(tokenizer(prompt).input_ids)

                    generate_prefix = tokenizer(prompt,return_tensors="pt").to("cuda").input_ids #依据prompt生成response
                    response=""
                    if generate_round%2==0:
                        generate_token_num=generate_token_num_1
                    else:
                        generate_token_num=generate_token_num_2
                    gen = model.generate(generate_prefix, max_new_tokens=generate_token_num, do_sample=True,temperature=temperature)
                    generated_ids = gen[0,prompt_token_length:]
                    generated_tokens = tokenizer.decode(generated_ids, skip_special_tokens=False)
                    response+=generated_tokens
                    print(f"response:{response}")
                    response_list.append(response)

                    response_processed=process_completion_predict_based_on_response(completion_predict=round_completion_predict
                                                ,response=response)
                    print(f"response_processed:{response_processed}") #组合round_completion_predict和response
                
                    if IS_IMPROVE(context,round_completion_predict,response_processed)==2: #如果有提升则把round_completion_predict更新为response_processed，否则不更新
                        round_completion_predict=response_processed
                        improved_response_list.append(response)

                    generate_round=generate_round+1

                if output_path!=None:
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"house":house,"floor":floor,"bound":bound,"context":context,
                                            "prompt": prompt,"completion_predict":completion_predict,
                                            "response_list":response_list,
                                            "improved_response_list":improved_response_list,
                                            "answer":response_processed,}
                                            , ensure_ascii=False) + "\n")

def generate_answer_filt_auged(input_path,sample_points,generate_token_num_1=100,generate_token_num_2=200,temperature=1,MAX_GENERATE_ROUND=40,output_path=None):
    generate_num=0
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in sample_points:   # 读取指定数据
                generate_num=generate_num+1
                print(i,MODEL_PATH,OUTPUT_PATH)
                record = json.loads(line)
                context=record["context"]
                prompt = record["prompt"]
                house=record["house"]
                floor=record["floor"]
                completion_predict=record["completion_predict"]
                bound=record["bound"]
                pre_post_text=record["pre_post_text"]
                pre_lineload_text=record["pre_lineload_text"]

                base_result=get_design_score(context,completion_predict)
                completion_predict_filt=from_valid_segments_to_completion_predict(valid_beams=base_result["valid_beams"],valid_shearwalls=base_result["shear_walls_valid"])
                
                generate_round=0
                round_completion_predict=completion_predict_filt
                response_list,improved_response_list=[],[]
                round_completion_predict=process_completion_predict_filt(context=context,completion_predict=round_completion_predict,response="")
                while generate_round<MAX_GENERATE_ROUND:
                    print(i,MODEL_PATH,OUTPUT_PATH)
                    print(f"ASK ROUND:{generate_round}")
                    prompt=construct_prompt(context=context,completion_predict=round_completion_predict,
                                        pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text) #构造prompt
                    prompt_token_length=len(tokenizer(prompt).input_ids)

                    generate_prefix = tokenizer(prompt,return_tensors="pt").to("cuda").input_ids #依据prompt生成response
                    response=""
                    if generate_round%2==0:
                        generate_token_num=generate_token_num_1
                    else:
                        generate_token_num=generate_token_num_2
                    gen = model.generate(generate_prefix, max_new_tokens=generate_token_num, do_sample=True,temperature=temperature)
                    generated_ids = gen[0,prompt_token_length:]
                    generated_tokens = tokenizer.decode(generated_ids, skip_special_tokens=False)
                    response+=generated_tokens
                    print(f"response:{response}")
                    response_list.append(response)

                    response_processed=process_completion_predict_filt(context=context,
                            completion_predict=round_completion_predict,response=response)
                    print(f"response_processed:{response_processed}") #组合round_completion_predict和response
                    CUR_IS_IMPROVE,_=IS_IMPROVE(context,round_completion_predict,response_processed)
                
                    if CUR_IS_IMPROVE==2: #如果有提升则把round_completion_predict更新为response_processed，否则不更新
                        round_completion_predict=response_processed
                        improved_response_list.append(response)
                    generate_round=generate_round+1

                if output_path!=None:
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"house":house,"floor":floor,"bound":bound,"context":context,
                                            "prompt": prompt,"completion_predict":completion_predict,
                                            "response_list":response_list,
                                            "improved_response_list":improved_response_list,
                                            "answer":response_processed,
                                            "pre_post_text":pre_post_text,
                                            "pre_lineload_text":pre_lineload_text}
                                            , ensure_ascii=False) + "\n")

def generate_answer_super_auged(input_path,sample_points,index,generate_token_num_1=100,generate_token_num_2=200,temperature=2.0,MAX_GENERATE_ROUND=99999999,output_path=None):
    generate_num=0
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in sample_points:   # 读取指定数据
                generate_num=generate_num+1
                if generate_num<index or generate_num>index:
                    continue 
                print(i,MODEL_PATH,OUTPUT_PATH)
                record = json.loads(line)
                context=record["context"]
                prompt = record["prompt"]
                house=record["house"]
                floor=record["floor"]
                completion_predict=record["completion_predict"]
                bound=record["bound"]
                pre_post_text=record["pre_post_text"]
                pre_lineload_text=record["pre_lineload_text"]
                generate_round=0
                round_completion_predict=completion_predict
                response_list,improved_response_list=[],[]
                print(house,floor)
                while generate_round<MAX_GENERATE_ROUND:
                    print(f"ASK ROUND:{generate_round}")
                    prompt=construct_prompt(context=context,completion_predict=round_completion_predict,
                                            pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text) #构造prompt
                    prompt_token_length=len(tokenizer(prompt).input_ids)

                    generate_prefix = tokenizer(prompt,return_tensors="pt").to("cuda").input_ids #依据prompt生成response
                    response=""
                    if generate_round%2==0:
                        generate_token_num=generate_token_num_1
                    else:
                        generate_token_num=generate_token_num_2
                    gen = model.generate(generate_prefix, max_new_tokens=generate_token_num, do_sample=True,temperature=temperature)
                    generated_ids = gen[0,prompt_token_length:]
                    generated_tokens = tokenizer.decode(generated_ids, skip_special_tokens=False)
                    response+=generated_tokens
                    print(f"response:{response}")
                    response_list.append(response)

                    response_processed=process_completion_predict_based_on_response(completion_predict=round_completion_predict
                                                ,response=response)
                    print(f"response_processed:{response_processed}") #组合round_completion_predict和response
                
                    if IS_IMPROVE(context,round_completion_predict,response_processed)==2: #如果有提升则把round_completion_predict更新为response_processed，否则不更新
                        round_completion_predict=response_processed
                        improved_response_list.append(response)

                    generate_round=generate_round+1

                    if output_path!=None and generate_round%100==0:
                        with open(output_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"house":house,"floor":floor,"bound":bound,"context":context,
                                                "generate_round":generate_round,
                                                "prompt": prompt,"completion_predict":completion_predict,
                                                "response_list":response_list,
                                                "improved_response_list":improved_response_list,
                                                "answer":response_processed,}
                                                , ensure_ascii=False) + "\n")

def generate_answer_super_auged_filt(input_path,sample_points,index,generate_token_num_1=100,generate_token_num_2=200,temperature=0.7,MAX_GENERATE_ROUND=99999999,output_path=None):
    generate_num=0
    IS_IMPROVE_FLAG=False
    improve_step,improve_ratio=[],[]
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in sample_points:   # 读取指定数据
                generate_num=generate_num+1
                if generate_num<index or generate_num>index:
                    continue 
                print(i,MODEL_PATH,OUTPUT_PATH)
                record = json.loads(line)
                context=record["context"]
                prompt = record["prompt"]
                house=record["house"]
                floor=record["floor"]
                completion_predict=record["completion_predict"]
                bound=record["bound"]
                generate_round=0
                round_completion_predict=completion_predict
                response_list,improved_response_list=[],[]
                round_completion_predict=process_completion_predict_filt(context=context,completion_predict=round_completion_predict,response="")
                print(house,floor)
                while generate_round<MAX_GENERATE_ROUND:
                    print(index,house,floor)
                    print(f"ASK ROUND:{generate_round}")
                    prompt=construct_prompt(context=context,completion_predict=round_completion_predict) #构造prompt
                    prompt_token_length=len(tokenizer(prompt).input_ids)

                    generate_prefix = tokenizer(prompt,return_tensors="pt").to("cuda").input_ids #依据prompt生成response
                    response=""
                    if generate_round%2==0:
                        generate_token_num=generate_token_num_1
                    else:
                        generate_token_num=generate_token_num_2
                    gen = model.generate(generate_prefix, max_new_tokens=generate_token_num, do_sample=True,temperature=temperature)
                    print(f"温度为{temperature}")
                    generated_ids = gen[0,prompt_token_length:]
                    generated_tokens = tokenizer.decode(generated_ids, skip_special_tokens=False)
                    response+=generated_tokens
                    print(f"response:{response}")
                    response_list.append(response)

                    response_processed=process_completion_predict_filt(context=context,completion_predict=round_completion_predict
                                                ,response=response)
                    print(f"response_processed:{response_processed}") #组合round_completion_predict和response
                
                    IS_IMPROVED,_=IS_IMPROVE(context,round_completion_predict,response_processed)
                    if IS_IMPROVED==2: #如果有提升则把round_completion_predict更新为response_processed，否则不更新
                        round_completion_predict=response_processed
                        improved_response_list.append(response)
                        improve_step.append(generate_round)
                        improve_ratio.append(ratio_after)
                        IS_IMPROVE_FLAG=True
                    
                    print(f"IS_IMPROVE_FLAG:{IS_IMPROVE_FLAG}")

                    generate_round=generate_round+1

                    if output_path!=None and generate_round%40==0:
                        #if IS_IMPROVE_FLAG==False and temperature<2.5:
                        #   temperature=temperature+0.1
                        #   print(f"温度提升至{temperature}")
                        if IS_IMPROVE_FLAG==False:
                            temperature=2*random.random()
                            print(f"温度变为{temperature}")
                        IS_IMPROVE_FLAG=False
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(json.dumps({"house":house,"floor":floor,"bound":bound,"generate_round":generate_round,
                                                "improve_step":improve_step,
                                                "improve_ratio":improve_ratio,
                                                "context":context,
                                                "prompt": prompt,"completion_predict":completion_predict,
                                                #"response_list":response_list,
                                                "improved_response_list":improved_response_list,
                                                "answer":response_processed,}
                                                , ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # ------------------------
    #构建测试数据
    # ------------------------
    SAMPLE_NUM=50
    GENERATE_TOKEN_NUM=200
    train_sample_points = np.linspace(0, TRAIN_NUM-1, SAMPLE_NUM)
    train_sample_points = np.round(train_sample_points).astype(int)
    test_sample_points=np.linspace(0,TEST_NUM-1,SAMPLE_NUM)
    test_sample_points= np.round(test_sample_points).astype(int)
    NEED_ASK_TRAIN=1
    NEED_ASK_TEST=1
    NEED_OUTPUT=True
    #SUPER_INDEX=27

    print(train_sample_points)
    print(len(train_sample_points),len(test_sample_points))
    for i in range(8,9):
        MODEL_PATH="five_plus_two_optimization/train_model_new/train_base_model/TRAIN_RESULTS/ernie_base_model_9"
        #MODEL_PATH=f"five_plus_two_optimization/train_model_new/GRPO/save_model/grpo_fpt_api/grpo_fpt_api_2/step_{(i+1)*1000}" #var
        #OUTPUT_PATH=f"five_plus_two_optimization/train_model_new/GRPO/test/test_jsonl/grpo_fpt_api_2/auged_filt/{(i+1)*1000}_step_auged_filt.jsonl" #var
        OUTPUT_PATH="/home/jiuxing_li/five_plus_two_optimization/train_model_new/GRPO/test/test_jsonl/ernie_base_model_9_auged_filt.jsonl"
        #OUTPUT_PATH=f"five_plus_two_optimization/train_model_new/GRPO/test/test_jsonl/grpo_fpt_new_6_{(i+1)*1000}_step_auged.jsonl" #var
        #OUTPUT_PATH=f"five_plus_two_optimization/train_model_new/GRPO/test/test_jsonl/grpo_fpt_new_6_{(i+1)*1000}_step_super_auged_{SUPER_INDEX}.jsonl" #var
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,device_map="auto",torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model.eval()
        #训练集输出
        if NEED_ASK_TRAIN==1:
            if NEED_OUTPUT==True:
                #generate_answer(input_path=TRAIN_JSON,sample_points=train_sample_points,ask_round=ASK_ROUND,generate_token_num=GENERATE_TOKEN_NUM,output_path=OUTPUT_PATH)
                #generate_answer_auged(input_path=TRAIN_JSON,sample_points=train_sample_points,output_path=OUTPUT_PATH)
                generate_answer_filt_auged(input_path=TRAIN_JSON,sample_points=train_sample_points,output_path=OUTPUT_PATH)
                #generate_answer_super_auged(input_path=TRAIN_JSON,sample_points=train_sample_points,index=SUPER_INDEX,output_path=OUTPUT_PATH)
                #generate_answer_super_auged_filt(input_path=TRAIN_JSON,sample_points=train_sample_points,index=SUPER_INDEX,output_path=OUTPUT_PATH)
            else:
                #generate_answer(input_path=TRAIN_JSON,sample_points=train_sample_points,ask_round=ASK_ROUND,generate_token_num=GENERATE_TOKEN_NUM)
                #generate_answer_auged(input_path=TRAIN_JSON,sample_points=train_sample_points)
                generate_answer_filt_auged(input_path=TRAIN_JSON,sample_points=train_sample_points)
                #generate_answer_super_auged(input_path=TRAIN_JSON,sample_points=train_sample_points,index=SUPER_INDEX,output_path=OUTPUT_PATH)
                #generate_answer_super_auged_filt(input_path=TRAIN_JSON,sample_points=train_sample_points,index=SUPER_INDEX)
        if NEED_ASK_TEST==1:
            if NEED_OUTPUT==True:
                #generate_answer(input_path=TEST_JSON,sample_points=test_sample_points,ask_round=ASK_ROUND,generate_token_num=GENERATE_TOKEN_NUM,output_path=OUTPUT_PATH)
                #generate_answer_auged(input_path=TEST_JSON,sample_points=test_sample_points,output_path=OUTPUT_PATH)
                generate_answer_filt_auged(input_path=TEST_JSON,sample_points=test_sample_points,output_path=OUTPUT_PATH)
                #generate_answer_super_auged(input_path=TEST_JSON,sample_points=test_sample_points,index=SUPER_INDEX,output_path=OUTPUT_PATH)
                #generate_answer_super_auged_filt(input_path=TEST_JSON,sample_points=test_sample_points,index=SUPER_INDEX,output_path=OUTPUT_PATH)
            else:
                #generate_answer(input_path=TEST_JSON,sample_points=test_sample_points,ask_round=ASK_ROUND,generate_token_num=GENERATE_TOKEN_NUM)
                #generate_answer_auged(input_path=TEST_JSON,sample_points=test_sample_points)
                generate_answer_filt_auged(input_path=TEST_JSON,sample_points=test_sample_points)
                #generate_answer_super_auged(input_path=TEST_JSON,sample_points=test_sample_points,index=SUPER_INDEX)
                #generate_answer_super_auged_filt(input_path=TEST_JSON,sample_points=test_sample_points,index=SUPER_INDEX)
        
        end_time = time.time()
        print(f"代码平均执行时间：{(end_time - start_time)/((NEED_ASK_TRAIN+NEED_ASK_TEST)*SAMPLE_NUM):.4f} 秒")