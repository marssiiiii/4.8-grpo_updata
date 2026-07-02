import json
import torch
import sys
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/train_model_new/base_code")
sys.path.append("five_plus_two_optimization/train_model_new/GRPO/test/test_code")
sys.path.append("five_plus_two_optimization/train_model_new/train_base_model")
from reward_design_trunc import process_completion_predict_based_on_response,get_design_score,filter_completion_predict
from reward_design_dcr_price_trunc import get_floor_pre_designs,get_api_floor_design_score_item
from ernie_base_model_train import initialize_token_embedding
from five_plus_two_optimization.train_model_new.base_code.reward_design_trunc import process_completion_predict_filt
from five_plus_two_optimization.train_model_new.train_base_model.cmp_actionized_and_tokenized_api import construct_prompt
MODEL_ID="baidu/ERNIE-4.5-0.3B-PT"

from unsupport_force_calculate import get_unsolved_force_item
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
SAMPLE_NUM=50
#TRAIN_JSON = f"train_json_data/five_plus_two_train_jsonl_data/design_3.27/validation_data/grpo_train.jsonl"#var
TEST_JSON = f"train_json_data/five_plus_two_train_jsonl_data/design_3.27/validation_data/grpo_test.jsonl" 
TRAIN_JSON = f"train_json_data/five_plus_two_train_jsonl_data/design_3.27/grpo_phase_1_train_data/overfitting_data/train_set_inbox_force_prompt_overfitting(q(13)).jsonl"
#TEST_JSON = f"train_json_data/five_plus_two_train_jsonl_data/design_3.27/grpo_phase_1_train_data/test_set_inbox_force_prompt.jsonl"

TRAIN_NUM,TEST_NUM=13613,4803
#TRAIN_NUM,TEST_NUM=14651,3295
# ------------------------
# 加载模型与tokenizer
# ------------------------

class IS_IMPROVE():
    def __init__(self,house,design,floor,context,completion_predict,answer,pre_post_text,pre_lineload_text):
        self.house=house
        self.design=design
        self.floor=floor
        self.context=context
        self.completion_predict=completion_predict
        self.answer=answer
        self.pre_post_text=pre_post_text
        self.pre_lineload_text=pre_lineload_text
    
    @classmethod
    def is_pareto_evo(cls,delta_area,delta_dcr,delta_price,delta_error_num):
        # 改善方向：delta_area>0, delta_dcr>0, delta_error_num<0, delta_price<0
        # 前三项不允许恶化
        if delta_area < 0 or delta_dcr < 0 or delta_error_num > 0:
            return False
        front_three_improved = (delta_area > 0) or (delta_dcr > 0) or (delta_error_num < 0)
        if front_three_improved:
            return True
        # 前三项均未变化，price下降才算有效提升
        return delta_price < 0

    @classmethod
    def is_conditional_evo(cls,delta_area,delta_error_num,area_ratio,error_num,delta_dcr=None,delta_price=None):
        if error_num>1 or area_ratio<0.85:
            if delta_area>0 and (delta_error_num<=0 or (delta_error_num<=1 and error_num<=3)):
                return True
            if delta_error_num<0 and (delta_area>=0 or (delta_area>=-0.05 and area_ratio>0.5)):
                return True
        else:
            if delta_dcr<0:
                return False
            front_four_improved = (delta_area > 0) or (delta_dcr > 0) or (delta_error_num < 0) or (delta_price<0)
            return front_four_improved
        return False
    
    def is_improve(self):
        force_support_result_befor=get_unsolved_force_item(context=self.context,answer=self.completion_predict,
            pre_lineload_text=self.pre_lineload_text,
            pre_post_text=self.pre_post_text,BETWEEN_FLOOR_EPS=30,OVERLAP_THRESHOLD=0.5)
        force_support_result_after=get_unsolved_force_item(context=self.context,answer=self.answer,
            pre_lineload_text=self.pre_lineload_text,pre_post_text=self.pre_post_text,
            BETWEEN_FLOOR_EPS=30,OVERLAP_THRESHOLD=0.5)
        error_num_before=force_support_result_befor["unsolved_lineload_num"]+force_support_result_befor["unsolved_post_num"]
        error_num_after=force_support_result_after["unsolved_lineload_num"]+force_support_result_after["unsolved_post_num"]
        delta_error_num = error_num_after-error_num_before

        result_before,result_after=get_design_score(self.context,self.completion_predict),get_design_score(self.context,self.answer)
        delta_area=result_after["area_ratio"]-result_before["area_ratio"]

        if error_num_after>1 or result_after["area_ratio"]<0.85:
            result_record={"area_before":result_before["area_ratio"],"area_after":result_after["area_ratio"],
                "error_num_before":error_num_before,"error_num_after":error_num_after,
                "delta_area":delta_area,"delta_error_num":delta_error_num}
            print(result_record)
            
            if self.is_conditional_evo(delta_area=delta_area,delta_error_num=delta_error_num,
                                       area_ratio=result_after["area_ratio"],error_num=error_num_after)==True:
                print("符合有条件进化！")
                return 2,result_record
            else:
                print("不符合有条件进化")
                return 0,result_record
        
        designs,floor_design_pre,pre_context,pre_answer=get_floor_pre_designs(
            target_house=self.house,target_design=self.design,target_floor=self.floor,DATA_PATH=TRAIN_JSON)
        
        base_api_design_result,api_design_result=None,None
        while base_api_design_result is None or api_design_result is None:
            base_api_design_result=get_api_floor_design_score_item(context=self.context,house=self.house,answer=self.completion_predict,
                        floor=self.floor,designs=designs,floor_design_pre=floor_design_pre,
                        pre_post_text=self.pre_post_text,pre_lineload_text=self.pre_lineload_text,design=self.design)
            api_design_result=get_api_floor_design_score_item(context=self.context,answer=self.answer,
                            house=self.house,floor=self.floor,designs=designs,floor_design_pre=floor_design_pre,
                            pre_post_text=self.pre_post_text,pre_lineload_text=self.pre_lineload_text,design=self.design)
        delta_dcr = api_design_result["dcr"]-base_api_design_result["dcr"]
        delta_price = api_design_result["floor_price"]-base_api_design_result["floor_price"]

        result_record={"area_before":result_before["area_ratio"],"area_after":result_after["area_ratio"],
            "dcr_before":base_api_design_result["dcr"],"dcr_after":api_design_result["dcr"],
            "price_before":base_api_design_result["floor_price"],"price_after":api_design_result["floor_price"],
            "error_num_before":error_num_before,"error_num_after":error_num_after,
            "delta_area":delta_area,"delta_dcr":delta_dcr,"delta_error_num":delta_error_num,"delta_price":delta_price}
        print(result_record)
        
        if self.is_conditional_evo(delta_area=delta_area,delta_dcr=delta_dcr,delta_price=delta_price,
                        delta_error_num=delta_error_num,area_ratio=result_after["area_ratio"],error_num=error_num_after)==True:
            print("符合有条件进化！")
            return 2,result_record
        else:
            print("不符合有条件进化")
            return 0,result_record
    
    def is_improve_only_area(self):
        result_before,result_after=get_design_score(self.context,self.completion_predict),get_design_score(self.context,self.answer)
        delta_area=result_after["area_ratio"]-result_before["area_ratio"]
        result_record={"area_before":result_before["area_ratio"],"area_after":result_after["area_ratio"],
            "delta_area":delta_area}
        print(result_record)
        if delta_area>0:
            print("有效面积优化！")
            return 2,result_record
        else:
            print("有效面积未优化")
            return 0,result_record
    
    def is_improve_only_error(self):
        if self.pre_post_text!=None and self.pre_lineload_text!=None:
            force_support_result_befor=get_unsolved_force_item(context=self.context,answer=self.completion_predict,
                pre_lineload_text=self.pre_lineload_text,
                pre_post_text=self.pre_post_text,BETWEEN_FLOOR_EPS=30,OVERLAP_THRESHOLD=0.5)
            force_support_result_after=get_unsolved_force_item(context=self.context,answer=self.answer,
                pre_lineload_text=self.pre_lineload_text,pre_post_text=self.pre_post_text,
                BETWEEN_FLOOR_EPS=30,OVERLAP_THRESHOLD=0.5)
            error_num_before=force_support_result_befor["unsolved_lineload_num"]+force_support_result_befor["unsolved_post_num"]
            error_num_after=force_support_result_after["unsolved_lineload_num"]+force_support_result_after["unsolved_post_num"]
        else:
            error_num_before,error_num_after=0,0
        delta_error_num = error_num_after-error_num_before
        result_record={"error_num_before":error_num_before,"error_num_after":error_num_after,
            "delta_error_num":delta_error_num}
        print(result_record)
        
        if delta_error_num<0:
            print("error_num优化！")
            return 2,result_record
        else:
            print("error_num未优化")
            return 0,result_record
    
    def is_improve_only_dcr(self):
        designs,floor_design_pre,pre_context,pre_answer=get_floor_pre_designs(
                target_house=self.house,target_design=self.design,target_floor=self.floor,DATA_PATH=TRAIN_JSON)
        base_api_design_result,api_design_result=None,None
        while base_api_design_result is None or api_design_result is None:
            base_api_design_result=get_api_floor_design_score_item(context=self.context,house=self.house,answer=self.completion_predict,
                        floor=self.floor,designs=designs,floor_design_pre=floor_design_pre,
                        pre_post_text=self.pre_post_text,pre_lineload_text=self.pre_lineload_text,design=self.design)
            api_design_result=get_api_floor_design_score_item(context=self.context,answer=self.answer,
                            house=self.house,floor=self.floor,designs=designs,floor_design_pre=floor_design_pre,
                            pre_post_text=self.pre_post_text,pre_lineload_text=self.pre_lineload_text,design=self.design)
        #print(f"api_design_result:{api_design_result},base_api_design_result:{base_api_design_result}")
        delta_dcr = api_design_result["dcr"]-base_api_design_result["dcr"]

        result_record={"dcr_before":base_api_design_result["dcr"],"dcr_after":api_design_result["dcr"],
            "delta_dcr":delta_dcr}
        print(result_record)
        
        if delta_dcr>0:
            print("dcr优化！")
            return 2,result_record
        else:
            print("dcr未优化！")
            return 0,result_record

def generate_answer_filt_auged(input_path,PARAMETERS,output_path=None,down=None,up=None):
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if down!=None and i<down:
                continue
            if up!=None and i>=up:
                continue
            if i>SAMPLE_NUM:
                continue
            print(i,MODEL_PATH,OUTPUT_PATH)
            record = json.loads(line)
            house=record["house"]
            design=record["design"]
            floor=record["floor"]
            context=record["context"]
            prompt = record["prompt"]
            completion_predict=record["completion_predict"]
            bound=record["bound"]
            pre_post_text=record["pre_post_text"]
            pre_lineload_text=record["pre_lineload_text"]
            
            generate_round=0
            response_list,improved_response_list,improved_result_list=[],[],[]
            round_completion_predict=process_completion_predict_filt(context=context,completion_predict=completion_predict,response="")
            while generate_round<PARAMETERS["MAX_GENERATE_ROUND"]:
                print(i,MODEL_PATH,OUTPUT_PATH)
                print(f"{i}>={down} and {i}<{up}")
                print(f"ASK ROUND:{generate_round}")
                prompt=construct_prompt(context=context,completion_predict=round_completion_predict,
                                    pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text) #构造prompt
                prompt_token_length=len(tokenizer(prompt).input_ids)
                
                generate_prefix = tokenizer(prompt,return_tensors="pt").to("cuda").input_ids #依据prompt生成response
                response=""
                if generate_round%2==0:
                    generate_token_num=PARAMETERS["generate_token_num_1"]
                else:
                    generate_token_num=PARAMETERS["generate_token_num_2"]
                gen = model.generate(generate_prefix, max_new_tokens=generate_token_num, do_sample=True,temperature=PARAMETERS["temperature"])
                generated_ids = gen[0,prompt_token_length:]
                generated_tokens = tokenizer.decode(generated_ids, skip_special_tokens=False)
                response+=generated_tokens
                #print(f"response:{response}")
                response_list.append(response)

                response_processed=process_completion_predict_filt(context=context,
                        completion_predict=round_completion_predict,response=response)
                #print(f"response_processed:{response_processed}") #组合round_completion_predict和response
                cls_is_improve=IS_IMPROVE(house=house,floor=floor,design=design,context=context,
                completion_predict=round_completion_predict,answer=response_processed,pre_post_text=pre_post_text,
                pre_lineload_text=pre_lineload_text)

                CUR_IS_IMPROVE,improve_result=cls_is_improve.is_improve_only_area() #var
                if CUR_IS_IMPROVE==2: #如果有提升则把round_completion_predict更新为response_processed，否则不更新
                    round_completion_predict=response_processed
                    improved_response_list.append(response)
                    improved_result_list.append(improve_result)
                generate_round=generate_round+1

            if output_path!=None:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"house":house,"floor":floor,"design":design,"bound":bound,"context":context,
                                        "prompt": prompt,"completion_predict":completion_predict,
                                        #"response_list":response_list,
                                        "improved_response_list":improved_response_list,
                                        "improved_result_list":improved_result_list,
                                        "answer":response_processed,
                                        "pre_post_text":pre_post_text,
                                        "pre_lineload_text":pre_lineload_text}
                                        , ensure_ascii=False) + "\n")

def generate_answer_super_auged_filt(input_path,index,generate_token_num_1=100,generate_token_num_2=200,temperature=0.7,MAX_GENERATE_ROUND=99999999,output_path=None):
    IS_IMPROVE_FLAG=False
    improve_step,improve_ratio=[],[]
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(i,MODEL_PATH,OUTPUT_PATH)
            record = json.loads(line)
            house=record["house"]
            floor=record["floor"]
            design=record["design"]
            context=record["context"]
            prompt = record["prompt"]
            completion_predict=record["completion_predict"]
            bound=record["bound"]
            pre_post_text=record["pre_post_text"]
            pre_lineload_text=record["pre_lineload_text"]
            generate_round=0
            response_list,improved_response_list,improved_result_list=[],[],[]
            round_completion_predict=process_completion_predict_filt(context=context,completion_predict=completion_predict,response="")
            print(house,floor)
            while generate_round<MAX_GENERATE_ROUND:
                print(index,house,floor)
                print(f"ASK ROUND:{generate_round}")
                prompt=construct_prompt(context=context,completion_predict=round_completion_predict,pre_post_text=pre_post_text,
                pre_lineload_text=pre_lineload_text) #构造prompt
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
                #print(f"response:{response}")
                response_list.append(response)

                response_processed=process_completion_predict_filt(context=context,completion_predict=round_completion_predict
                                            ,response=response)
                #print(f"response_processed:{response_processed}") #组合round_completion_predict和response
            
                cls_is_improve=IS_IMPROVE(house=house,floor=floor,design=design,context=context,
                completion_predict=round_completion_predict,answer=response_processed,pre_post_text=pre_post_text,
                pre_lineload_text=pre_lineload_text)
                CUR_IS_IMPROVE,improve_result=cls_is_improve.is_improve_only_error()

                if CUR_IS_IMPROVE==2: #如果有提升则把round_completion_predict更新为response_processed，否则不更新
                    round_completion_predict=response_processed
                    improved_response_list.append(response)
                    improved_result_list.append(improve_result)
                    improve_step.append(generate_round)
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
                        f.write(json.dumps({"house":house,"floor":floor,"design":design,"bound":bound,
                                            "improve_step":improve_step,
                                            "generate_round":generate_round,
                                            "context":context,
                                            "prompt": prompt,"completion_predict":completion_predict,
                                            #"response_list":response_list,
                                            "improved_response_list":improved_response_list,
                                            "improved_result_list":improved_result_list,"answer":response_processed,
                                            "pre_post_text":pre_post_text,"pre_lineload_text":pre_lineload_text}
                                            , ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # ------------------------
    # 构建测试数据
    # ------------------------
    NEED_ASK_TRAIN=1
    NEED_ASK_TEST=1
    NEED_OUTPUT=True
    #SUPER_INDEX=50 #[17,20,34,65,73,100]
    PARAMETERS={"generate_token_num_1":100,"generate_token_num_2":200,"temperature":1,"MAX_GENERATE_ROUND":5}
    #DOWN,UP=1900,2000
    for i in range(0,1):
        MODEL_PATH="five_plus_two_optimization/train_model_new/GRPO/save_model/grpo_fpt_api/grpo_fpt_api_5/step_1000"
        OUTPUT_PATH="five_plus_two_optimization/train_model_new/train_base_model/test/test_jsonl/post_train_2/base_area_q_13.jsonl"
        #MODEL_PATH=f"five_plus_two_optimization/train_model_new/GRPO/save_model/grpo_fpt_api/grpo_fpt_api_5/step_{(i+1)*1000}"
        #OUTPUT_PATH=f"five_plus_two_optimization/train_model_new/GRPO/test/test_jsonl/grpo_fpt_api_5/error_auged_filt/{i+1}_auged_filt_5round.jsonl" #var
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,device_map="auto",torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        '''
        #check-point加载
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID) #自动加载Gemma3对应tokenizer
        if tokenizer.pad_token is None: #如果tokenizer中没有pad_token,就用eos_token填充
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
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
        
        model.eval()
        #训练集输出
        if NEED_ASK_TRAIN==1:
            if NEED_OUTPUT==True:
                generate_answer_filt_auged(input_path=TRAIN_JSON,output_path=OUTPUT_PATH,PARAMETERS=PARAMETERS)
                #generate_answer_super_auged_filt(input_path=TRAIN_JSON,index=SUPER_INDEX+1,output_path=OUTPUT_PATH)
            else:
                generate_answer_filt_auged(input_path=TRAIN_JSON,PARAMETERS=PARAMETERS)
                #generate_answer_super_auged_filt(input_path=TRAIN_JSON,index=SUPER_INDEX+1)
        if NEED_ASK_TEST==1:
            if NEED_OUTPUT==True:
                generate_answer_filt_auged(input_path=TEST_JSON,output_path=OUTPUT_PATH,PARAMETERS=PARAMETERS)
                #generate_answer_super_auged_filt(input_path=TEST_JSON,index=SUPER_INDEX,output_path=OUTPUT_PATH)
            else:
                generate_answer_filt_auged(input_path=TEST_JSON,PARAMETERS=PARAMETERS)
                #generate_answer_super_auged_filt(input_path=TEST_JSON,index=SUPER_INDEX)
        
        end_time = time.time()
        print(f"代码平均执行时间：{(end_time - start_time)/((NEED_ASK_TRAIN+NEED_ASK_TEST)*100):.4f} 秒")