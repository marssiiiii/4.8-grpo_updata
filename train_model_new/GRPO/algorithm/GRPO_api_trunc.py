# -*- coding: utf-8 -*-
from typing import Dict
import torch
from base_five_plus_two import AlgorithmBase
import os, re, random, time, json, requests
from ref_client_five_plus_two import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list 
from vllm import LLM, SamplingParams
from modelscope.msdatasets import MsDataset
from math_verify import parse, verify, ExprExtractionConfig
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import numpy as np
import sys
import random
import math
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/train_model_new/base_code")

sys.path.append("/home/jiuxing_li")
sys.path.append("/home/jiuxing_li/code/GRPO/test/test_code")
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/five_plus_two_data_embedding_3_11")
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/five_plus_two_test")

from five_plus_two_optimization.train_model_new.base_code.reward_design_dcr_price_trunc import trunc_floor_design_response_and_calculate_score,get_floor_pre_designs
from five_plus_two_optimization.train_model_new.base_code.solve_force_error import \
    get_resolve_support_response_from_prompt,visualize_gen_sample_process,combine_resp

os.environ["DS_SKIP_CUDA_CHECK"] = "1"

def get_design_price_score(floor_price_list, diaphragm_area_ratio_list, error_num_list):
    floor_price = np.array(floor_price_list)
    area_ratio = np.array(diaphragm_area_ratio_list)
    error_num = np.array(error_num_list)
    n = len(floor_price)

    sorted_idx = np.argsort(floor_price)
    rank = np.empty(n)
    rank[sorted_idx] = np.arange(n)
    rank_norm = rank / (n - 1) #rank_norm属于[0,1]

    score = (1 - rank_norm) * area_ratio * np.exp(-error_num)
    return score.tolist()

def std_list(lst):
    mean = sum(lst) / len(lst)
    std = (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5
    if std==0:
        return [0.0] * len(lst)
    return [(x - mean) / std for x in lst]

def get_floor_design_score(diaphragm_delta_list,response_encoded_list, dcr_delta_list, error_num_list,
                    invalid_process_num_list, floor_price_list,diaphragm_area_ratio_list,PARAMETERS):
    floor_design_score_list=[]
    design_error_score_list,design_invalid_process_score_list,design_diaphragm_score_list,design_dcr_score_list,design_price_score_list=[],[],[],[],[]
    design_price_score_list=get_design_price_score(floor_price_list, diaphragm_area_ratio_list, error_num_list)
    for i in range(len(diaphragm_delta_list)):
        delta_diaphragm, delta_dcr, error_num, invalid_process_num,diaphragm_area_ratio,response_len_ratio = \
            diaphragm_delta_list[i], dcr_delta_list[i], error_num_list[i], invalid_process_num_list[i],\
            diaphragm_area_ratio_list[i], len(response_encoded_list[i].input_ids)/PARAMETERS["MAX_TOKEN_NUM"]
        design_error_score=-0.5*error_num
        design_invalid_process_score=-0.2*invalid_process_num
        if delta_diaphragm!=0:
            design_diaphragm_score=2*delta_diaphragm/response_len_ratio
        else:
            design_diaphragm_score=-0.5*response_len_ratio
        design_dcr_score=delta_dcr*diaphragm_area_ratio*math.exp(-error_num)

        design_error_score_list.append(design_error_score)
        design_invalid_process_score_list.append(design_invalid_process_score)
        design_diaphragm_score_list.append(design_diaphragm_score)
        design_dcr_score_list.append(design_dcr_score)
    
    design_error_score_list = std_list(design_error_score_list)
    design_invalid_process_score_list = std_list(design_invalid_process_score_list)
    design_diaphragm_score_list = std_list(design_diaphragm_score_list)
    design_dcr_score_list = std_list(design_dcr_score_list)
    design_price_score_list = std_list(design_price_score_list)

    for i in range(len(design_diaphragm_score_list)):
        floor_design_score=PARAMETERS["lambda_1"]*design_error_score_list[i]+PARAMETERS["lambda_2"]*design_invalid_process_score_list[i]+\
            PARAMETERS["lambda_3"]*design_diaphragm_score_list[i]+PARAMETERS["lambda_4"]*design_dcr_score_list[i]+\
            PARAMETERS["lambda_5"]*design_price_score_list[i]
        floor_design_score=max(-1*PARAMETERS["SCORE_RANGE"],min(PARAMETERS["SCORE_RANGE"],floor_design_score))
        floor_design_score_list.append(floor_design_score)
    
    return {
        "floor_design_score_list": floor_design_score_list,
        "design_error_score_list": design_error_score_list,
        "design_invalid_process_score_list": design_invalid_process_score_list,
        "design_diaphragm_score_list": design_diaphragm_score_list,
        "design_dcr_score_list": design_dcr_score_list,
        "design_price_score_list": design_price_score_list
    }

class Algorithm(AlgorithmBase):
    #engin:模型推理/训练使用的engine(类似于model,输入inputs返回下一步token的logits)
    #beta:KL正则惩罚参数
    #clip_param:PPO的clipping参数
    #compute_gen_logps:确定是否计算生成的log probability
    def __init__(self, engine, tokenizer, beta: float, clip_param: float, compute_gen_logps: bool, **_extra):
        super().__init__(engine, tokenizer, beta=beta, clip_param=clip_param, compute_gen_logps=compute_gen_logps)

    @staticmethod
    # Generate sub-processes sampling & scoring
    #一个静态方法，用于在子进程中生成数据、计算奖励、上传到服务器
    #Q:队列，用于接受更新的模型参数
    #mode_path:模型路径
    #ref_server_url:上传奖励和数据的参考服务器
    #num_pre_Q:每个问题生成多少候选答案
    #train_batch_size:训练的batch大小
    #Q_batch_size:每次采样多少问题生成答案
    #compute_gen_logps:确定是否计算生成的log probability
    def gen_worker(Q, model_path: str, gen_device: int, ref_server_url: str,
                num_pre_Q: int, train_batch_size: int, compute_gen_logps: bool,
                  Q_batch_size: int,TRAIN_JSONL:str,TRAIN_BEG:int,TRAIN_END:int):  # ✅ 新增参数
        GPU_UTILIZATION=0.12
        PARAMETERS={"lambda_1": 0.5, "lambda_2": 0.5, "lambda_3": 0.5, "lambda_4": 0.5,"lambda_5": 0.5, #var
                "MAX_TOKEN_NUM": 200,"SCORE_RANGE":2.5,"TEMPERATURE":0.8}
        BASE_TRUNC_RESULT={"prompt_list": [], "response_list": [], "response_encoded_list": [], "answer_list": [],
                    "diaphragm_delta_score_list": [],"diaphragm_area_ratio_list": [],"dcr_delta_list": [],"error_num_list": [],
                    "invalid_process_num_list": [],"floor_price_list": []}

        # ✅ 设置当前worker的GPU
        torch.cuda.set_device(gen_device)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gen_device)
        print(f"GEN_WORKER Starting on GPU {gen_device}, PID: {os.getpid()}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # ✅ 调整GPU利用率，避免多worker时OOM
        print("gen_worker初始化vllm_gen")
        vllm_gen = LLM(model=model_path, gpu_memory_utilization=GPU_UTILIZATION) #gpu_memory_utilization指的是vLLM 在「每一张 GPU 上」允许占用的显存比例上限
        print("gen_worker初始化vllm_gen成功")
        ref_server_ver = 'tensor'

        sampling_params = SamplingParams(n=num_pre_Q, temperature=PARAMETERS["TEMPERATURE"], max_tokens=PARAMETERS["MAX_TOKEN_NUM"])

        #加载数据
        QAs = []
        with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < TRAIN_BEG or i > TRAIN_END:
                    continue
                # ✅ 关键：数据分片，每个worker处理 i % num_workers == worker_id 的数据
                if line.strip():
                    data = json.loads(line)
                    QAs.append({
                        "house": data["house"],
                        "floor": data["floor"],
                        "design": data["design"],
                        "bound": data["bound"],
                        "context": data["context"],
                        "completion_predict": data["completion_predict"],
                        "prompt": data['prompt'],
                        "pre_post_text": data["pre_post_text"],
                        "pre_lineload_text": data["pre_lineload_text"]
                    })

        # ✅ 新增：模型版本追踪
        current_model_version = 0
        last_check_time = time.time()

        #一句prompt返回答案以及答案token_id(回答的多样性由sampling_params控制)
        def gen_answers(prompt_list):
            GEN_ANSWER_NEED_VISUALIZE=False
            tip_text = []
            for prompt in prompt_list:
                tip_text.append(prompt)
            voutputs = vllm_gen.generate(tip_text, sampling_params, use_tqdm=False) #依据sampling_params生成输出序列prompt1->[ans_1,ans_2,ans_num_pre_Q]
            response_list = []
            for v in voutputs: #将所有生成的answer都打平,返回answer+answer_token_id
                for z in v.outputs:
                    response=tokenizer.decode(z.token_ids)
                    response_list.append(response)
            
            full_prompt_list,response_after_list=[],[]
            for prompt in prompt_list:
                for _ in range(num_pre_Q):
                    full_prompt_list.append(prompt)
            for resp_index in range(len(response_list)):
                resp_pre=response_list[resp_index]
                if random.random()<0.5:
                    added_resp=get_resolve_support_response_from_prompt(prompt=full_prompt_list[resp_index],response=resp_pre)
                    resp_after=combine_resp(resp_pre=resp_pre,added_resp=added_resp)
                    #print(f"resp_pre:{resp_pre}")
                    #print(f"added_resp:{added_resp}")
                    #print(f"resp_after:{resp_after}")
                else:
                    resp_after=combine_resp(resp_pre=resp_pre,added_resp="")
                response_after_list.append(resp_after)

            if GEN_ANSWER_NEED_VISUALIZE:
                print("开始可视化生成样本的过程...")
                visualize_gen_sample_process(prompt_list=full_prompt_list,
                        response_list=response_list,response_after_list=response_after_list,num_pre_Q=num_pre_Q)
                print("可视化完成！")
            #print(f"gen_answers, prompt_list:{prompt_list}, response_list:{response_list}")
            return response_after_list

        # 奖励函数分两部分：正确性奖励+格式奖励
        # 正确性奖励:当前的逻辑是依据无支撑次数和房间数误差进行打分
        def reward_correct(context_list,completion_predict_list,response_list,house,floor,bound,pre_post_text,pre_lineload_text,design,gen_num):
            QUEUE_IS_FULL=False
            batch_return_result={"prompt_list":[],"response_list":[],"response_encoded_list":[],"score_list":[],
            "diaphragm_delta_score_list":[],"dcr_delta_list":[],"error_num_list":[],"invalid_process_num_list":[],
            "floor_price_list":[],"diaphragm_area_ratio_list":[],"is_improved":[]}
            for i in range(len(context_list)):
                if QUEUE_IS_FULL==True:
                    print("QUEUE IS FULL!")
                    break
                context,completion_predict,response=\
                    context_list[i],completion_predict_list[i],response_list[i]
                
                designs,floor_design_pre,pre_context,pre_answer=get_floor_pre_designs(target_house=house,target_design=design,target_floor=floor,DATA_PATH=TRAIN_JSONL)
                trunc_result={}
                for key in BASE_TRUNC_RESULT.keys():
                    trunc_result[key]=[]
                trunc_result=trunc_floor_design_response_and_calculate_score( #对response进行截断打分
                house=house,floor=floor,design=design,context=context,completion_predict=completion_predict,
                pre_context=pre_context,pre_answer=pre_answer,
                response=response,designs=designs,floor_design_pre=floor_design_pre,
                trunc_result=trunc_result,tokenizer=tokenizer,pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text,
                trunc_index=0,NEED_VISUALIZE=False,bound=bound)

                for trunc_index in range(len(trunc_result["prompt_list"])): #将trunc后的结果记录，如果大于等于gen_num则停止
                    if len(batch_return_result["prompt_list"])<gen_num:
                        batch_return_result["prompt_list"].append(trunc_result["prompt_list"][trunc_index])
                        batch_return_result["response_list"].append(trunc_result["response_list"][trunc_index])
                        batch_return_result["response_encoded_list"].append(trunc_result["response_encoded_list"][trunc_index])

                        batch_return_result["diaphragm_delta_score_list"].append(trunc_result["diaphragm_delta_score_list"][trunc_index])
                        batch_return_result["dcr_delta_list"].append(trunc_result["dcr_delta_list"][trunc_index])
                        batch_return_result["error_num_list"].append(trunc_result["error_num_list"][trunc_index])
                        batch_return_result["invalid_process_num_list"].append(trunc_result["invalid_process_num_list"][trunc_index])
                        batch_return_result["floor_price_list"].append(trunc_result["floor_price_list"][trunc_index])
                        batch_return_result["diaphragm_area_ratio_list"].append(trunc_result["diaphragm_area_ratio_list"][trunc_index])
                        if trunc_result["diaphragm_delta_score_list"][trunc_index]>0:
                            batch_return_result["is_improved"].append(1)
                        else:
                            batch_return_result["is_improved"].append(0)
                    else:
                        print(f"当前批次已满{gen_num}条数据，停止添加")
                        QUEUE_IS_FULL=True
                        break
                if i%4==0:
                    print(f"{i}/{len(context_list)},house:{house},floor:{floor},diaphragm_delta_score_list:{trunc_result['diaphragm_delta_score_list']},\
                dcr_delta_list:{trunc_result['dcr_delta_list']},error_num_list:{trunc_result['error_num_list']},\
                invalid_process_num_list:{trunc_result['invalid_process_num_list']},\
                floor_price_list:{trunc_result['floor_price_list']},\
                diaphragm_area_ratio_list:{trunc_result['diaphragm_area_ratio_list']}") #打印第一条数据的结果以供检验

            floor_design_score_result=get_floor_design_score(
            diaphragm_delta_list=batch_return_result["diaphragm_delta_score_list"],response_encoded_list=batch_return_result["response_encoded_list"], 
            dcr_delta_list=batch_return_result["dcr_delta_list"], error_num_list=batch_return_result["error_num_list"],
            invalid_process_num_list=batch_return_result["invalid_process_num_list"], 
            floor_price_list=batch_return_result["floor_price_list"],diaphragm_area_ratio_list=batch_return_result["diaphragm_area_ratio_list"],
            PARAMETERS=PARAMETERS)

            batch_return_result["score_list"]=floor_design_score_result["floor_design_score_list"]
            batch_return_result["design_error_score_list"] = floor_design_score_result["design_error_score_list"]
            batch_return_result["design_invalid_process_score_list"] = floor_design_score_result["design_invalid_process_score_list"]
            batch_return_result["design_diaphragm_score_list"] = floor_design_score_result["design_diaphragm_score_list"]
            batch_return_result["design_dcr_score_list"] = floor_design_score_result["design_dcr_score_list"]
            batch_return_result["design_price_score_list"] = floor_design_score_result["design_price_score_list"]

            batch_is_improved=batch_return_result["is_improved"]
            print(f"house:{house},floor:{floor},batch_is_improved:{batch_is_improved},floor_design_score_result:{floor_design_score_result}")
            return batch_return_result

        #对每个问题生成多个答案，对每个候选答案存储提问问题(含答案),奖励,答案,答案token_id
        def gen_samples(inputs):
            prompt_list = [x["prompt"] for x in inputs]
            house_list,floor_list=[x["house"] for x in inputs],[x["floor"] for x in inputs]
            response_list = gen_answers(prompt_list) #依据inputs提问生成答案
            return_result={"prompt_list":[],"response_list":[],"response_encoded_list":[],"score_list":[],"is_improved":[],
            "diaphragm_delta_score_list":[],"dcr_delta_list":[],"error_num_list":[],"invalid_process_num_list":[],
            "floor_price_list":[],"diaphragm_area_ratio_list":[],
            "design_error_score_list":[],"design_invalid_process_score_list":[],"design_diaphragm_score_list":[],
            "design_dcr_score_list":[],"design_price_score_list":[],
            "house_list":house_list,"floor_list":floor_list}
            for i, inp in enumerate(inputs):
                sample_context_list,sample_completion_predict_list,sample_response_list=[],[],[]
                for response in response_list[i*num_pre_Q:(i+1)*num_pre_Q]:
                    sample_context_list.append(inp["context"])
                    sample_completion_predict_list.append(inp["completion_predict"])
                    sample_response_list.append(response)
                batch_return_result=\
                    reward_correct(context_list=sample_context_list,completion_predict_list=sample_completion_predict_list
                    ,response_list=sample_response_list,house=inp["house"],floor=inp["floor"],
                    pre_post_text=inp["pre_post_text"],pre_lineload_text=inp["pre_lineload_text"],
                    design=inp["design"],gen_num=num_pre_Q,bound=inp["bound"])
                
                return_result["prompt_list"]=return_result["prompt_list"]+batch_return_result["prompt_list"] #合并数据
                return_result["response_list"]=return_result["response_list"]+batch_return_result["response_list"]
                return_result["response_encoded_list"]=return_result["response_encoded_list"]+batch_return_result["response_encoded_list"]
                return_result["score_list"]=return_result["score_list"]+batch_return_result["score_list"]
                return_result["is_improved"]=return_result["is_improved"]+batch_return_result["is_improved"]

                return_result["diaphragm_delta_score_list"]=return_result["diaphragm_delta_score_list"]+batch_return_result["diaphragm_delta_score_list"]
                return_result["dcr_delta_list"]=return_result["dcr_delta_list"]+batch_return_result["dcr_delta_list"]
                return_result["error_num_list"]=return_result["error_num_list"]+batch_return_result["error_num_list"]
                return_result["invalid_process_num_list"]=return_result["invalid_process_num_list"]+batch_return_result["invalid_process_num_list"]
                return_result["floor_price_list"]=return_result["floor_price_list"]+batch_return_result["floor_price_list"]
                return_result["diaphragm_area_ratio_list"]=return_result["diaphragm_area_ratio_list"]+batch_return_result["diaphragm_area_ratio_list"]

                return_result["design_error_score_list"] = return_result["design_error_score_list"] + batch_return_result["design_error_score_list"]
                return_result["design_invalid_process_score_list"] = return_result["design_invalid_process_score_list"] + batch_return_result["design_invalid_process_score_list"]
                return_result["design_diaphragm_score_list"] = return_result["design_diaphragm_score_list"] + batch_return_result["design_diaphragm_score_list"]
                return_result["design_dcr_score_list"] = return_result["design_dcr_score_list"] + batch_return_result["design_dcr_score_list"]
                return_result["design_price_score_list"] = return_result["design_price_score_list"] + batch_return_result["design_price_score_list"]

            return_result["score_list"] = [
                torch.tensor(s, dtype=torch.float32)
                for s in return_result["score_list"]
            ]
            return return_result
        
        # ✅ 新增：模型版本追踪
        def check_and_update_model(it):
            """检查并更新模型"""
            nonlocal current_model_version, last_check_time
            nonlocal vllm_gen
            
            # 每30秒检查一次
            if time.time() - last_check_time < 30:
                return False
            
            last_check_time = time.time()
            
            try:
                # 查询最新模型版本
                response = requests.get(f"{ref_server_url}/model_version", timeout=5)
                version_info = response.json()
                print(f"gen_worker_version_info:{version_info}")
                
                new_version = version_info.get('version', 0)
                model_path_new = version_info.get('path')
                train_is_updating = version_info.get('is_updating', False)
                
                #如果train过程正在更新模型,返回False
                if train_is_updating:
                    print(f'[GEN_WORKER] Model is being updated by train process, waiting...')
                    return False
                
                # 如果有新版本
                if new_version > current_model_version and model_path_new:

                    # 通知ref_client：gen_worker开始更新
                    lock_data = {'it': it}
                    response = requests.post(
                        f"{ref_server_url}/lock_gen_worker_update",
                        data=json.dumps(lock_data).encode(),
                        timeout=5
                    )
                    print(f"[GEN_WORKER] update locked: {response.json()}")

                    print(f'[GEN_WORKER] Updating model from version {current_model_version} to {new_version}')
                    print(f'[GEN_WORKER] Loading from: {model_path_new}')
                    
                    # ✅ 修复：不用 del，直接覆盖赋值
                    # 先把旧对象引用清空，再清显存，再加载新模型
                    vllm_gen = None          # 断开旧对象引用
                    torch.cuda.empty_cache() # 清理显存
                    
                    vllm_gen = LLM(model=model_path_new, gpu_memory_utilization=GPU_UTILIZATION)
                    current_model_version = new_version
                    
                    print(f'[GEN_WORKER] Model updated successfully to version {new_version}')

                    # 4. 通知ref_client：更新完成（解锁）
                    unlock_data = {
                        'version': new_version,
                        'path': model_path_new,
                        'it': it
                    }
                    response = requests.post(
                        f"{ref_server_url}/unlock_gen_worker_update",
                        data=json.dumps(unlock_data).encode(),
                        timeout=5
                    )
                    print(f"[GEN_WORKER] update unlocked: {response.json()}")

                    return True
            except Exception as e:
                print(f'[GEN_WORKER] Error checking/updating model: {e}')
            
            return False
            
        def get_queue_size(ref_server_url: str) -> int:
            """查询ref_server队列大小"""
            try:
                r = requests.get(f"{ref_server_url}/queue_size")
                data = r.json()  # 解析 JSON
                return data['queue_size']
            except Exception as e:
                print("Failed to get queue size:", e)
                return 0
        
        #每轮随机采样问题，生成答案，计算奖励，上传给服务器
        MODEL_KEEP_UPDATED=False
        for it in range(999999999):
            MODEL_UPDATED=False
            print(f"gen_worker begin:{it}")

            # ✅ 动态阈值：不同worker有不同的等待策略，避免同步阻塞
            queue_size = get_queue_size(ref_server_url)
            if queue_size >= 150:  # 如果队列堆积超过5,进行睡眠,缓慢问题生成过程
                print(f'Queue full ({queue_size}), waiting...')
                time.sleep(5)
                continue

            # ✅ 定期检查模型更新
            if it % 3 == 0:
                MODEL_UPDATED=check_and_update_model(it)

            if MODEL_UPDATED==False and MODEL_KEEP_UPDATED==True: #如果上一步没有上传数据成功，保留模型更新标志,继续生成数据
                MODEL_UPDATED=True

            print(f"当前队列长度：{queue_size}")

            #只有模型刚开始或者模型更新一次后（在成功生成一批数据的情况下）
            if it!=0 and MODEL_UPDATED==False:
                print(f"模型未更新，持续等待")
                time.sleep(5)
                continue

            inputs = random.sample(QAs,Q_batch_size) #采样问题
            tic = time.time()
            #prompt_list,response_list,response_encoded_list,score_list,is_improved,house_list,floor_list = gen_samples(inputs) #依据采样问题得到采样答案
            gen_sample_return_result = gen_samples(inputs)
            #print(f"gen_sample检验：prompt_inputs:{prompt_inputs}, rewards:{rewards},answers:{answers}, ans_token_ids:{ans_token_ids}")
            #print(f'time: {time.time()-tic:.2f}s    ', 'rewards:', rewards)

            # advantage standardization
            NUM_GENERATE=0
            MODEL_KEEP_UPDATED=True
            while NUM_GENERATE<Q_batch_size:
                i=NUM_GENERATE
                print(f"当前是{i+1}/{Q_batch_size}")
                pp=gen_sample_return_result["prompt_list"][i]
                prompt_ids = tokenizer(pp, return_tensors="pt")["input_ids"]
                plen = prompt_ids.shape[1]
                house,floor=gen_sample_return_result["house_list"][i],gen_sample_return_result["floor_list"][i]
                curr_scores = gen_sample_return_result["score_list"][i*num_pre_Q:(i+1)*num_pre_Q]
                curr_is_improved = gen_sample_return_result["is_improved"][i*num_pre_Q:(i+1)*num_pre_Q]
                curr_response_encoded_list = gen_sample_return_result["response_encoded_list"][i*num_pre_Q:(i+1)*num_pre_Q]

                curr_diaphragm_delta_score_list = gen_sample_return_result["diaphragm_delta_score_list"][i*num_pre_Q:(i+1)*num_pre_Q]
                curr_dcr_delta_list = gen_sample_return_result["dcr_delta_list"][i*num_pre_Q:(i+1)*num_pre_Q]
                curr_error_num_list = gen_sample_return_result["error_num_list"][i*num_pre_Q:(i+1)*num_pre_Q]
                curr_invalid_process_num_list = gen_sample_return_result["invalid_process_num_list"][i*num_pre_Q:(i+1)*num_pre_Q]
                curr_floor_price_list = gen_sample_return_result["floor_price_list"][i*num_pre_Q:(i+1)*num_pre_Q]
                curr_diaphragm_area_ratio_list = gen_sample_return_result["diaphragm_area_ratio_list"][i*num_pre_Q:(i+1)*num_pre_Q]

                curr_design_error_score_list = gen_sample_return_result["design_error_score_list"][i*num_pre_Q:(i+1)*num_pre_Q]
                curr_design_invalid_process_score_list = gen_sample_return_result["design_invalid_process_score_list"][i*num_pre_Q:(i+1)*num_pre_Q]
                curr_design_diaphragm_score_list = gen_sample_return_result["design_diaphragm_score_list"][i*num_pre_Q:(i+1)*num_pre_Q]
                curr_deisgn_dcr_score_list = gen_sample_return_result["design_dcr_score_list"][i*num_pre_Q:(i+1)*num_pre_Q]
                curr_design_price_score_list = gen_sample_return_result["design_price_score_list"][i*num_pre_Q:(i+1)*num_pre_Q]

                #求得curr_rewards进行标准化
                mean_s,std_s = np.mean(curr_scores),np.std(curr_scores)
                
                if np.sum(curr_is_improved)<3: #var
                    print("显著提升原设计样本不足！")
                    NUM_GENERATE=NUM_GENERATE+1
                    continue
                
                NUM_GENERATE=NUM_GENERATE+1
                MODEL_KEEP_UPDATED=False #如果成功上传，则不保留更新标志
                curr_scores_std = (curr_scores - mean_s) / (std_s + 1e-4)

                if ref_server_ver == 'tensor':
                    for ii in range(0, num_pre_Q, train_batch_size):
                        sub_scores=curr_scores[ii:ii+train_batch_size]
                        sub_scores_std = curr_scores_std[ii:ii+train_batch_size]

                        sub_diaphragm_delta_score_list = curr_diaphragm_delta_score_list[ii:ii+train_batch_size]
                        sub_dcr_delta_list = curr_dcr_delta_list[ii:ii+train_batch_size]
                        sub_error_num_list = curr_error_num_list[ii:ii+train_batch_size]
                        sub_invalid_process_num_list = curr_invalid_process_num_list[ii:ii+train_batch_size]
                        sub_floor_price_list = curr_floor_price_list[ii:ii+train_batch_size]
                        sub_diaphragm_area_ratio_list = curr_diaphragm_area_ratio_list[ii:ii+train_batch_size]

                        sub_design_error_score_list = curr_design_error_score_list[ii:ii+train_batch_size]
                        sub_design_invalid_process_score_list = curr_design_invalid_process_score_list[ii:ii+train_batch_size]
                        sub_design_diaphragm_score_list = curr_design_diaphragm_score_list[ii:ii+train_batch_size]
                        sub_deisgn_dcr_score_list = curr_deisgn_dcr_score_list[ii:ii+train_batch_size]
                        sub_design_price_score_list = curr_design_price_score_list[ii:ii+train_batch_size]
                        
                        sub_scores = torch.tensor(sub_scores, dtype=torch.float32) if len(sub_scores) > 0 else torch.tensor([], dtype=torch.float32)
                        sub_scores_std = torch.tensor(sub_scores_std, dtype=torch.float32) if len(sub_scores_std) > 0 else torch.tensor([], dtype=torch.float32)

                        sub_diaphragm_delta_score_list = torch.tensor(sub_diaphragm_delta_score_list, dtype=torch.float32) if len(sub_diaphragm_delta_score_list) > 0 else torch.tensor([], dtype=torch.float32)
                        sub_dcr_delta_list = torch.tensor(sub_dcr_delta_list, dtype=torch.float32) if len(sub_dcr_delta_list) > 0 else torch.tensor([], dtype=torch.float32)
                        sub_error_num_list = torch.tensor(sub_error_num_list, dtype=torch.float32) if len(sub_error_num_list) > 0 else torch.tensor([], dtype=torch.float32)
                        sub_invalid_process_num_list = torch.tensor(sub_invalid_process_num_list, dtype=torch.float32) if len(sub_invalid_process_num_list) > 0 else torch.tensor([], dtype=torch.float32)
                        sub_floor_price_list = torch.tensor(sub_floor_price_list, dtype=torch.float32) if len(sub_floor_price_list) > 0 else torch.tensor([], dtype=torch.float32)
                        sub_diaphragm_area_ratio_list = torch.tensor(sub_diaphragm_area_ratio_list, dtype=torch.float32) if len(sub_diaphragm_area_ratio_list) > 0 else torch.tensor([], dtype=torch.float32)

                        sub_design_error_score_list = torch.tensor(sub_design_error_score_list, dtype=torch.float32) if len(sub_design_error_score_list) > 0 else torch.tensor([], dtype=torch.float32)
                        sub_design_invalid_process_score_list = torch.tensor(sub_design_invalid_process_score_list, dtype=torch.float32) if len(sub_design_invalid_process_score_list) > 0 else torch.tensor([], dtype=torch.float32)
                        sub_design_diaphragm_score_list = torch.tensor(sub_design_diaphragm_score_list, dtype=torch.float32) if len(sub_design_diaphragm_score_list) > 0 else torch.tensor([], dtype=torch.float32)
                        sub_deisgn_dcr_score_list = torch.tensor(sub_deisgn_dcr_score_list, dtype=torch.float32) if len(sub_deisgn_dcr_score_list) > 0 else torch.tensor([], dtype=torch.float32)
                        sub_design_price_score_list = torch.tensor(sub_design_price_score_list, dtype=torch.float32) if len(sub_design_price_score_list) > 0 else torch.tensor([], dtype=torch.float32)

                        sub_response_encoded_batch = curr_response_encoded_list[ii: ii + train_batch_size]
                        if len(sub_response_encoded_batch)==0:
                            print("当前子批次没有数据，跳过上传")
                            continue
                        sub_response_ids = [x["input_ids"] for x in sub_response_encoded_batch]
                        sub_response_ids_tensor = [torch.tensor(lst) for lst in sub_response_ids]
                        sub_response_ids_pad = pad_sequence(sub_response_ids_tensor, batch_first=True, padding_value=tokenizer.pad_token_id)

                        Qrep = prompt_ids.repeat(1, sub_response_ids_pad.shape[0]).view(-1, plen)
                        merged_ids = torch.cat([Qrep, sub_response_ids_pad], dim=1)
                        data = [json.dumps({"house":house,"floor":floor,"plen": plen}).encode(),
                        tensor_to_bytes(merged_ids), tensor_to_bytes(sub_scores_std),tensor_to_bytes(sub_scores),
                        tensor_to_bytes(sub_diaphragm_delta_score_list),tensor_to_bytes(sub_dcr_delta_list),tensor_to_bytes(sub_error_num_list),
                        tensor_to_bytes(sub_invalid_process_num_list),tensor_to_bytes(sub_floor_price_list),tensor_to_bytes(sub_diaphragm_area_ratio_list),
                        tensor_to_bytes(sub_design_error_score_list),tensor_to_bytes(sub_design_invalid_process_score_list),tensor_to_bytes(sub_design_diaphragm_score_list),
                        tensor_to_bytes(sub_deisgn_dcr_score_list),tensor_to_bytes(sub_design_price_score_list)]
                        xdata = make_bytes_list(data)
                        r = requests.post(f"{ref_server_url}/upload", data=xdata)
                        if r.content == b'string':
                            ref_server_ver = 'string'
            #time.sleep(10)

    #计算input_ids每个token的log_probability,用于PPO的ratio计算
    @staticmethod
    def _get_per_token_logps(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1) #log_probs:[T,V],input_ids:[T,1]
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1) #取出input_ids中每个token_id对应的log_prob
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    #使用 PPO clipping（标准）*advatage+KL penalty / 1*advantage+KL penalty
    def step(self, batch: Dict,curr_step:int,log_file:str,rank:int) -> torch.Tensor:
        engine = self.engine
        tokenizer = self.tokenizer

        house,floor=batch['house'],batch['floor']
        prompt_length = batch['plen']
        inputs = batch['inputs'].to(engine.device)
        advantages = batch['rewards_std'].to(engine.device) #[B,T]
        rewards=batch['rewards'].to(engine.device)

        diaphragm_delta_score_list=batch['diaphragm_delta_score_list'].to(engine.device)
        dcr_delta_list=batch['dcr_delta_list'].to(engine.device)
        error_num_list=batch['error_num_list'].to(engine.device)
        invalid_process_num_list=batch['invalid_process_num_list'].to(engine.device)
        floor_price_list=batch['floor_price_list'].to(engine.device)
        diaphragm_area_ratio=batch['diaphragm_area_ratio'].to(engine.device)

        design_error_score_list=batch['design_error_score_list'].to(engine.device)
        design_invalid_process_score_list=batch['design_invalid_process_score_list'].to(engine.device)
        design_diaphragm_score_list=batch['design_diaphragm_score_list'].to(engine.device)
        deisgn_dcr_score_list=batch['deisgn_dcr_score_list'].to(engine.device)
        design_price_score_list=batch['design_price_score_list'].to(engine.device)

        logits = engine(inputs).logits #logits代表[prompt|answer]每个token的概率
        logits = logits[:, :-1, :]  # (B, L-1, V)
        input_ids = inputs[:, 1:]   # (B, L-1)

        per_token_logps = self._get_per_token_logps(logits, input_ids)
        per_token_logps = per_token_logps[:, prompt_length - 1:]
        ref_per_token_logps = batch['refs'].to(per_token_logps.device)

        d = ref_per_token_logps - per_token_logps #d为ref模型的log π(a|s)-现有模型的log π(a|s)
        per_token_kl = torch.exp(d) - d - 1 #KL-divergence公式 DKL​(πref​∣∣πθ​)≈exp(d)−d−1

        completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int() #pad为0,有效值为1

        '''
        #检查对齐
        texts = tokenizer.batch_decode(inputs, skip_special_tokens=False)
        for i, t in enumerate(texts):
            if advantages[i].size(0)!=per_token_logps[i].size(0):
                print("=" * 40)
                print("发生不匹配！")
                print(f"[sample {i}]")
                print(t)
                print("input_len:", inputs[i].size(0))
                print("logp_len:", per_token_logps[i].size(0))
                print("adv_len:", advantages[i].size(0))
                print("rewards_len:",rewards[i].size(0))
                print("mask_len:", completion_mask[i].size(0))
        '''

        per_token_obj = torch.exp(per_token_logps - per_token_logps.detach()) * advantages #否则简化ratio为1,相当于logp_new*advantage
        assert self.compute_gen_logps is False

        per_token_loss = -(per_token_obj - self.beta * per_token_kl) #Lt​=−(rt​At​​​−βDKL​​​)
        
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean() #对每个trajectory先求平均,再对batch内的所有trajectory求平均,序列长度对最终per_token_loss影响相同
        
        #保存每个step的advantage和kl散度
        if curr_step%2==0 and rank==0:
            with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"house":house,"floor":floor,"step":curr_step,"rewards":rewards.detach().cpu().tolist(),
                                        "advantage":advantages.detach().cpu().tolist(),

                                        "diaphragm_delta_score_list":diaphragm_delta_score_list.detach().cpu().tolist(),
                                        "dcr_delta_list":dcr_delta_list.detach().cpu().tolist(),
                                        "error_num_list":error_num_list.detach().cpu().tolist(),
                                        "invalid_process_num_list":invalid_process_num_list.detach().cpu().tolist(),
                                        "floor_price_list":floor_price_list.detach().cpu().tolist(),
                                        "diaphragm_area_ratio":diaphragm_area_ratio.detach().cpu().tolist(),

                                        "design_error_score_list":design_error_score_list.detach().cpu().tolist(),
                                        "design_invalid_process_score_list":design_invalid_process_score_list.detach().cpu().tolist(),
                                        "design_diaphragm_score_list":design_diaphragm_score_list.detach().cpu().tolist(),
                                        "deisgn_dcr_score_list":deisgn_dcr_score_list.detach().cpu().tolist(),
                                        "design_price_score_list":design_price_score_list.detach().cpu().tolist(),

                                        "per_token_kl":per_token_kl.detach().cpu().tolist(),
                                        "per_token_obj":per_token_obj.detach().cpu().tolist(),
                                        "per_token_loss":per_token_loss.detach().cpu().tolist()}
                                        ,ensure_ascii=False) + "\n")
        return loss