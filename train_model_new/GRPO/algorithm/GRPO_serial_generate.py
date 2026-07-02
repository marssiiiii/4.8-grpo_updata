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
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/train_model_new/base_code")

sys.path.append("/home/jiuxing_li")
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/train_model_new/GRPO/test/test_code")
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/five_plus_two_data_embedding_3_11")
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/five_plus_two_test")
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/train_model_new/GRPO/test/test_code")

from reward_design_trunc import get_design_score,get_score_from_cmp_and_response
from model_test_new_token_five_plus_two import construct_prompt,process_completion_predict_filt,process_completion_predict_based_on_response

os.environ["DS_SKIP_CUDA_CHECK"] = "1"

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
                num_pre_Q: int,serial_len: int, train_batch_size: int, compute_gen_logps: bool,
                  Q_batch_size: int,TRAIN_JSONL:str,TRAIN_BEG:int,TRAIN_END:int):  # ✅ 新增参数
        GPU_UTILIZATION=0.5

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

        sampling_params = SamplingParams(n=1, temperature=0.8, max_tokens=200) #生成问题答案的参数(生成答案数量,多样性,token数限制) #var

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
                        "context": data["context"],
                        "completion_predict": data["completion_predict"],
                        "prompt": data['prompt']
                    })

        # ✅ 新增：模型版本追踪
        current_model_version = 0
        last_check_time = time.time()

        #一句prompt返回答案以及答案token_id(回答的多样性由sampling_params控制)
        def gen_answers(context_list,completion_predict_list):
            #print(f"gen_answers检验，初始 context_list:{context_list}, 初始 completion_predict_list:{completion_predict_list}") #检验输入
            sample_num = int(num_pre_Q / serial_len) 
            final_context_list,final_cmp_list,final_prompt_list,final_response_list = [],[],[],[]
            for context,completion_predict in zip(context_list, completion_predict_list): #遍历初始房屋结构
                initial_completion_predict_list=[] #初始化round_completion_predict_list
                for _ in range(sample_num): #初始化10条初始路径
                    initial_completion_predict_list.append(process_completion_predict_filt(context=context,
                                    completion_predict=completion_predict,response=""))
                    
                for initial_completion_predict in initial_completion_predict_list: #共生成num_pre_Q/serial_len个路径
                    round_completion_predict=initial_completion_predict
                    for _ in range(serial_len): #每个路径长度为serial_len
                        prompt=construct_prompt(context=context,completion_predict=round_completion_predict)#构造prompt并生成response
                        tip_text = []
                        tip_text.append(prompt)
                        voutputs = vllm_gen.generate(tip_text, sampling_params, use_tqdm=False)
                        for v in voutputs: #将所有生成的answer都打平,返回answer+answer_token_id
                            for z in v.outputs:
                                response=tokenizer.decode(z.token_ids)
                                final_context_list.append(context)
                                final_cmp_list.append(round_completion_predict)
                                final_prompt_list.append(prompt)
                                final_response_list.append(response)
                                response_processed=process_completion_predict_filt(context=context,
                                        completion_predict=round_completion_predict,response=response)
                                #print(f"response_processed:{response_processed}")
                                round_completion_predict=response_processed #将当前轮的回答作为下一轮的输入,继续生成
            return final_context_list,final_cmp_list,final_prompt_list,final_response_list

        # 奖励函数分两部分：正确性奖励+格式奖励
        # 正确性奖励:当前的逻辑是依据无支撑次数和房间数误差进行打分
        def reward_correct(context_list,completion_predict_list,response_list,house,floor):
            batch_score_list,is_improved=[],[]
            for i in range(len(context_list)):
                context,completion_predict,response=\
                    context_list[i],completion_predict_list[i],response_list[i]
                score,delta,invalid_process_num,illegal_num=get_score_from_cmp_and_response(context=context,
                                                completion_predict=completion_predict,response=response,tokenizer=tokenizer)
                if delta>0:
                    is_improved.append(1)
                else:
                    is_improved.append(0)
                batch_score_list.append(score)

                if i==0:
                    print(f"house:{house},floor:{floor},response:{response},delta:{delta},score:{score},invalid_process_num:{invalid_process_num},illegal_num:{illegal_num}") #打印第一条数据的结果以供检验
            print(f"house:{house},floor:{floor},is_improved:{is_improved}")
            return batch_score_list,is_improved

        #对每个问题生成多个答案，对每个候选答案存储提问问题(含答案),奖励,答案,答案token_id
        def gen_samples(inputs):
            context_list,completion_predict_list = [x["context"] for x in inputs],[x["completion_predict"] for x in inputs]
            house_list,floor_list=[x["house"] for x in inputs],[x["floor"] for x in inputs]
            context_list,completion_predict_list,prompt_list,response_list = gen_answers(context_list=context_list,completion_predict_list=completion_predict_list) #依据inputs提问生成答案
            response_encoded_list=[tokenizer(response, return_offsets_mapping=False) for response in response_list]
            score_list,is_improved = [],[] #依据答案生成reward,标准的prompt_ids
            for i, inp in enumerate(inputs):
                sample_context_list,sample_completion_predict_list,sample_response_list=\
                    context_list[i*num_pre_Q:(i+1)*num_pre_Q],completion_predict_list[i*num_pre_Q:(i+1)*num_pre_Q],\
                    response_list[i*num_pre_Q:(i+1)*num_pre_Q]
                
                batch_score_list,batch_is_improved=\
                    reward_correct(context_list=sample_context_list,completion_predict_list=sample_completion_predict_list
                    ,response_list=sample_response_list,house=inp["house"],floor=inp["floor"])
                
                score_list=score_list+batch_score_list
                is_improved=is_improved+batch_is_improved

            score_list = [
                torch.tensor(s, dtype=torch.float32)
                for s in score_list
            ]

            return prompt_list,response_list,response_encoded_list,score_list,is_improved,house_list,floor_list
        
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
            prompt_list,response_list,response_encoded_list,score_list,is_improved,house_list,floor_list = gen_samples(inputs) #依据采样问题得到采样答案
            #print(f"gen_sample检验：prompt_list{len(prompt_list),prompt_list[:2]}, response_list:{len(response_list)}, score_list:{len(score_list)}, is_improved:{len(is_improved)},\
            #      house_list:{house_list},floor_list:{floor_list}")
            #print(f'time: {time.time()-tic:.2f}s    ', 'rewards:', rewards)

            # advantage standardization
            NUM_GENERATE=0
            MODEL_KEEP_UPDATED=True
            while NUM_GENERATE<Q_batch_size:
                i=NUM_GENERATE
                print(f"当前是{i+1}/{Q_batch_size}")
                pp=prompt_list[i]
                prompt_ids = tokenizer(pp, return_tensors="pt")["input_ids"]
                plen = prompt_ids.shape[1]
                house,floor=house_list[i],floor_list[i]
                curr_response_list = response_list[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_scores = score_list[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_is_improved=is_improved[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_response_encoded_list = response_encoded_list[i*num_pre_Q:(i+1)*num_pre_Q]
                #求得curr_rewards进行标准化
                mean_s,std_s = np.mean(curr_scores),np.std(curr_scores)
                if np.sum(curr_is_improved)<5:
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
                        #sub_scores,sub_scores_std=pad_sequence(sub_scores, batch_first=True, padding_value=tokenizer.pad_token_id),\
                        #pad_sequence(sub_scores_std, batch_first=True, padding_value=tokenizer.pad_token_id)
                        sub_scores = torch.tensor(sub_scores, dtype=torch.float32) if len(sub_scores) > 0 else torch.tensor([], dtype=torch.float32)
                        sub_scores_std = torch.tensor(sub_scores_std, dtype=torch.float32) if len(sub_scores_std) > 0 else torch.tensor([], dtype=torch.float32)

                        sub_response_encoded_batch = curr_response_encoded_list[ii: ii + train_batch_size]
                        if len(sub_response_encoded_batch)==0:
                            print("当前子批次没有数据，跳过上传")
                            continue
                        sub_response_ids = [x["input_ids"] for x in sub_response_encoded_batch]
                        sub_response_ids_tensor = [torch.tensor(lst) for lst in sub_response_ids]
                        sub_response_ids_pad = pad_sequence(sub_response_ids_tensor, batch_first=True, padding_value=tokenizer.pad_token_id)


                        Qrep = prompt_ids.repeat(1, sub_response_ids_pad.shape[0]).view(-1, plen)
                        merged_ids = torch.cat([Qrep, sub_response_ids_pad], dim=1)
                        data = [json.dumps({"house":house,"floor":floor,"plen": plen}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(sub_scores_std),tensor_to_bytes(sub_scores)]
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
        if curr_step%10==0 and rank==0:
            with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"house":house,"floor":floor,"step":curr_step,"rewards":rewards.detach().cpu().tolist(),
                                        "per_token_kl":per_token_kl.detach().cpu().tolist(),"advantage":advantages.detach().cpu().tolist(),
                                        "per_token_obj":per_token_obj.detach().cpu().tolist(),"per_token_loss":per_token_loss.detach().cpu().tolist()}
                                        , ensure_ascii=False) + "\n")
        return loss