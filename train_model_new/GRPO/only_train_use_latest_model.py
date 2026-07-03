# -*- coding: utf-8 -*-
# deepspeed --include localhost:0 only_train_use_latest_model.py
import os, json, re, random, time, importlib, argparse, requests
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
import numpy as np
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ["DS_SKIP_CUDA_CHECK"] = "1"
from ref_client_five_plus_two import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

def get_batch(ref_server_url: str,device):
    try:
        r = requests.get(f"{ref_server_url}/get").content
        if r == b'empty':
            return None
    except Exception:
        return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0])
    data['inputs'] = bytes_to_tensor(dd[1],device)
    data['rewards_std'] = bytes_to_tensor(dd[2],device)
    data['refs'] = bytes_to_tensor(dd[3],device)
    data['rewards']=bytes_to_tensor(dd[4],device)

    data['diaphragm_delta_score_list']=bytes_to_tensor(dd[5],device)
    data['dcr_delta_list']=bytes_to_tensor(dd[6],device)
    data['error_num_list']=bytes_to_tensor(dd[7],device)
    data['invalid_process_num_list']=bytes_to_tensor(dd[8],device)
    data['floor_price_list']=bytes_to_tensor(dd[9],device)
    data['diaphragm_area_ratio']=bytes_to_tensor(dd[10],device)

    data['design_error_score_list']=bytes_to_tensor(dd[11],device)
    data['design_invalid_process_score_list']=bytes_to_tensor(dd[12],device)
    data['design_diaphragm_score_list']=bytes_to_tensor(dd[13],device)
    data['deisgn_dcr_score_list']=bytes_to_tensor(dd[14],device)
    data['design_price_score_list']=bytes_to_tensor(dd[15],device)
    data['latest_refs']=bytes_to_tensor(dd[16],device)

    return data

def main():
    # ✅ 新增：模型版本计数器
    model_version = 0

    parser = argparse.ArgumentParser() #在调用过程读取超参数，指定模型和rank
    parser.add_argument('--algo', type=str, default='GRPO_api_trunc_use_latest_model', help="choose algorithms") #选择模型
    parser.add_argument('--local_rank', type=int, default=-1)

    args, _ = parser.parse_known_args()
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)

    algo_mod = importlib.import_module(f"algorithm.{args.algo}") #导入模型
    cfg_mod  = importlib.import_module(f"configs.config_{args.algo}") #导入模型训练参数
    Config = cfg_mod.Config
    make_ds_config = cfg_mod.make_ds_config
    AlgoClass = algo_mod.Algorithm  
    cfg = Config()

    deepspeed.init_distributed() #deepspeed提供并行多卡训练环境
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main = local_rank == 0 #只在rank=0的主进程上记录训练过程
    if is_main==False:
        os.environ["WANDB_MODE"] = "disabled"
    
    if is_main==True:
        import wandb
        wandb.init(
            #mode="online",          
            project="awesome-grpo",
            entity="buqi",
            name=f"{args.algo}-train",
            group=f"{args.algo}",
            config={"algo": args.algo}
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    #tokenizer = AutoTokenizer.from_pretrained("/mnt/efs/jiuxing_li/ERNIE_train_results/text_predict/generate_lora_9(5e-5,new_data)")

    print("加载裸模型")
    model = AutoModelForCausalLM.from_pretrained( #加载engin，定义algo类(grpo engine实例)
        cfg.model_path, torch_dtype=torch.bfloat16
    )
    print("裸模型加载成功")
    print("加载engine")
    ds_config = make_ds_config(cfg)
    engine, optimizer, _, _ = deepspeed.initialize(
        config=ds_config, model=model, model_parameters=model.parameters()
    )
    print("engine加载成功")

    print("加载Algo类")
    algo_kwargs = {
        "beta": cfg.beta,
        "clip_param": cfg.clip_param,
        "compute_gen_logps": cfg.compute_gen_logps,
    }
    algo = AlgoClass(engine=engine, tokenizer=tokenizer, **algo_kwargs) #初始化强化学习对象
    print("Algo类加载成功")

    progress = range(1, cfg.all_steps + 1) #进行训练
    if dist.get_rank() == 0: 
        progress = tqdm(progress)
    print("训练开始")

    output_file="/home/jiuxing_li/five_plus_two_optimization/train_model_new/GRPO/save_model/grpo_fpt_api/grpo_fpt_api_5" #var
    
    for step in progress:#训练all_steps步
        batch = get_batch(cfg.ref_server,engine.device) #从ref_server中拉去batch
        while batch is None:
            print('waiting for batch...')
            time.sleep(5)
            batch = get_batch(cfg.ref_server,engine.device)
        
        log_file = f"{output_file}/loss_history.jsonl"
        loss = algo.step(batch,step,log_file,dist.get_rank())
        if is_main and step%10==0: #保存loss
            loss_val = loss.detach().cpu().item()
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"step":step,"loss_val":loss_val}
                                    , ensure_ascii=False) + "\n")
        engine.backward(loss)
        engine.step()

        if dist.get_rank() == 0:
            progress.set_description(f"{args.algo.upper()} | Loss: {loss.item():.6f}")
            if is_main:
                loss_val = float(loss.detach().cpu())
                wandb.log({"train/loss": loss_val}, step=step)
        
        # ✅ 新增：定期更新gen_worker的模型
        if step % cfg.gen_update_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                # 查询最新模型版本
                response = requests.get(f"{cfg.ref_server}/model_version", timeout=5)
                version_info = response.json()
                gen_worker_is_updating=version_info["gen_worker_is_updating"]
                print(f"train_version_info:{version_info}")
                if gen_worker_is_updating==False: #检查gen_worker是否正在更新，如果gen_worker没有在更新，则train更新模型

                    print(f'[TRAINING PROC] Updating train model at step {step}')

                    # 1. 通知ref_client：开始更新（加锁）
                    lock_data = {'step': step, 'version': model_version}
                    response = requests.post(
                        f"{cfg.ref_server}/lock_train_update",
                        data=json.dumps(lock_data).encode(),
                        timeout=5
                    )
                    print(f"[TRAINING] Model update locked: {response.json()}")
                    
                    model_version += 1
                    update_path = f"{output_file}/latest_model"
                    os.makedirs(update_path, exist_ok=True)

                    # 2. 保存模型到共享路径
                    state_dict = engine.module.state_dict()
                    state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                    engine.module.save_pretrained(update_path, state_dict=state_dict)
                    tokenizer.save_pretrained(update_path)
                    
                    # 3. 通过ref_client通知gen_worker
                    try:
                        notification = {
                            'version': model_version,
                            'path': update_path,
                            'step': step
                        }
                        response = requests.post(
                            f"{cfg.ref_server}/notify_model",
                            data=json.dumps(notification).encode(),
                            timeout=5
                        )
                        print(f'[TRAINING PROC] Model notification sent: {response.json()}')
                    except Exception as e:
                        print(f'[TRAINING PROC] Failed to notify model update: {e}')
                    
                    # 4. 通知ref_client：更新完成（解锁）
                    unlock_data = {
                        'version': model_version,
                        'path': update_path,
                        'step': step
                    }
                    response = requests.post(
                        f"{cfg.ref_server}/unlock_train_update",
                        data=json.dumps(unlock_data).encode(),
                        timeout=5
                    )
                    print(f"[TRAINING] Model update unlocked: {response.json()}")
            
            dist.barrier()

        if step % cfg.save_steps == 0: #保存check_point
            dist.barrier() #更新check_point前进行进程同步，再更新
            if dist.get_rank() == 0:
                print('saving model')
                save_name = f"{output_file}/step_{step}"
                state_dict = engine.module.state_dict()
                state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                engine.module.save_pretrained(save_name, state_dict=state_dict)
                tokenizer.save_pretrained(save_name)
            dist.barrier()
    
    # ✅ 修复：先让所有进程同步，再清理vLLM
    print(f'[Rank {dist.get_rank()}] Training loop finished, waiting for sync...')
    dist.barrier()  # 所有进程都必须先到达这里

    # ✅ 最后再次同步，确保清理完成
    dist.barrier()

    if is_main:
        print("wandb已记录")
        wandb.finish()

if __name__ == '__main__':
    main()
