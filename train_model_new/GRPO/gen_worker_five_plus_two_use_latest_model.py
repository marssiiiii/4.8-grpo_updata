import os, json, re, random, time, importlib, argparse, requests
import torch
import torch.multiprocessing as mp
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def main():
    parser = argparse.ArgumentParser() #在调用过程读取超参数，指定模型和rank
    parser.add_argument('--algo', type=str, default='GRPO_api_trunc_use_latest_model', help="choose algorithms") #选择模型
    parser.add_argument('--local_rank', type=int, default=0) #var

    args, _ = parser.parse_known_args()
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)

    algo_mod = importlib.import_module(f"algorithm.{args.algo}") #导入模型
    cfg_mod  = importlib.import_module(f"configs.config_{args.algo}") #导入模型训练参数
    Config = cfg_mod.Config
    AlgoClass = algo_mod.Algorithm
    cfg = Config()

    print('\nSTART vLLM generation...\n')
    mp.set_start_method('spawn', force=True) #使用全新python解释器，不继承父进程的CUDA context
    Q = mp.Queue() #创建跨进程通信队列Q

    #得到gen_worker
    gen_worker = getattr(AlgoClass, "gen_worker", None)
    if gen_worker is None:
        raise RuntimeError(f"Algorithm '{args.algo}' must provide a gen_worker(Q, cfg) function.")
    else:
        print("识别到了gen_worker")

    gen_worker(
        Q=Q,
        model_path=cfg.model_path,
        gen_device=0, #var
        ref_server_url=cfg.ref_server,
        num_pre_Q=cfg.num_pre_Q,
        train_batch_size=cfg.train_batch_size,
        compute_gen_logps=cfg.compute_gen_logps,
        Q_batch_size=cfg.Q_batch_size,
        TRAIN_JSONL="train_json_data/five_plus_two_train_jsonl_data/design_3.27/grpo_phase_1_train_data/train_set_inbox_force_prompt.jsonl", #var
        TRAIN_BEG=0, TRAIN_END=13613, #var
        #TRAIN_BEG=0, TRAIN_END=1, #var
    )

if __name__ == '__main__':
    main()