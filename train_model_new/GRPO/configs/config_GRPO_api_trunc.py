from dataclasses import dataclass

@dataclass
class Config:
    #model_path: str = "/home/jiuxing_li/five_plus_two_optimization/train_model_new/GRPO/base_model/ernie_base_model_10" #var
    model_path:str = "/home/jiuxing_li/five_plus_two_optimization/train_model_new/GRPO/save_model/grpo_fpt_api/grpo_fpt_api_4/step_4000"
    gen_device: int = 0 #var
    #ref_server: str = "http://127.0.0.1:59875"
    ref_server: str = "http://127.0.0.1:59874"

    all_steps: int = 20000 #var
    Q_batch_size: int = 3 #var
    num_pre_Q: int = 40
    train_batch_size: int = 1
    gen_update_steps: int = 24 #var
    save_steps: int = 1000 #var

    beta: float = 0.04
    compute_gen_logps: bool = False
    clip_param: float = 0.2

def make_ds_config(cfg: "Config") -> dict:
    return {
        "train_micro_batch_size_per_gpu": cfg.train_batch_size,
        "gradient_accumulation_steps": 40, #var
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-6}}, #var
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            "stage3_gather_16bit_weights_on_model_save": True,
            "offload_optimizer": {"device": "cpu"}
        },
        "lr":1e-5, #var
    }