import json
from peft import LoraConfig, get_peft_model
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForCausalLM, TrainerCallback
import gc
import numpy as np
import os

# ------------------------
# 配置
# ------------------------
# Set HF_TOKEN in the environment before running this script.
MODEL_ID   = "baidu/ERNIE-4.5-0.3B-PT"
#MODEL_ID = "google/gemma-3-1b-it"
OUTPUT_DIR = "five_plus_two_optimization/train_model_new/train_base_model/TRAIN_RESULTS/ernie_base_model_11_1"
TRAIN_JSONL = "train_json_data/five_plus_two_train_jsonl_data/design_3.27/post_train_auged_sft_data/sft_data_5_1000step/area_auged/90017.jsonl"
TEST_JSONL  = "train_json_data/five_plus_two_train_jsonl_data/design_3.27/base_model_train/test_set_100_actionized_sort_floor_force_auged(3_times).jsonl"

BATCH_SIZE, GRAD_ACCUM = 1,2
LEARNING_RATE = 5e-5
EPOCH = 1
LORA_R, LORA_ALPHA = 8, 8
MAX_LENGTH = 10000
TARGET_MODULES = [ #进行LoRA adapter的模块
    "q_proj", #self-Attention Query
    "v_proj", #self-Attention Value
    "k_proj", #self-Attention Key
    "o_proj", #self-Attention attention输出线性变换
    "gate_proj", #FFN前半部分门控部分
    "up_proj", #FFN前半部分扩张变换部分
    "down_proj", #FFN后半部分降维部分
]
TRAIN_NUM, TEST_NUM = 14651, 3295

# ── 正交约束配置 ────────────────────────────────────────────────────────────────
# 各层的正交惩罚权重（key = 0-indexed 层编号，对应 hidden_states[layer_idx+1]）
# L1-L5：强约束（1.0 → 0.6），L6-L9：弱约束（0.4 → 0.1），L10+：不约束
LAYER_WEIGHTS = {
    0: 1.0,   # L1
    1: 0.9,   # L2
    2: 0.8,   # L3
    3: 0.7,   # L4
    4: 0.6,   # L5
    5: 0.5,   # L6
    6: 0.4,   # L7
    7: 0.3,   # L8
    8: 0.2,   # L9
    9: 0.1    # L10
}
LAMBDA_ORTH = 0.005

# ------------------------
# 工具函数
# ------------------------
def initialize_token_embedding(model, tokenizer):
    existing_tokens = ["wall", "exterior_wall", "opening", "boundary",
                       "beam", "shearwall", "post", "lineload", "add", "remove"]
    for i in range(256):
        existing_tokens.append(f'{i}')
    existing_embs = []
    for token in existing_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        emb = model.get_input_embeddings().weight.data[ids].mean(dim=0)
        existing_embs.append(emb)

    new_tokens = ['<wall>', '<exterior_wall>', '<opening>', '<inoutbox>',
                  '<beam>', '<shearwall>', '<POST>', '<LINELOAD>', '<add>', '<remove>']
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    model.resize_token_embeddings(len(tokenizer))

    new_ids = [tokenizer.convert_tokens_to_ids(t) for t in new_tokens]
    for i, new_id in enumerate(new_ids):
        with torch.no_grad():
            model.get_input_embeddings().weight[new_id] = existing_embs[i]

def chars_to_token_range(offsets: list, start_char: int, end_char: int) -> tuple:
    start_tok, end_tok = None, None
    for i, (c_s, c_e) in enumerate(offsets):
        if c_s == 0 and c_e == 0:   # BOS / EOS special token
            continue
        if start_tok is None and c_e > start_char:
            start_tok = i
        if c_s < end_char:
            end_tok = i + 1
    return (start_tok or 0, end_tok or len(offsets))

def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("Memory cleaned.")

# ------------------------
# 自定义数据 Collator
# ------------------------

class CustomDataCollator(DataCollatorForSeq2Seq):
    """
    在 DataCollatorForSeq2Seq 基础上，额外处理 section 范围字段。
    这些整数字段不需要 padding，直接转成 tensor 附在 batch 上。
    """
    RANGE_KEYS = ("ctx_start", "ctx_end", "upper_start", "upper_end", "has_force")

    def __call__(self, features):
        # 先把自定义字段取出，避免传入父类的 padding 逻辑
        range_vals = {k: [f.pop(k, 0) for f in features] for k in self.RANGE_KEYS}
        batch = super().__call__(features)
        for k, vals in range_vals.items():
            batch[k] = torch.tensor(vals, dtype=torch.long)
        return batch

# ------------------------
# 自定义 Trainer
# ------------------------
class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 训练阶段：在两次 log 之间累积分项 loss，写入时取均值
        self._sft_accum   = 0.0
        self._orth_accum  = 0.0
        self._accum_n     = 0
        # eval 阶段：跨 batch 累积，evaluate() 结束后写入 log_history
        self._eval_sft_accum  = 0.0
        self._eval_orth_accum = 0.0
        self._eval_n          = 0

    # ── 正交损失计算 ─────────────────────────────────────────────────────────────
    def _compute_orth_loss(
        self,
        hidden_states: tuple,
        ctx_starts: torch.Tensor,
        ctx_ends: torch.Tensor,
        upper_starts: torch.Tensor,
        upper_ends: torch.Tensor,
        has_forces: torch.Tensor,
    ) -> torch.Tensor:
        """
        对 batch 中每个含受力信息的样本，计算各约束层的正交损失，
        再按层权重加权求和，最终取 batch 均值。
        """
        device = ctx_starts.device
        total_orth = torch.tensor(0.0, device=device)
        valid_count = 0

        for b in range(has_forces.shape[0]): #遍历batch中的每一个样本
            if has_forces[b].item() == 0:
                continue

            ctx_s = ctx_starts[b].item()
            ctx_e = ctx_ends[b].item()
            up_s  = upper_starts[b].item()
            up_e  = upper_ends[b].item()

            # 区间合法性检查（含越界守卫：防止 prompt 被截断导致范围超出 seq_len）
            # 新格式顺序：上层受力 → context → structures，故要求 up_e <= ctx_s
            seq_len = hidden_states[1][b].shape[0]
            if ctx_e <= ctx_s or up_e <= up_s or ctx_s < up_e:
                continue
            if ctx_s >= seq_len or up_e > seq_len:
                continue

            sample_orth = torch.tensor(0.0, device=device)
            for layer_idx, weight in LAYER_WEIGHTS.items():
                # hidden_states[0] 是 embedding 层，[1:] 才是各 transformer 层
                h = hidden_states[layer_idx + 1][b]   # (seq_len, d_model)，保留梯度

                h1 = h[ctx_s:ctx_e].mean(dim=0)       # context 区域均值
                h2 = h[up_s:up_e].mean(dim=0)         # upper layer 区域均值

                h1_norm = h1 / (h1.norm() + 1e-8) #使两向量模长均为1，则最终乘积的大小只与方向有关
                h2_norm = h2 / (h2.norm() + 1e-8)

                # 余弦相似度平方作为惩罚（越高越被惩罚）
                cos_sim = (h1_norm * h2_norm).sum()
                sample_orth = sample_orth + weight * cos_sim.pow(2)

            total_orth = total_orth + sample_orth
            valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=device)
        return total_orth / valid_count

    # ── 损失计算入口 ──────────────────────────────────────────────────────────────
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        ctx_starts   = inputs.pop("ctx_start",   None)
        ctx_ends     = inputs.pop("ctx_end",     None)
        upper_starts = inputs.pop("upper_start", None)
        upper_ends   = inputs.pop("upper_end",   None)
        has_forces   = inputs.pop("has_force",   None)

        if not hasattr(self, "_hf_diag_count"):
            self._hf_diag_count = 0
        if self._hf_diag_count < 5:
            self._hf_diag_count += 1
            print(f"\n[DIAG2 #{self._hf_diag_count}] has_forces={has_forces} "
                  f"ctx_start={ctx_starts} upper_start={upper_starts}")

        # 只在 batch 含受力数据时才存储隐层，无受力 batch 不产生额外显存开销
        has_force_batch = has_forces is not None and has_forces.sum().item() > 0
        outputs  = model(**inputs, output_hidden_states=has_force_batch)
        sft_loss = outputs.loss

        if has_force_batch:
            if outputs.hidden_states is None:
                print(f"\n[DIAG] has_force_batch=True but hidden_states=None (model ignored flag)")
                orth_loss = torch.tensor(0.0, device=sft_loss.device)
            else:
                if not hasattr(self, "_diag_printed"):
                    self._diag_printed = True
                    b = 0
                    cs = ctx_starts[b].item(); ce = ctx_ends[b].item()
                    us = upper_starts[b].item(); ue = upper_ends[b].item()
                    h = outputs.hidden_states[1][b].float()
                    h1 = h[cs:ce].mean(dim=0); h2 = h[us:ue].mean(dim=0)
                    h1n = h1/(h1.norm()+1e-8); h2n = h2/(h2.norm()+1e-8)
                    cos = (h1n*h2n).sum().item()
                    print(f"\n[DIAG] first force batch: ctx={cs}:{ce} upper={us}:{ue} "
                          f"n_layers={len(outputs.hidden_states)} "
                          f"cos_sim(layer1)={cos:.6f}")
                orth_loss = self._compute_orth_loss(
                    outputs.hidden_states,
                    ctx_starts, ctx_ends,
                    upper_starts, upper_ends,
                    has_forces,
                )
        else:
            orth_loss = torch.tensor(0.0, device=sft_loss.device)

        total_loss = sft_loss + LAMBDA_ORTH * orth_loss

        weighted_orth = LAMBDA_ORTH * orth_loss
        if model.training:
            self._sft_accum  += sft_loss.item()
            self._orth_accum += weighted_orth.item()
            self._accum_n    += 1
            if self.state.global_step % 100 == 0:
                print(f"\n[Step {self.state.global_step}] "
                      f"sft={sft_loss.item():.4f}  "
                      f"orth={weighted_orth.item():.4f}  "
                      f"total={total_loss.item():.4f}")
        else:
            self._eval_sft_accum  += sft_loss.item()
            self._eval_orth_accum += weighted_orth.item()
            self._eval_n          += 1

        return (total_loss, outputs) if return_outputs else total_loss

    # ── 训练日志：把累积的分项 loss 注入同一条 log_history 记录 ─────────────────────
    def log(self, logs, *args, **kwargs):
        if "loss" in logs and self._accum_n > 0:
            logs = {
                **logs,
                "sft_loss":  round(self._sft_accum  / self._accum_n, 6),
                "orth_loss": round(self._orth_accum / self._accum_n, 6),
            }
            self._sft_accum = self._orth_accum = 0.0
            self._accum_n   = 0
        super().log(logs, *args, **kwargs)

    # ── eval 结束：把累积均值追加到最后一条 log_history 记录 ──────────────────────
    def evaluate(self, *args, **kwargs):
        self._eval_sft_accum = self._eval_orth_accum = 0.0
        self._eval_n = 0
        metrics = super().evaluate(*args, **kwargs)
        if self._eval_n > 0 and self.state.log_history:
            self.state.log_history[-1].update({
                "eval_sft_loss":  round(self._eval_sft_accum  / self._eval_n, 6),
                "eval_orth_loss": round(self._eval_orth_accum / self._eval_n, 6),
            })
        return metrics

    # ── Embedding 保存（与原版相同）────────────────────────────────────────────────
    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)
        try:
            emb = self.model.base_model.model.model.embed_tokens.weight.data.detach().cpu()
        except Exception:
            emb = self.model.get_input_embeddings().weight.data.detach().cpu()
        emb_path = os.path.join(output_dir, "embedding.safetensors")
        torch.save(emb, emb_path)
        print(f"\n[Embedding Saved] -> {emb_path}\n")
        if hasattr(self, "tokenizer") and self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)

# ------------------------
# Callback（与原版相同）
# ------------------------

class SaveLossCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        loss_logs = [e for e in state.log_history if "loss" in e or "eval_loss" in e]
        with open(os.path.join(ckpt_dir, "loss_log.json"), "w", encoding="utf8") as f:
            json.dump(loss_logs, f, ensure_ascii=False, indent=2)
        print(f"Saved loss log at {ckpt_dir}/loss_log.json")
        return control

# ------------------------
# 主流程
# ------------------------

if __name__ == "__main__":
    clean_memory()

    # ── 模型与 tokenizer ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        use_cache=False,
    )
    initialize_token_embedding(model, tokenizer)

    # ── 数据预处理 ────────────────────────────────────────────────────────────────
    def preprocess_function(example):
        prompt   = example["prompt"]
        response = example["response"]

        full_text  = f"{prompt}{response}"
        encodings  = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, return_tensors=None)
        prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH)["input_ids"]
        labels     = [-100] * len(prompt_ids) + encodings["input_ids"][len(prompt_ids):]

        # ── 定位两类信息的 token 范围 ──────────────────────────────────────────
        # 直接检测 prompt 中是否含有受力关键字，比依赖 metadata 字段更可靠
        has_force = int("<POST>" in prompt or "<LINELOAD>" in prompt)
        ctx_start = ctx_end = upper_start = upper_end = 0

        if has_force:
            # 新格式顺序：上层受力（lineload/post）→ context → structures
            upper_post_char     = prompt.find("upper layer post:")
            upper_lineload_char = prompt.find("upper layer lineload:")
            candidates          = [c for c in [upper_post_char, upper_lineload_char] if c != -1]
            upper_start_char    = min(candidates) if candidates else -1
            ctx_char            = prompt.find("context:")

            if upper_start_char != -1 and ctx_char != -1:
                enc_off = tokenizer(
                    prompt,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                )
                offsets = enc_off["offset_mapping"]

                # context+structures 段：context: 内容 → </s> 之前（不含结尾 EOS）
                eos_char         = prompt.rfind("</s>")
                ctx_end_char     = eos_char

                ctx_start, ctx_end = chars_to_token_range(offsets, ctx_char, ctx_end_char)
                # 受力段：upper_end 直接取 ctx_start 的前一个 token，避免边界 token 污染
                upper_start, _ = chars_to_token_range(offsets, upper_start_char, ctx_char)
                upper_end      = ctx_start
                #print(f"upper:{prompt[upper_start_char:ctx_char]},ctx:{prompt[ctx_char:ctx_end_char]}")
        return {
            "input_ids":      encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels":         labels,
            "ctx_start":      ctx_start,
            "ctx_end":        ctx_end,
            "upper_start":    upper_start,
            "upper_end":      upper_end,
            "has_force":      has_force,
        }

    print("Loading dataset...")
    train_data_list = []
    with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_data_list.append(json.loads(line))
    train_dataset = Dataset.from_list(train_data_list)
    train_dataset = train_dataset.map(
        preprocess_function, batched=False,
        remove_columns=train_dataset.column_names,
    )
    _n_force = sum(train_dataset["has_force"])
    print(f"[DIAG3] train_dataset has_force sum={_n_force}/{len(train_dataset)}")
    print(f"[DIAG3] first 10 has_force: {train_dataset['has_force'][:10]}")

    test_data_list = []
    test_sample_point = np.linspace(0, TEST_NUM - 1, 10, dtype=int)
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i not in test_sample_point:
                continue
            if line.strip():
                test_data_list.append(json.loads(line))
    test_dataset = Dataset.from_list(test_data_list)
    test_dataset = test_dataset.map(
        preprocess_function, batched=False,
        remove_columns=test_dataset.column_names,
    )
    print(f"Train size: {len(train_dataset)}  Test size: {len(test_dataset)}")

    # ── LoRA ─────────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    for name, p in model.named_parameters():
        if "embed_tokens" in name or "lora" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # ── 训练参数（与原版保持一致）────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        max_steps=-1,
        num_train_epochs=EPOCH,
        logging_steps=2,
        prediction_loss_only=True,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=20,
        per_device_eval_batch_size=1,
        save_steps=1000,
        save_total_limit=50,
        bf16=torch.cuda.is_available(),
        fp16=False,
        optim="adamw_torch",
        report_to="none",
        lr_scheduler_type="constant",
        remove_unused_columns=False,
    )

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=CustomDataCollator(tokenizer=tokenizer, padding=True),
        callbacks=[SaveLossCallback()],
    )
    trainer.train()

    loss_dir = f"{OUTPUT_DIR}/loss_history.json"
    with open(loss_dir, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"Done! Model saved to {OUTPUT_DIR}")

    trainer.model = trainer.model.merge_and_unload()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)