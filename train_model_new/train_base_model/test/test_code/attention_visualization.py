import json
import torch
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
_fm._load_fontmanager(try_read_cache=False)
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/train_model_new/train_base_model")
from ernie_base_model_train import initialize_token_embedding
from peft import PeftModel,LoraConfig

# ------------------------
# 配置
# ------------------------
MODEL_PATH = "five_plus_two_optimization/train_model_new/train_base_model/TRAIN_RESULTS/ernie_base_model_9/checkpoint-3000"
TEST_JSON  = "train_json_data/five_plus_two_train_jsonl_data/design_3.27/base_model_train/test_set_100_actionized_sort_floor_force_1.jsonl"
OUTPUT_DIR = "five_plus_two_optimization/train_model_new/train_base_model/test/attention_vis/ernie_9"
SAVE_ID = "model_9_3000step"

MAX_NEW_TOKENS = 500         # generate 最多生成多少 token
MAX_SEQ_TOKENS = 9999        # forward pass 总长度截断
SAMPLE_INDICES = [3, 8]  # 可视化哪几条样本
VIS_LAYERS     = None        # None = 全部层; 或 [0, 5, 11]
VIS_HEADS      = None        # None = 只画平均; 或 [0, 1] 同时画指定 head
GEN_TEMPERATURE = 0.7
GEN_DO_SAMPLE   = True

# 只关注 response 对这些 prompt 关键词的注意力
KEYWORDS = {
    '<wall>', '<exterior_wall>', '<opening>', '<inoutbox>',
    '<beam>', '<shearwall>', '<POST>', '<LINELOAD>', '<add>', '<remove>',
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_sample(data_json: str, sample_idx: int) -> dict:
    with open(data_json, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == sample_idx:
                return json.loads(line)
    raise IndexError(f"样本索引 {sample_idx} 超出范围")


# ── 生成 & 注意力提取 ──────────────────────────────────────────────────────────

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int) -> list[int]:
    """用 model.generate 生成 response，返回生成部分的 token id 列表（不含 prompt）"""
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device).input_ids
    with torch.no_grad():
        gen = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=GEN_DO_SAMPLE,
            temperature=GEN_TEMPERATURE,
        )
    prompt_len = input_ids.shape[1]
    return gen[0, prompt_len:].tolist()


def get_cross_attentions(model, tokenizer,
                         prompt: str, generated_ids: list[int],
                         max_tokens: int):
    """
    将 prompt + generated_response 拼接后做一次前向推断（output_attentions=True），
    返回 response→prompt 的注意力子矩阵。

    返回:
        cross_attentions : list[np.ndarray]  每层 (H, len_resp, len_prompt)
        prompt_tokens    : list[str]
        response_tokens  : list[str]
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids

    # 截断：优先保留 response，prompt 从末尾截取
    if len(prompt_ids) + len(generated_ids) > max_tokens:
        keep_resp   = min(len(generated_ids), max_tokens // 2)
        keep_prompt = max_tokens - keep_resp
        prompt_ids    = prompt_ids[-keep_prompt:]
        generated_ids = generated_ids[:keep_resp]

    len_prompt = len(prompt_ids)
    full_ids   = prompt_ids + generated_ids
    input_ids  = torch.tensor([full_ids]).to(model.device)

    with torch.no_grad():
        out = model(input_ids=input_ids, output_attentions=True)

    cross_attentions = []
    for a in out.attentions:
        a_np  = a.squeeze(0).cpu().float().numpy()   # (H, T, T)
        cross = a_np[:, len_prompt:, :len_prompt]    # (H, len_resp, len_prompt)
        cross_attentions.append(cross)

    prompt_tokens   = tokenizer.convert_ids_to_tokens(prompt_ids)
    response_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
    return cross_attentions, prompt_tokens, response_tokens


# ── 关键词过滤 ────────────────────────────────────────────────────────────────

def filter_keyword_columns(cross_attentions, prompt_tokens, keywords):
    """
    从 cross_attentions 中只保留 prompt_tokens 里属于 keywords 的列，
    同一关键词的多次出现聚合（求和）为单列。

    返回:
        kw_attentions : list[np.ndarray]  每层 (H, len_resp, num_unique_kw)
        kw_labels     : list[str]         每列的标签，如 "<wall>(×3)"
    """
    # 按关键词类型收集所有出现位置
    from collections import OrderedDict
    kw_occurrences: dict[str, list[int]] = OrderedDict()
    for kw in sorted(keywords):          # 固定顺序，方便对比不同样本
        kw_occurrences[kw] = []
    for i, t in enumerate(prompt_tokens):
        if t in keywords:
            kw_occurrences[t].append(i)

    # 只保留实际出现过的关键词
    present = [(kw, idxs) for kw, idxs in kw_occurrences.items() if idxs]
    if not present:
        raise ValueError("prompt 中未找到任何关键词 token，请检查 tokenizer 是否已添加这些特殊 token。")

    kw_labels = [
        f"{kw}(×{len(idxs)})" if len(idxs) > 1 else kw
        for kw, idxs in present
    ]

    # 对同一关键词的所有列求和，得到 (H, len_resp, num_unique_kw)
    agg_attentions = []
    for a in cross_attentions:           # a: (H, len_resp, len_prompt)
        cols = np.stack(
            [a[:, :, idxs].sum(axis=-1) for _, idxs in present],
            axis=-1
        )                                # (H, len_resp, num_unique_kw)
        agg_attentions.append(cols)

    return agg_attentions, kw_labels


# ── 绘图函数 ──────────────────────────────────────────────────────────────────

def _fig_size(x_tokens, response_tokens):
    w = max(6,  len(x_tokens)        * 0.45)
    h = max(4,  len(response_tokens) * 0.28)
    return w, h


def plot_single_head(attn_matrix, prompt_tokens, response_tokens,
                     layer, head, sample_idx, save_dir):
    fig, ax = plt.subplots(figsize=_fig_size(prompt_tokens, response_tokens))
    sns.heatmap(attn_matrix, xticklabels=prompt_tokens, yticklabels=response_tokens,
                ax=ax, cmap="viridis", cbar=True, square=False, linewidths=0)
    ax.set_title(f"Sample {sample_idx} | Layer {layer} | Head {head}\n"
                 f"Response (Q) → Prompt (K)", fontsize=10)
    ax.set_xlabel("Keyword tokens (Key)", fontsize=9)
    ax.set_ylabel("Response tokens (Query)", fontsize=9)
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.tick_params(axis="y", labelsize=6)
    plt.tight_layout()
    fname = os.path.join(save_dir, f"sample{sample_idx}_layer{layer}_head{head}.png")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {fname}")


def plot_avg_heads(attn_layer, prompt_tokens, response_tokens,
                   layer, sample_idx, save_dir):
    avg = attn_layer.mean(axis=0)  # (len_resp, len_prompt)
    fig, ax = plt.subplots(figsize=_fig_size(prompt_tokens, response_tokens))
    sns.heatmap(avg, xticklabels=prompt_tokens, yticklabels=response_tokens,
                ax=ax, cmap="magma", cbar=True, square=False, linewidths=0)
    ax.set_title(f"Sample {sample_idx} | Layer {layer} | Avg heads\n"
                 f"Response (Q) → Prompt (K)", fontsize=10)
    ax.set_xlabel("Keyword tokens (Key)", fontsize=9)
    ax.set_ylabel("Response tokens (Query)", fontsize=9)
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.tick_params(axis="y", labelsize=6)
    plt.tight_layout()
    fname = os.path.join(save_dir, f"sample{sample_idx}_layer{layer}_avg_heads.png")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {fname}")


def plot_all_heads_grid(attn_layer, prompt_tokens, response_tokens,
                        layer, sample_idx, save_dir):
    num_heads = attn_layer.shape[0]
    cols = 4
    rows = (num_heads + cols - 1) // cols
    w = max(3, len(prompt_tokens)   * 0.15)
    h = max(2, len(response_tokens) * 0.12)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * w, rows * h))
    axes = np.array(axes).flatten()
    for head in range(num_heads):
        ax = axes[head]
        ax.imshow(attn_layer[head], aspect="auto", cmap="Blues",
                  vmin=0, vmax=attn_layer[head].max())
        ax.set_title(f"Head {head}", fontsize=8)
        ax.axis("off")
    for ax in axes[num_heads:]:
        ax.axis("off")
    fig.suptitle(f"Sample {sample_idx} | Layer {layer} | All heads\n"
                 f"Response (Q) → Prompt (K)", fontsize=10, y=1.01)
    plt.tight_layout()
    fname = os.path.join(save_dir, f"sample{sample_idx}_layer{layer}_all_heads_grid.png")
    fig.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {fname}")


def plot_layer_summary(cross_attentions, prompt_tokens, sample_idx, save_dir):
    """
    纵轴 = 层，横轴 = prompt token。
    每格 = 该层所有 head 均值，再对所有 response token 取均值，
    直观反映各层 response 整体最关注 prompt 的哪些位置。
    """
    num_layers = len(cross_attentions)
    layer_mean = np.stack(
        [a.mean(axis=0).mean(axis=0) for a in cross_attentions]
    )  # (L, len_prompt)
    fig, ax = plt.subplots(
        figsize=(max(10, len(prompt_tokens) * 0.35), max(4, num_layers * 0.4))
    )
    sns.heatmap(layer_mean,
                xticklabels=prompt_tokens,
                yticklabels=[f"L{i}" for i in range(num_layers)],
                ax=ax, cmap="YlOrRd", cbar=True, linewidths=0)
    ax.set_title(f"Sample {sample_idx} | Layer-wise summary\n"
                 f"Response→Prompt attention (avg over Q & heads)", fontsize=10)
    ax.set_xlabel("Keyword tokens (Key)", fontsize=9)
    ax.set_ylabel("Layer", fontsize=9)
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    fname = os.path.join(save_dir, f"{SAVE_ID}_sample{sample_idx}_layer_summary.png")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {fname}")

# ── 主流程 ────────────────────────────────────────────────────────────────────
def visualize_sample(model, tokenizer, sample_idx: int, data_json: str,
                     save_dir: str, vis_layers=None, vis_heads=None):
    record = load_sample(data_json, sample_idx)
    prompt = record["prompt"]
    
    # ① 让模型自己生成 response
    print(f"\n[样本 {sample_idx}] 生成 response ...")
    generated_ids = generate_response(model, tokenizer, prompt, MAX_NEW_TOKENS)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    print(f"  生成 token 数: {len(generated_ids)}")
    print(f"  生成内容: {generated_text[:200]}{'...' if len(generated_text) > 200 else ''}")

    # ② 对 prompt + generated_response 做前向推断，提取 response→prompt 注意力
    cross_attentions, prompt_tokens, response_tokens = get_cross_attentions(
        model, tokenizer, prompt, generated_ids, MAX_SEQ_TOKENS
    )
    num_layers = len(cross_attentions)
    print(f"  prompt tokens: {len(prompt_tokens)},  response tokens: {len(response_tokens)}"
          f",  层数: {num_layers},  head 数: {cross_attentions[0].shape[0]}")

    # ③ 只保留关键词列
    kw_attentions, kw_labels = filter_keyword_columns(cross_attentions, prompt_tokens, KEYWORDS)
    print(f"  关键词列数: {len(kw_labels)}  {kw_labels}")

    # ④ 可视化（x 轴 = 关键词 token，y 轴 = response token）
    plot_layer_summary(kw_attentions, kw_labels, sample_idx, save_dir)
    
    '''
    layers_to_vis = vis_layers if vis_layers is not None else list(range(num_layers))
    for layer in layers_to_vis:
        if layer >= num_layers:
            continue
        attn_layer = kw_attentions[layer]  # (H, len_resp, num_kw)
        plot_all_heads_grid(attn_layer, kw_labels, response_tokens,
                            layer, sample_idx, save_dir)
        plot_avg_heads(attn_layer, kw_labels, response_tokens,
                       layer, sample_idx, save_dir)
        if vis_heads is not None:
            for head in vis_heads:
                if head < attn_layer.shape[0]:
                    plot_single_head(attn_layer[head], kw_labels, response_tokens,
                                     layer, head, sample_idx, save_dir)
    '''

if __name__ == "__main__":
    '''
    print("加载 tokenizer & 模型 ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float32,
        attn_implementation="eager",  # 必须 eager，否则部分实现不返回注意力权重
    )
    model.eval()
    print("模型加载完成。")
    '''

    #无第一阶段训练的check-point加载
    #构建模型与tokenizer
    BASE_MODEL_ID = "baidu/ERNIE-4.5-0.3B-PT"
    # tokenizer 从 checkpoint 加载（含扩充后的特殊 token）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 必须先加载干净的 base model，否则 from_pretrained 会检测到 adapter_config.json
    # 并自动挂载 adapter，导致 embed_tokens vocab 尺寸不匹配 (101314 vs 103424)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        use_cache=False,
        attn_implementation="eager",
    )
    # 扩词表（与训练时一致），之后再挂 LoRA adapter
    initialize_token_embedding(model, tokenizer)
    model = PeftModel.from_pretrained(model, MODEL_PATH)
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

    for idx in SAMPLE_INDICES:
        print(f"\n===== 可视化样本 {idx} =====")
        visualize_sample(
            model=model,
            tokenizer=tokenizer,
            sample_idx=idx,
            data_json=TEST_JSON,
            save_dir=OUTPUT_DIR,
            vis_layers=VIS_LAYERS,
            vis_heads=VIS_HEADS,
        )

    print(f"\n全部可视化图已保存至: {OUTPUT_DIR}")
