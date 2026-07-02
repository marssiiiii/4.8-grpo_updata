"""
各层对两类 prompt 信息（几何约束 vs 受力信息）编码相似度分析与可视化。

prompt 结构（新格式）：
  <s>...upper layer lineload:<LINELOAD>...upper layer post:<POST>...context:<exterior_wall>...structures:...</s>

分析：对每一层的隐层表示，分别对 context 区域和 upper layer 区域做 mean pooling，
      计算余弦相似度，观察模型各层是否对两类信息产生了不同的表示。
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
_fm._load_fontmanager(try_read_cache=False)
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.append("/home/jiuxing_li/five_plus_two_optimization/train_model_new/train_base_model")
from ernie_base_model_train import initialize_token_embedding

# ─── 配置 ─────────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "baidu/ERNIE-4.5-0.3B-PT"
MODEL_PATH = "five_plus_two_optimization/train_model_new/train_base_model/TRAIN_RESULTS/ernie_base_model_11"
DATA_JSON   = "train_json_data/five_plus_two_train_jsonl_data/design_3.27/base_model_train/train_set_100_actionized_sort_floor_force_auged(3_times).jsonl"
OUTPUT_DIR  = "five_plus_two_optimization/train_model_new/train_base_model/test/test_pic/similarity_vis/ernie_11"
SAVE_ID     = "ernie_11"

MAX_SEQ_TOKENS   = 9999
MAX_FORCE_SAMPLES = 10    # 最多取多少个含受力信息的样本
CONSTRAINED_LAYERS = 10    # 建议施加正交约束的层数（用于图中标注）

#os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 数据加载 ──────────────────────────────────────────────────────────────────

def load_samples_with_force(data_json: str, n: int) -> list[tuple[int, dict]]:
    """加载前 n 个含受力信息（POST 或 LINELOAD）的样本"""
    results = []
    with open(data_json, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            if record.get("pre_post_text") or record.get("pre_lineload_text"):
                results.append((i, record))
            if len(results) >= n:
                break
    return results

# ─── Token 范围定位 ────────────────────────────────────────────────────────────

def find_section_token_ranges(prompt: str, tokenizer) -> dict[str, tuple[int, int]]:
    upper_start_char = prompt.find("upper layer lineload:")
    ctx_char            = prompt.find("context:")

    # context+structures：到 </s> 之前，不含结尾 EOS
    eos_char     = prompt.rfind("</s>")
    ctx_end_char = eos_char

    encoding = tokenizer(
        prompt,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_SEQ_TOKENS,
    )
    offsets = encoding["offset_mapping"]  # list[(char_start, char_end)]

    def chars_to_token_range(start_char: int, end_char: int) -> tuple[int, int]:
        start_tok, end_tok = None, None
        for i, (c_s, c_e) in enumerate(offsets):
            if c_s == 0 and c_e == 0:   # BOS/EOS special token
                continue
            if start_tok is None and c_e > start_char:
                start_tok = i
            if c_s < end_char:
                end_tok = i + 1
        return (start_tok or 0, end_tok or len(offsets))

    ctx_start, ctx_end = chars_to_token_range(ctx_char, ctx_end_char)
    # upper_end 直接取 ctx_start，即 context 内容起始 token，避免边界空白 token 污染
    upper_start, _ = chars_to_token_range(upper_start_char, ctx_char)
    upper_end       = ctx_start

    return {
        "context":     (ctx_start, ctx_end),
        "upper_layer": (upper_start, upper_end),
    }

# ─── 隐层提取 ─────────────────────────────────────────────────────────────────

def extract_hidden_states(model, tokenizer, prompt: str) -> list[np.ndarray]:
    """
    对 prompt 做一次前向推断，返回每个 transformer 层的隐层表示。
    返回：list of np.ndarray，shape (seq_len, d_model)，长度 = 层数（不含 embedding 层）。
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_TOKENS,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states[0] 是 embedding 层输出，[1:] 才是各 transformer 层
    return [
        h.squeeze(0).cpu().float().numpy()
        for h in outputs.hidden_states[1:]
    ]

# ─── 相似度计算 ───────────────────────────────────────────────────────────────

def compute_cosine_similarity_per_layer(
    hidden_states: list[np.ndarray],
    ctx_range: tuple[int, int],
    upper_range: tuple[int, int],
) -> np.ndarray:
    """
    对每一层，分别对 context 区域和 upper layer 区域做 mean pooling，
    计算两者的余弦相似度。

    返回：np.ndarray shape (num_layers,)
    """
    sims = []
    for h in hidden_states:
        h1 = h[ctx_range[0]:ctx_range[1]].mean(axis=0)
        h2 = h[upper_range[0]:upper_range[1]].mean(axis=0)
        cos = np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-8)
        sims.append(float(cos))
    return np.array(sims)

# ─── 可视化 ───────────────────────────────────────────────────────────────────

def _constraint_band(ax, num_layers: int):
    """在图上标注建议施加正交约束的区域"""
    end = min(CONSTRAINED_LAYERS + 0.5, num_layers)
    ax.axvspan(0.5, end, alpha=0.08, color="red")
    ax.axvline(x=end, color="red", linestyle="--", linewidth=1.2,
               label=f"建议正交约束边界（层 1-{CONSTRAINED_LAYERS}）")

def plot_similarity_curves(all_sims: dict[str, np.ndarray]):
    """每个样本一条曲线：x=层编号，y=余弦相似度"""
    num_layers = next(iter(all_sims.values())).shape[0]
    layers = list(range(1, num_layers + 1))

    fig, ax = plt.subplots(figsize=(11, 6))
    for label, sims in all_sims.items():
        ax.plot(layers, sims, marker="o", markersize=4, linewidth=1.5, label=label)

    _constraint_band(ax, num_layers)
    ax.axhline(y=0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Transformer Layer Code", fontsize=12)
    ax.set_ylabel("Cosin Similarity (context vs upper layer)", fontsize=12)
    ax.set_title(
        " prompt similarity\n",
        fontsize=13,
    )
    ax.set_xticks(layers)
    ax.set_ylim(-0.15, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"{SAVE_ID}_similarity_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {path}")

def plot_similarity_heatmap(all_sims: dict[str, np.ndarray]):
    """热力图：行=样本，列=层，颜色=余弦相似度（红=高相似，绿=低相似）"""
    labels = list(all_sims.keys())
    matrix = np.stack([all_sims[l] for l in labels])   # (n_samples, n_layers)
    num_layers = matrix.shape[1]

    fig, ax = plt.subplots(
        figsize=(max(10, num_layers * 0.65), max(4, len(labels) * 0.5))
    )
    sns.heatmap(
        matrix,
        xticklabels=[f"L{i}" for i in range(1, num_layers + 1)],
        yticklabels=labels,
        ax=ax,
        cmap="RdYlGn_r",   # 红=高相似（差），绿=低相似（好）
        vmin=-0.1,
        vmax=1.0,
        cbar_kws={"label": "cosin similarity"},
        linewidths=0.3,
    )
    # 标注约束边界
    ax.axvline(x=CONSTRAINED_LAYERS, color="red", linewidth=2.0, linestyle="--")
    ax.text(
        CONSTRAINED_LAYERS + 0.1, -0.5,
        f"← 建议约束边界",
        color="red", fontsize=9, va="top",
    )
    ax.set_title(
        "context vs upper layer similarity\n",
        fontsize=12,
    )
    ax.set_xlabel("layer code", fontsize=11)
    ax.set_ylabel("sample", fontsize=11)
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"{SAVE_ID}_similarity_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {path}")

def plot_mean_with_std(all_sims: dict[str, np.ndarray]):
    """均值 ± 标准差曲线：直观反映各层分离程度及跨样本稳定性"""
    matrix = np.stack(list(all_sims.values()))   # (n_samples, n_layers)
    mean_sim = matrix.mean(axis=0)
    std_sim  = matrix.std(axis=0)
    num_layers = matrix.shape[1]
    layers = list(range(1, num_layers + 1))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(layers, mean_sim, marker="o", color="steelblue", linewidth=2, label="均值")
    ax.fill_between(
        layers,
        mean_sim - std_sim,
        mean_sim + std_sim,
        alpha=0.2, color="steelblue", label="±1 标准差",
    )

    _constraint_band(ax, num_layers)
    ax.axhline(y=0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Transformer 层编号", fontsize=12)
    ax.set_ylabel("余弦相似度", fontsize=12)
    ax.set_title(
        f"各层平均编码相似度（{len(all_sims)} 个含受力样本）\n"
        "context 区域 vs upper layer 区域",
        fontsize=13,
    )
    ax.set_xticks(layers)
    ax.set_ylim(-0.15, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"{SAVE_ID}_similarity_mean_std.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {path}")

def save_similarity_json(all_sims: dict[str, np.ndarray]):
    """将数值结果保存为 JSON，方便后续对比不同 checkpoint"""
    data = {label: sims.tolist() for label, sims in all_sims.items()}
    path = os.path.join(OUTPUT_DIR, f"{SAVE_ID}_similarity_values.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"保存: {path}")

def print_summary(all_sims: dict[str, np.ndarray]):
    """打印各层平均相似度数值摘要"""
    matrix = np.stack(list(all_sims.values()))
    mean_sim = matrix.mean(axis=0)
    print("\n各层平均余弦相似度（context vs upper layer）:")
    print(f"{'层':>5}  {'相似度':>8}  备注")
    print("-" * 35)
    for i, sim in enumerate(mean_sim):
        tag = "← 建议施加正交约束" if i < CONSTRAINED_LAYERS else ""
        print(f"  L{i+1:2d}  {sim:8.4f}  {tag}")

# ─── 提取正确性测试 ──────────────────────────────────────────────────────────────

def test_section_extraction(data_json: str, tokenizer, n_samples: int = 3):
    """
    打印前 n_samples 个含受力信息样本的提取结果，供人工核验：
      - 各关键字的字符位置
      - 提取到的 context / upper_layer token 范围
      - 对应的实际文本片段（前 200 字符）
    """
    print("=" * 70)
    print("【提取正确性测试】")
    print("=" * 70)

    found = 0
    with open(data_json, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            record = json.loads(line)
            prompt = record.get("prompt", "")
            if "<POST>" not in prompt and "<LINELOAD>" not in prompt:
                continue

            print(f"\n── 样本 {line_idx}  house={record.get('house')}  floor={record.get('floor')} ──")

            # 关键字位置
            kw_positions = {
                "upper layer lineload:": prompt.find("upper layer lineload:"),
                "upper layer post:":     prompt.find("upper layer post:"),
                "context:":              prompt.find("context:"),
                "structures:":           prompt.find("structures:"),
            }
            print("  关键字字符位置:")
            for kw, pos in kw_positions.items():
                print(f"    {kw!r:30s} -> {pos}")

            # token 范围
            ranges = find_section_token_ranges(prompt, tokenizer)
            if not ranges:
                print("  !! find_section_token_ranges 返回空，跳过")
                found += 1
                if found >= n_samples:
                    break
                continue

            ctx_r   = ranges["context"]
            upper_r = ranges["upper_layer"]
            print(f"  token 范围:")
            print(f"    upper_layer           tokens: [{upper_r[0]}, {upper_r[1]})  长度={upper_r[1]-upper_r[0]}")
            print(f"    context+structures    tokens: [{ctx_r[0]}, {ctx_r[1]})  长度={ctx_r[1]-ctx_r[0]}")

            # 合法性检查
            if upper_r[1] > ctx_r[0]:
                print("  !! 警告：upper_layer 与 context 区间重叠或顺序错误！")
            else:
                print("  ✓ 区间不重叠，顺序正确（upper → context）")

            # 实际文本内容预览（通过 encode→decode 还原）
            enc = tokenizer(
                prompt,
                return_offsets_mapping=True,
                add_special_tokens=True,
                truncation=True,
                max_length=MAX_SEQ_TOKENS,
            )
            input_ids = enc["input_ids"]

            def decode_range(start, end):
                ids = input_ids[start:end]
                return tokenizer.decode(ids, skip_special_tokens=False)

            upper_text = decode_range(*upper_r)
            ctx_text   = decode_range(*ctx_r)
            print(f"  upper_layer 文本预览 (前200字符):")
            print(f"    {upper_text[:]!r}")
            print(f"  context+structures 文本预览 (前200字符):")
            print(f"    {ctx_text[:]!r}")

            found += 1
            if found >= n_samples:
                break

    print("\n" + "=" * 70)
    print(f"测试完成，共检查 {found} 个含受力信息的样本")
    print("=" * 70)


# ─── 单样本处理 ───────────────────────────────────────────────────────────────

def analyze_sample(
    model, tokenizer, sample_idx: int, record: dict
) -> np.ndarray | None:
    prompt = record["prompt"]

    ranges = find_section_token_ranges(prompt, tokenizer)
    if not ranges:
        print(f"  [样本 {sample_idx}] 无法定位区域，跳过")
        return None

    ctx_range   = ranges["context"]
    upper_range = ranges["upper_layer"]

    print(f"  [样本 {sample_idx}] "
          f"context tokens [{ctx_range[0]}, {ctx_range[1]})  "
          f"upper layer tokens [{upper_range[0]}, {upper_range[1]})")

    hidden_states = extract_hidden_states(model, tokenizer, prompt)
    return compute_cosine_similarity_per_layer(hidden_states, ctx_range, upper_range)

# ─── 主流程 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── 仅测试提取逻辑（无需加载大模型，快速验证）─────────────────────────────
    _test_tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if _test_tok.pad_token is None:
        _test_tok.pad_token = _test_tok.eos_token
    # 加载自定义 special tokens（使 <POST> <LINELOAD> 等被正确识别）
    new_tokens = ['<wall>', '<exterior_wall>', '<opening>', '<inoutbox>',
                    '<beam>', '<shearwall>', '<POST>', '<LINELOAD>', '<add>', '<remove>']
    _test_tok.add_special_tokens({"additional_special_tokens": new_tokens})
    test_section_extraction(DATA_JSON, _test_tok, n_samples=3)

    # ── 加载模型（与 attention_visualization.py 保持一致）──────────────────────
    '''
    print("加载 tokenizer & 模型 ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        use_cache=False,
        attn_implementation="eager",
    )
    initialize_token_embedding(model, tokenizer)
    model = PeftModel.from_pretrained(model, MODEL_PATH)
    model.eval()

    emb = torch.load(f"{MODEL_PATH}/embedding.safetensors")
    emb = emb.to(model.get_input_embeddings().weight.dtype)
    model.get_input_embeddings().weight.data.copy_(emb)
    print("LoRA + Embedding 恢复成功\n")
    '''
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,device_map="auto",torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()

    # ── 加载含受力信息的样本 ──────────────────────────────────────────────────
    print(f"加载含受力信息的样本（最多 {MAX_FORCE_SAMPLES} 个）...")
    samples = load_samples_with_force(DATA_JSON, n=MAX_FORCE_SAMPLES)
    print(f"找到 {len(samples)} 个样本\n")

    # ── 逐样本分析 ────────────────────────────────────────────────────────────
    all_sims: dict[str, np.ndarray] = {}
    for sample_idx, record in samples:
        print(f"[样本 {sample_idx}] house={record['house']}  floor={record['floor']}")
        sims = analyze_sample(model, tokenizer, sample_idx, record)
        if sims is not None:
            label = f"s{sample_idx}_{record['house']}_{record['floor']}"
            all_sims[label] = sims

    if not all_sims:
        print("没有有效样本，退出")
        sys.exit(1)

    # ── 可视化 & 保存 ─────────────────────────────────────────────────────────
    print(f"\n生成可视化图表（共 {len(all_sims)} 个样本）...")
    plot_similarity_curves(all_sims)
    plot_similarity_heatmap(all_sims)
    plot_mean_with_std(all_sims)
    save_similarity_json(all_sims)
    print_summary(all_sims)

    print(f"\n全部结果已保存至: {OUTPUT_DIR}")
