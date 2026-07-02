"""
实验二：分布内模型能力测试 — 生成回答的熵分布

对训练域内的 prompt，让两个模型逐步生成 token，
记录每一生成步的 logit 分布熵，比较 trained vs untrained 的变化。

熵定义：H(t) = -Σ p_i · log(p_i)    （单位：nats）
  - 高熵 → 模型不确定，概率分散在大量 token 上
  - 低熵 → 模型确定，概率集中在少数 token 上

指标：
  1. 每 token 熵分布（violin）
  2. 熵随生成位置的轨迹（mean ± std）
  3. 每样本平均熵 vs varentropy（散点）
  4. 熵热图（样本 × 位置）
  5. 归一化熵 H/log(V)、有效词表大小 exp(H)
"""

import json
import os
import sys
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
_fm._load_fontmanager(try_read_cache=False)
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append("/home/jiuxing_li")
from five_plus_two_optimization.train_model_new.train_base_model.ernie_base_model_train import initialize_token_embedding

# ─── 配置 ─────────────────────────────────────────────────────────────────────
#BASE_MODEL_ID = "baidu/ERNIE-4.5-0.3B-PT"
#TRAINED_PATH  = "five_plus_two_optimization/train_model_new/train_base_model/TRAIN_RESULTS/ernie_base_model_10"
BASE_MODEL_ID = "google/gemma-3-1b-it"
TRAINED_PATH  = "five_plus_two_optimization/train_model_new/train_base_model/TRAIN_RESULTS/gemma_base_model_1"
DATA_PATH     = ("train_json_data/five_plus_two_train_jsonl_data/design_3.27/"
                 "base_model_train/train_set_100_actionized_sort_floor_force_auged(3_times).jsonl")
OUTPUT_DIR    = "five_plus_two_optimization/train_model_new/train_base_model/test/test_pic/entropy_test"
SAVE_ID       = "gemma_1" #var

N_SAMPLES      = 30    # 取多少条训练数据做评估
MAX_NEW_TOKENS = 80    # 每条最多生成多少 token
HEATMAP_STEPS  = 60    # 热图截断到多少步（对齐短样本）

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 数据加载 ──────────────────────────────────────────────────────────────────

def load_prompts(path: str, n: int) -> list[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            prompts.append(record["prompt"])
            if len(prompts) >= n:
                break
    print(f"加载 {len(prompts)} 条 in-distribution prompt")
    return prompts

# ─── 模型加载 ──────────────────────────────────────────────────────────────────

def load_untrained_model():
    print("加载 untrained 模型 ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=True,
    )
    initialize_token_embedding(model, tokenizer)
    model.eval()
    return tokenizer, model


def load_trained_model():
    print("加载 trained 模型 ...")
    model = AutoModelForCausalLM.from_pretrained(
        TRAINED_PATH, device_map="auto", torch_dtype=torch.bfloat16, use_cache=True
    )
    tokenizer = AutoTokenizer.from_pretrained(TRAINED_PATH)
    model.eval()
    return tokenizer, model

# ─── 熵计算核心 ───────────────────────────────────────────────────────────────

def compute_entropy_trajectory(model, tokenizer, prompts: list[str]) -> list[dict]:
    """
    对每条 prompt 逐步生成 MAX_NEW_TOKENS 个 token，
    记录每一步的熵（nats）、归一化熵、有效词表大小。

    返回：每条 prompt 的 dict，包含 entropy_traj 等字段。
    """
    vocab_size   = tokenizer.vocab_size   # 用 tokenizer 的 vocab_size 归一化
    log_V        = math.log(vocab_size)
    results      = []

    for idx, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            gen_out = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # gen_out.scores: tuple，长度 = 实际生成步数
        # 每个元素 shape: (1, vocab_size)，为原始 logit（未做 softmax）
        entropy_traj    = []
        norm_entropy    = []
        eff_vocab       = []
        generated_token_ids = []

        for step_logits in gen_out.scores:
            logits = step_logits[0].float()           # (vocab_size,)
            probs  = torch.softmax(logits, dim=-1)    # 精确概率分布
            H      = -torch.sum(probs * torch.log(probs + 1e-12)).item()

            entropy_traj.append(H)
            norm_entropy.append(H / log_V)
            eff_vocab.append(math.exp(H))
            generated_token_ids.append(logits.argmax().item())

        n_steps = len(entropy_traj)
        results.append({
            "idx":              idx,
            "n_steps":          n_steps,
            "entropy_traj":     entropy_traj,           # list[float], nats
            "norm_entropy":     norm_entropy,            # list[float], ∈[0,1]
            "eff_vocab":        eff_vocab,               # list[float]
            "mean_entropy":     float(np.mean(entropy_traj)),
            "median_entropy":   float(np.median(entropy_traj)),
            "varentropy":       float(np.var(entropy_traj)),   # 熵的方差
            "mean_norm_entropy": float(np.mean(norm_entropy)),
            "mean_eff_vocab":   float(np.mean(eff_vocab)),
        })

        print(f"  [{idx+1:02d}/{len(prompts)}] "
              f"steps={n_steps}  "
              f"mean_H={results[-1]['mean_entropy']:.3f}  "
              f"varentropy={results[-1]['varentropy']:.3f}  "
              f"norm={results[-1]['mean_norm_entropy']:.3f}")

    return results

# ─── 统计 ──────────────────────────────────────────────────────────────────────

def compute_stats(results: list[dict]) -> dict:
    all_entropies    = [h for r in results for h in r["entropy_traj"]]
    all_norm         = [h for r in results for h in r["norm_entropy"]]
    all_eff_vocab    = [v for r in results for v in r["eff_vocab"]]
    per_sample_mean  = [r["mean_entropy"]   for r in results]
    per_sample_var   = [r["varentropy"]     for r in results]
    per_sample_norm  = [r["mean_norm_entropy"] for r in results]

    # 对齐截断到 HEATMAP_STEPS 并填 NaN 用于轨迹均值
    max_steps = max(r["n_steps"] for r in results)
    traj_matrix = np.full((len(results), max_steps), np.nan)
    for i, r in enumerate(results):
        traj_matrix[i, :r["n_steps"]] = r["entropy_traj"]

    return {
        "all_entropies":    all_entropies,
        "all_norm":         all_norm,
        "all_eff_vocab":    all_eff_vocab,
        "per_sample_mean":  per_sample_mean,
        "per_sample_var":   per_sample_var,
        "per_sample_norm":  per_sample_norm,
        "traj_matrix":      traj_matrix,          # (N, max_steps), NaN-padded
        "traj_mean":        np.nanmean(traj_matrix, axis=0),
        "traj_std":         np.nanstd(traj_matrix, axis=0),
        "global_mean":      float(np.mean(all_entropies)),
        "global_median":    float(np.median(all_entropies)),
        "global_std":       float(np.std(all_entropies)),
        "mean_varentropy":  float(np.mean(per_sample_var)),
        "mean_norm":        float(np.mean(all_norm)),
    }

# ─── 可视化 ───────────────────────────────────────────────────────────────────

def plot_overview(s_ut, s_tr, results_ut):
    """6宫格概览图"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"实验二：生成回答熵分布  ({SAVE_ID})", fontsize=14)

    # ── 1. 每 token 熵分布 violin ─────────────────────────────────────────────
    ax = axes[0, 0]
    parts = ax.violinplot(
        [s_ut["all_entropies"], s_tr["all_entropies"]],
        positions=[1, 2], showmedians=True, showextrema=True,
    )
    for pc, c in zip(parts["bodies"], ["#5B9BD5", "#ED7D31"]):
        pc.set_facecolor(c); pc.set_alpha(0.7)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Untrained", "Trained"])
    ax.set_ylabel("熵 H（nats）")
    ax.set_title("每 token 熵分布")
    ax.grid(axis="y", alpha=0.3)
    for pos, data, c in zip([1, 2], [s_ut["all_entropies"], s_tr["all_entropies"]],
                             ["#5B9BD5", "#ED7D31"]):
        ax.text(pos, ax.get_ylim()[0] + 0.1,
                f"μ={np.mean(data):.2f}\nσ={np.std(data):.2f}",
                ha="center", fontsize=9, color=c)

    # ── 2. 每样本平均熵分布 ───────────────────────────────────────────────────
    ax = axes[0, 1]
    parts2 = ax.violinplot(
        [s_ut["per_sample_mean"], s_tr["per_sample_mean"]],
        positions=[1, 2], showmedians=True, showextrema=True,
    )
    for pc, c in zip(parts2["bodies"], ["#5B9BD5", "#ED7D31"]):
        pc.set_facecolor(c); pc.set_alpha(0.7)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Untrained", "Trained"])
    ax.set_ylabel("样本平均熵（nats）")
    ax.set_title("每样本平均熵分布")
    ax.grid(axis="y", alpha=0.3)
    for pos, data in zip([1, 2], [s_ut["per_sample_mean"], s_tr["per_sample_mean"]]):
        ax.text(pos, min(s_ut["per_sample_mean"] + s_tr["per_sample_mean"]) - 0.1,
                f"μ={np.mean(data):.2f}", ha="center", fontsize=9)

    # ── 3. 归一化熵 H/log(V) 分布 ────────────────────────────────────────────
    ax = axes[0, 2]
    parts3 = ax.violinplot(
        [s_ut["all_norm"], s_tr["all_norm"]],
        positions=[1, 2], showmedians=True, showextrema=True,
    )
    for pc, c in zip(parts3["bodies"], ["#5B9BD5", "#ED7D31"]):
        pc.set_facecolor(c); pc.set_alpha(0.7)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Untrained", "Trained"])
    ax.set_ylabel("归一化熵 H / log(V)")
    ax.set_title(f"归一化熵分布  (V={results_ut[0]['n_steps']} steps, 首个样本)")
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    for pos, data in zip([1, 2], [s_ut["all_norm"], s_tr["all_norm"]]):
        ax.text(pos, 0.02, f"μ={np.mean(data):.3f}", ha="center", fontsize=9)

    # ── 4. 熵随位置的轨迹 ─────────────────────────────────────────────────────
    ax = axes[1, 0]
    max_plot = min(HEATMAP_STEPS, s_ut["traj_matrix"].shape[1], s_tr["traj_matrix"].shape[1])
    steps = np.arange(1, max_plot + 1)
    mean_ut = s_ut["traj_mean"][:max_plot]
    std_ut  = s_ut["traj_std"][:max_plot]
    mean_tr = s_tr["traj_mean"][:max_plot]
    std_tr  = s_tr["traj_std"][:max_plot]
    ax.plot(steps, mean_ut, color="#5B9BD5", linewidth=2, label="Untrained")
    ax.fill_between(steps, mean_ut - std_ut, mean_ut + std_ut,
                    color="#5B9BD5", alpha=0.2)
    ax.plot(steps, mean_tr, color="#ED7D31", linewidth=2, label="Trained")
    ax.fill_between(steps, mean_tr - std_tr, mean_tr + std_tr,
                    color="#ED7D31", alpha=0.2)
    ax.set_xlabel("生成步（token 位置）"); ax.set_ylabel("熵 H（nats）")
    ax.set_title("熵随生成位置的变化轨迹（mean ± std）")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # ── 5. Varentropy 分布（熵的方差）────────────────────────────────────────
    ax = axes[1, 1]
    parts4 = ax.violinplot(
        [s_ut["per_sample_var"], s_tr["per_sample_var"]],
        positions=[1, 2], showmedians=True, showextrema=True,
    )
    for pc, c in zip(parts4["bodies"], ["#5B9BD5", "#ED7D31"]):
        pc.set_facecolor(c); pc.set_alpha(0.7)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Untrained", "Trained"])
    ax.set_ylabel("Varentropy（熵的方差）")
    ax.set_title("Varentropy 分布\n（高 varentropy = 推理过程起伏大）")
    ax.grid(axis="y", alpha=0.3)
    for pos, data in zip([1, 2], [s_ut["per_sample_var"], s_tr["per_sample_var"]]):
        ax.text(pos, min(s_ut["per_sample_var"] + s_tr["per_sample_var"]) - 0.01,
                f"μ={np.mean(data):.3f}", ha="center", fontsize=9)

    # ── 6. 平均熵 vs Varentropy 散点（每个点=一个样本）──────────────────────
    ax = axes[1, 2]
    ax.scatter(s_ut["per_sample_mean"], s_ut["per_sample_var"],
               color="#5B9BD5", s=60, alpha=0.8, label="Untrained", edgecolors="white")
    ax.scatter(s_tr["per_sample_mean"], s_tr["per_sample_var"],
               color="#ED7D31", s=60, alpha=0.8, label="Trained", edgecolors="white",
               marker="^")
    ax.set_xlabel("样本平均熵 μ_H（nats）")
    ax.set_ylabel("Varentropy σ²_H")
    ax.set_title("平均熵 vs Varentropy（每点=一个样本）")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{SAVE_ID}_entropy_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存图表: {path}")


def plot_heatmaps(s_ut, s_tr):
    """熵热图：行=样本，列=生成位置，颜色=熵值"""
    n_cols = HEATMAP_STEPS
    mat_ut = s_ut["traj_matrix"][:, :n_cols]
    mat_tr = s_tr["traj_matrix"][:, :n_cols]

    vmin = np.nanmin(np.concatenate([mat_ut.ravel(), mat_tr.ravel()]))
    vmax = np.nanmax(np.concatenate([mat_ut.ravel(), mat_tr.ravel()]))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"熵热图（行=样本，列=生成位置）  {SAVE_ID}", fontsize=13)

    for ax, mat, title in [
        (axes[0], mat_ut, "Untrained"),
        (axes[1], mat_tr, "Trained"),
    ]:
        # 用 masked array 处理 NaN（样本提前结束的位置）
        masked = np.ma.masked_invalid(mat)
        cmap = matplotlib.cm.get_cmap("RdYlGn_r").copy()
        cmap.set_bad(color="#dddddd")
        im = ax.imshow(masked, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xlabel("生成步（token 位置）", fontsize=11)
        ax.set_ylabel("样本索引", fontsize=11)
        ax.set_title(f"{title}\n（红=高熵/不确定，绿=低熵/确定，灰=EOS后）",
                     fontsize=11)
        plt.colorbar(im, ax=ax, label="熵（nats）", fraction=0.03, pad=0.02)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{SAVE_ID}_entropy_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存图表: {path}")


def plot_trajectory_per_sample(results_ut, results_tr):
    """每个样本一条细线的轨迹图，透明度叠加"""
    max_plot = HEATMAP_STEPS
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    fig.suptitle(f"各样本熵轨迹  {SAVE_ID}", fontsize=12)

    for ax, results, color, title in [
        (axes[0], results_ut, "#5B9BD5", "Untrained"),
        (axes[1], results_tr, "#ED7D31", "Trained"),
    ]:
        for r in results:
            traj = r["entropy_traj"][:max_plot]
            ax.plot(range(1, len(traj) + 1), traj,
                    color=color, alpha=0.35, linewidth=0.8)
        # 均值线
        mean_traj = np.nanmean(
            np.array([r["entropy_traj"][:max_plot] +
                      [np.nan] * (max_plot - len(r["entropy_traj"][:max_plot]))
                      for r in results]),
            axis=0,
        )
        ax.plot(range(1, len(mean_traj) + 1), mean_traj,
                color="black", linewidth=2, label="均值")
        ax.set_xlabel("生成步"); ax.set_ylabel("熵（nats）")
        ax.set_title(title); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{SAVE_ID}_trajectory_per_sample.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存图表: {path}")

# ─── 保存 & 摘要 ──────────────────────────────────────────────────────────────

def save_results_json(results_ut, results_tr, s_ut, s_tr):
    def stats_to_dict(s):
        return {
            "global_mean":     s["global_mean"],
            "global_median":   s["global_median"],
            "global_std":      s["global_std"],
            "mean_varentropy": s["mean_varentropy"],
            "mean_norm":       s["mean_norm"],
        }
    out = {
        "untrained": {
            "stats": stats_to_dict(s_ut),
            "per_sample": [
                {k: v for k, v in r.items() if k != "traj_matrix"}
                for r in results_ut
            ],
        },
        "trained": {
            "stats": stats_to_dict(s_tr),
            "per_sample": [
                {k: v for k, v in r.items() if k != "traj_matrix"}
                for r in results_tr
            ],
        },
    }
    path = os.path.join(OUTPUT_DIR, f"{SAVE_ID}_entropy_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"保存结果: {path}")


def print_summary(s_ut, s_tr):
    print("\n" + "=" * 58)
    print("实验二  生成回答熵分布  结果摘要")
    print("=" * 58)
    print(f"{'指标':<20} {'Untrained':>12}  {'Trained':>12}  {'Δ':>8}")
    print("-" * 58)
    metrics = [
        ("平均熵（nats）",      s_ut["global_mean"],     s_tr["global_mean"]),
        ("熵标准差",            s_ut["global_std"],      s_tr["global_std"]),
        ("均值 Varentropy",     s_ut["mean_varentropy"], s_tr["mean_varentropy"]),
        ("归一化熵 H/log(V)",   s_ut["mean_norm"],       s_tr["mean_norm"]),
    ]
    for name, ut_val, tr_val in metrics:
        delta = tr_val - ut_val
        sign  = "+" if delta >= 0 else ""
        print(f"{name:<20} {ut_val:>12.4f}  {tr_val:>12.4f}  {sign}{delta:>7.4f}")
    print("-" * 58)
    delta_mean = s_tr["global_mean"] - s_ut["global_mean"]
    direction  = "↓ 熵降低（更确定）" if delta_mean < 0 else "↑ 熵升高（更不确定）"
    print(f"结论: {direction}  Δ={delta_mean:+.4f} nats")
    print("=" * 58)

# ─── 主流程 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prompts = load_prompts(DATA_PATH, N_SAMPLES)

    ut_tokenizer, ut_model = load_untrained_model()
    print(f"\n[Untrained] 开始计算熵轨迹（{N_SAMPLES} 条 prompt，最多 {MAX_NEW_TOKENS} 步）")
    results_ut = compute_entropy_trajectory(ut_model, ut_tokenizer, prompts)
    del ut_model; torch.cuda.empty_cache()

    tr_tokenizer, tr_model = load_trained_model()
    print(f"\n[Trained] 开始计算熵轨迹")
    results_tr = compute_entropy_trajectory(tr_model, tr_tokenizer, prompts)
    del tr_model; torch.cuda.empty_cache()

    s_ut = compute_stats(results_ut)
    s_tr = compute_stats(results_tr)

    print_summary(s_ut, s_tr)
    plot_overview(s_ut, s_tr, results_ut)
    plot_heatmaps(s_ut, s_tr)
    plot_trajectory_per_sample(results_ut, results_tr)
    save_results_json(results_ut, results_tr, s_ut, s_tr)

    print(f"\n全部结果已保存至: {OUTPUT_DIR}")
