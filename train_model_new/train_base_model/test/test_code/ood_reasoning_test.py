"""
实验一：通用推理能力测试（OOD — 数学推理 + 坐标几何推理）

Part A：数学多选题 log-prob 打分
Part B：数学生成式应用题（答案提取）
Part C：坐标几何多选题 log-prob 打分（坐标距离/平行垂直/中点变换/面积周长）
Part D：坐标几何生成题（答案提取）
"""

import json
import os
import re
import sys
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
OUTPUT_DIR    = "five_plus_two_optimization/train_model_new/train_base_model/test/test_pic/ood_reasoning"
SAVE_ID       = "gemma_1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Part A：多选题题库 ────────────────────────────────────────────────────────
# (题干, A, B, C, D, 正确答案字母, 难度1-3)
MCQ_QUESTIONS = [
    # 难度1：基础运算
    ("3 + 5 × 2 = ？",                         "13",   "16",   "11",   "10",   "A", 1),  # 3+10=13
    ("100 ÷ 4 ÷ 5 = ？",                        "20",   "10",   "5",    "2",    "C", 1),  # 25÷5=5
    ("1 + 2 + 3 + 4 + 5 = ？",                  "10",   "12",   "15",   "18",   "C", 1),
    ("√144 = ？",                               "11",   "12",   "13",   "14",   "B", 1),
    ("2 的 10 次方等于多少？",                   "512",  "1024", "2048", "256",  "B", 1),
    ("一个正方形边长为 6，其面积是多少？",        "24",   "30",   "36",   "42",   "C", 1),
    ("一个三角形三个内角之和是多少度？",          "90",   "360",  "270",  "180",  "D", 1),  # 180°
    ("0.5 × 0.5 = ？",                          "0.1",  "0.25", "0.5",  "1.0",  "B", 1),

    # 难度2：稍复杂推理
    ("一件商品原价 200 元，先涨价 50%，再打八折，最终价格是多少元？",
     "200",  "220",  "240",  "260",  "C", 2),  # 200×1.5×0.8 = 240
    ("小明有 48 颗糖，分给 6 个朋友，每人再多给 2 颗，共需要多少颗糖？",
     "48",   "52",   "56",   "60",   "D", 2),  # 48+6×2=60
    ("一列火车长 200 米，以 72 km/h 的速度通过一座 400 米长的桥，需要多少秒？",
     "25",   "30",   "35",   "40",   "C", 2),  # (200+400)/(72×1000/3600)=30... wait: 72km/h=20m/s, 600/20=30
    # Fix: 600/20 = 30 → B
    ("一列火车长 200 米，以 72 km/h 的速度通过一座 400 米长的桥，需要多少秒？",
     "25",   "30",   "35",   "40",   "B", 2),  # 72km/h=20m/s, (200+400)/20=30
    ("1/3 + 1/6 = ？",                          "1/2",  "1/3",  "1/4",  "2/9",  "A", 2),
    ("如果 x + 3 = 10，则 2x - 1 = ？",         "11",   "13",   "15",   "17",   "B", 2),  # x=7, 2×7-1=13
    ("圆的半径为 5，其面积最接近哪个值（π≈3.14）？",
     "15.7", "31.4", "78.5", "157",  "C", 2),  # π×25≈78.5

    # 难度3：多步推理
    ("甲乙两人同时从两地出发相向而行，距离 120 km，甲速 30 km/h，乙速 10 km/h，几小时后相遇？",
     "3",    "2",    "4",    "5",    "A", 3),  # 120/(30+10)=3
    ("连续三个偶数之和为 48，最小的偶数是多少？",
     "12",   "16",   "18",   "14",   "D", 3),  # x+(x+2)+(x+4)=48 → x=14
    ("一个数的 3 倍加 7 等于 28，这个数是多少？",
     "5",    "9",    "7",    "11",   "C", 3),  # (28-7)/3=7
    ("100 以内既能被 3 整除又能被 4 整除的最大正整数是？",
     "84",   "96",   "90",   "72",   "B", 3),  # lcm(3,4)=12, 最大倍数≤100=96
    ("一个等差数列首项为 2，公差为 3，第 10 项是多少？",
     "27",   "31",   "33",   "29",   "D", 3),  # 2+(10-1)×3=29
]

# 去掉重复题（第11题有两个版本，保留修正版）
MCQ_QUESTIONS = [q for i, q in enumerate(MCQ_QUESTIONS) if i != 10]

MCQ_TEMPLATE = (
    "以下是一道数学单项选择题，请直接给出正确选项字母。\n\n"
    "题目：{question}\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
    "答案："
)

# ─── Part B：生成式应用题 ──────────────────────────────────────────────────────
# (题干提示词, 标准答案数值, 难度1-3)
GEN_QUESTIONS = [
    # 难度1
    ("小明有 15 个苹果，吃了 4 个，还剩多少个？请直接给出数字答案。",
     11, 1),
    ("一块长方形土地长 8 米，宽 5 米，面积是多少平方米？请直接给出数字答案。",
     40, 1),
    ("班里有 32 名同学，分成 4 组，每组多少人？请直接给出数字答案。",
     8,  1),
    ("一支铅笔 2 元，买 7 支需要多少元？请直接给出数字答案。",
     14, 1),
    ("从 1 加到 5 的总和是多少？请直接给出数字答案。",
     15, 1),

    # 难度2
    ("小红的存款是小明的 3 倍，小明有 250 元，小红有多少元？请直接给出数字答案。",
     750, 2),
    ("一辆汽车以 60 km/h 行驶了 2.5 小时，行驶了多少千米？请直接给出数字答案。",
     150, 2),
    ("商场打七折促销，一件原价 300 元的外套现价多少元？请直接给出数字答案。",
     210, 2),
    ("工厂三天生产零件：第一天 120 个，第二天 150 个，第三天 180 个，平均每天多少个？请直接给出数字答案。",
     150, 2),
    ("一个数的平方等于 196，这个数是多少（取正数）？请直接给出数字答案。",
     14,  2),

    # 难度3
    ("鸡和兔共 20 只，腿共 56 条，有多少只兔？请直接给出数字答案。",
     8,   3),  # 鸡腿2x+兔腿4y=56, x+y=20 → y=8
    ("一项工程，甲单独做需 12 天，乙单独做需 18 天，两人合做需多少天？请直接给出数字答案。",
     # 1/12+1/18=5/36, 36/5=7.2
     # Actually let me use a cleaner answer
     # 1/12 + 1/18 = 3/36 + 2/36 = 5/36, so 36/5 = 7.2 days
     # Not a clean integer... let me change
     None, 3),  # skip this one
    ("水池有进水管和排水管，进水管单独注满需 6 小时，排水管单独排完需 8 小时，"
     "同时开启进水和排水，从空池开始，多少小时后注满？请直接给出数字答案。",
     24,  3),  # 1/6-1/8=1/24, 需24小时
    ("等差数列 2, 5, 8, 11, ... 第 20 项是多少？请直接给出数字答案。",
     59,  3),  # 2+(20-1)×3=59
    ("甲乙丙三人合伙投资，甲投 3 万，乙投 5 万，丙投 2 万，年利润 12 万元按投资比例分配，乙分到多少万元？请直接给出数字答案。",
     6,   3),  # 5/10×12=6
]
GEN_QUESTIONS = [(q, a, d) for q, a, d in GEN_QUESTIONS if a is not None]

GEN_TEMPLATE = "请解答以下数学题：\n{question}\n"

# ─── Part C：坐标几何多选题 ────────────────────────────────────────────────────
# (题干, A, B, C, D, 正确答案字母, 类别)
# 坐标用 (x,y) 格式，答案均为整数或简单小数，确保提取干净

GEO_MCQ_QUESTIONS = [
    # ── 坐标距离 ──────────────────────────────────────────────────────────────
    ("点 (0,0) 和点 (3,4) 之间的距离是多少？",
     "5",  "7",  "4",  "3",  "A", "坐标距离"),   # √(9+16)=5
    ("点 (1,1) 和点 (4,5) 之间的距离是多少？",
     "4",  "5",  "6",  "3",  "B", "坐标距离"),   # √(9+16)=5
    ("水平线段从 (2,3) 延伸到 (9,3)，长度是多少？",
     "5",  "6",  "8",  "7",  "D", "坐标距离"),   # 9-2=7
    ("竖直线段从 (5,2) 延伸到 (5,8)，长度是多少？",
     "6",  "5",  "4",  "3",  "A", "坐标距离"),   # 8-2=6
    ("点 (0,0) 和点 (8,6) 之间的距离是多少？",
     "14", "8",  "6",  "10", "D", "坐标距离"),   # √(64+36)=10

    # ── 平行与垂直 ─────────────────────────────────────────────────────────────
    ("过 (0,0) 和 (2,6) 的直线，其斜率是多少？",
     "3",  "2",  "6",  "0.5", "A", "平行垂直"),  # 6/2=3
    ("直线 L1 斜率为 2，直线 L2 斜率为 -1/2，两直线关系是？",
     "平行", "垂直", "重合", "相交但不垂直", "B", "平行垂直"),  # 2×(-1/2)=-1 → 垂直
    ("直线 L1 过 (0,0) 和 (1,3)，直线 L2 过 (0,2) 和 (1,5)，两直线关系是？",
     "垂直", "重合", "平行", "相交", "C", "平行垂直"),  # 斜率均为3，y截距不同 → 平行
    ("直线 L1 斜率为 4，直线 L2 与 L1 垂直，L2 的斜率是多少？",
     "4",  "-4",  "-1/4",  "1/4", "C", "平行垂直"),  # k1×k2=-1 → k2=-1/4
    ("过 (0,0) 和 (3,0) 的线段与过 (1,1) 和 (5,1) 的线段，两者关系是？",
     "垂直", "平行", "相交", "重合", "B", "平行垂直"),  # 两段都水平（斜率=0）→平行

    # ── 中点与坐标变换 ─────────────────────────────────────────────────────────
    ("点 A(2,4) 和点 B(8,10) 的中点坐标是？",
     "(5,7)", "(4,6)", "(6,8)", "(3,5)", "A", "中点变换"),  # ((2+8)/2,(4+10)/2)=(5,7)
    ("点 A(0,0) 和点 B(10,6) 中点的 x 坐标是多少？",
     "4",  "5",  "6",  "3",  "B", "中点变换"),   # 10/2=5
    ("从点 (3,2) 到点 (7,9) 的向量，y 方向分量是多少？",
     "4",  "9",  "7",  "6",  "C", "中点变换"),   # wait, 9-2=7 → C
    # Fix: 9-2=7 → C ✓
    ("矩形顶点为 (0,0),(8,0),(8,6),(0,6)，其中心坐标是？",
     "(4,3)", "(8,6)", "(3,4)", "(4,6)", "A", "中点变换"),  # (4,3)

    # ── 面积与周长 ─────────────────────────────────────────────────────────────
    ("矩形顶点为 (0,0),(5,0),(5,4),(0,4)，面积是多少？",
     "18", "20", "24", "16", "B", "面积周长"),   # 5×4=20
    ("直角三角形顶点为 (0,0),(6,0),(0,4)，面积是多少？",
     "10", "14", "16", "12", "D", "面积周长"),   # 0.5×6×4=12
    ("矩形顶点为 (1,1),(7,1),(7,5),(1,5)，周长是多少？",
     "20", "24", "16", "28", "A", "面积周长"),   # 2*(6+4)=20
    ("正方形顶点为 (0,0),(4,0),(4,4),(0,4)，面积是多少？",
     "8",  "12", "16", "4",  "C", "面积周长"),   # 4×4=16
]

GEO_MCQ_TEMPLATE = (
    "以下是一道坐标几何单项选择题，请直接给出正确选项字母。\n\n"
    "题目：{question}\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
    "答案："
)

# ─── Part D：坐标几何生成题 ────────────────────────────────────────────────────
# (题干, 标准答案数值, 类别)
GEO_GEN_QUESTIONS = [
    # 坐标距离
    ("点 A(0,0) 和点 B(5,12) 的距离是多少？请直接给出数字。",
     13, "坐标距离"),   # √(25+144)=13
    ("线段从 (3,2) 延伸到 (3,11)，长度是多少？请直接给出数字。",
     9,  "坐标距离"),   # 11-2=9
    ("点 (0,0) 和点 (6,8) 之间的距离是多少？请直接给出数字。",
     10, "坐标距离"),   # √(36+64)=10

    # 平行垂直
    ("直线过点 (0,0) 和点 (4,12)，斜率是多少？请直接给出数字。",
     3,  "平行垂直"),   # 12/4=3
    ("直线斜率为 5，与之垂直的直线斜率是多少（分数写成小数）？请直接给出数字。",
     -0.2, "平行垂直"), # -1/5=-0.2

    # 中点与变换
    ("点 A(2,3) 和点 B(14,9) 的中点 x 坐标是多少？请直接给出数字。",
     8,  "中点变换"),   # (2+14)/2=8
    ("从点 (1,4) 到点 (9,4)，线段中点的 x 坐标是多少？请直接给出数字。",
     5,  "中点变换"),   # (1+9)/2=5

    # 面积周长
    ("矩形顶点为 (0,0),(8,0),(8,6),(0,6)，周长是多少？请直接给出数字。",
     28, "面积周长"),   # 2*(8+6)=28
    ("直角三角形顶点为 (0,0),(4,0),(0,3)，面积是多少？请直接给出数字。",
     6,  "面积周长"),   # 0.5*4*3=6
    ("正方形边长为 7（顶点坐标整数），面积是多少？请直接给出数字。",
     49, "面积周长"),   # 7×7=49
]

GEO_GEN_TEMPLATE = "请解答以下坐标几何题：\n{question}\n"

# ─── 模型加载 ──────────────────────────────────────────────────────────────────

def load_untrained_model():
    print("加载 untrained 模型...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        use_cache=False,
    )
    initialize_token_embedding(model, tokenizer)
    model.eval()
    return tokenizer, model

def load_trained_model():
    print("加载 trained 模型...")
    model = AutoModelForCausalLM.from_pretrained(
        TRAINED_PATH, device_map="auto", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(TRAINED_PATH)
    model.eval()
    return tokenizer, model

# ─── Part A：多选题评估 ────────────────────────────────────────────────────────

def get_answer_token_ids(tokenizer):
    ids = {}
    for letter in ["A", "B", "C", "D"]:
        enc = tokenizer.encode(letter, add_special_tokens=False)
        assert len(enc) == 1, f"字母 {letter} 被分成了多个 token: {enc}"
        ids[letter] = enc[0]
    return ids


def score_mcq(model, tokenizer, answer_token_ids, question, a, b, c, d):
    prompt = MCQ_TEMPLATE.format(question=question, a=a, b=b, c=c, d=d)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]
    choice_logits = {l: logits[tid].item() for l, tid in answer_token_ids.items()}
    logit_vals = torch.tensor([choice_logits[l] for l in ["A", "B", "C", "D"]])
    probs = torch.softmax(logit_vals, dim=0).tolist()
    choice_probs = {l: probs[i] for i, l in enumerate(["A", "B", "C", "D"])}
    predicted = max(choice_logits, key=choice_logits.get)
    sorted_logits = sorted(choice_logits.values(), reverse=True)
    margin = sorted_logits[0] - sorted_logits[1]
    return predicted, choice_probs, margin

def evaluate_mcq(model, tokenizer, label):
    answer_token_ids = get_answer_token_ids(tokenizer)
    records = []
    print(f"\n{'='*60}\n[Part A 多选题] 评估: {label}\n{'='*60}")
    for i, entry in enumerate(MCQ_QUESTIONS):
        question, a, b, c, d, correct, difficulty = entry
        predicted, probs, margin = score_mcq(
            model, tokenizer, answer_token_ids, question, a, b, c, d
        )
        is_correct = (predicted == correct)
        records.append({
            "idx": i, "difficulty": difficulty, "question": question,
            "correct": correct, "predicted": predicted,
            "is_correct": is_correct, "correct_prob": probs[correct],
            "margin": margin,
        })
        status = "✓" if is_correct else "✗"
        print(f"  [{i+1:02d}] D{difficulty} {status} pred={predicted} ans={correct} "
              f"p={probs[correct]:.3f} margin={margin:.2f}  {question[:35]}")
    return records

# ─── Part B：生成式评估 ────────────────────────────────────────────────────────
def extract_number(text: str):
    """从生成文本中提取最后出现的数字（整数或小数）"""
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    return float(matches[-1])

def evaluate_gen(model, tokenizer, label, max_new_tokens=80):
    records = []
    print(f"\n{'='*60}\n[Part B 生成题] 评估: {label}\n{'='*60}")
    for i, (question, answer, difficulty) in enumerate(GEN_QUESTIONS):
        prompt = GEN_TEMPLATE.format(question=question)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        prompt_len = input_ids.shape[1]
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_ids = gen_ids[0, prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        pred_num = extract_number(generated_text)
        is_correct = (pred_num is not None and abs(pred_num - answer) < 0.5)
        records.append({
            "idx": i, "difficulty": difficulty, "question": question,
            "correct_answer": answer, "predicted_num": pred_num,
            "is_correct": is_correct, "generated_text": generated_text,
            "gen_len": len(generated_ids),
        })
        status = "✓" if is_correct else "✗"
        print(f"  [{i+1:02d}] D{difficulty} {status} pred={pred_num} ans={answer}  {question[:40]}")
        print(f"        生成: {generated_text[:80].strip()!r}")
    return records

# ─── Part C：坐标几何多选评估 ─────────────────────────────────────────────────

def evaluate_geo_mcq(model, tokenizer, label):
    answer_token_ids = get_answer_token_ids(tokenizer)
    records = []
    print(f"\n{'='*60}\n[Part C 几何多选] 评估: {label}\n{'='*60}")
    for i, (question, a, b, c, d, correct, category) in enumerate(GEO_MCQ_QUESTIONS):
        prompt = GEO_MCQ_TEMPLATE.format(question=question, a=a, b=b, c=c, d=d)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, :]
        choice_logits = {l: logits[tid].item() for l, tid in answer_token_ids.items()}
        logit_vals = torch.tensor([choice_logits[l] for l in ["A", "B", "C", "D"]])
        probs = torch.softmax(logit_vals, dim=0).tolist()
        choice_probs = {l: probs[i] for i, l in enumerate(["A", "B", "C", "D"])}
        predicted = max(choice_logits, key=choice_logits.get)
        sorted_logits = sorted(choice_logits.values(), reverse=True)
        margin = sorted_logits[0] - sorted_logits[1]
        is_correct = (predicted == correct)
        records.append({
            "idx": i, "category": category, "question": question,
            "correct": correct, "predicted": predicted,
            "is_correct": is_correct, "correct_prob": choice_probs[correct],
            "margin": margin,
        })
        status = "✓" if is_correct else "✗"
        print(f"  [{i+1:02d}] {status} [{category}] pred={predicted} ans={correct} "
              f"p={choice_probs[correct]:.3f} margin={margin:.2f}  {question[:40]}")
    return records


def evaluate_geo_gen(model, tokenizer, label, max_new_tokens=80):
    records = []
    print(f"\n{'='*60}\n[Part D 几何生成] 评估: {label}\n{'='*60}")
    for i, (question, answer, category) in enumerate(GEO_GEN_QUESTIONS):
        prompt = GEO_GEN_TEMPLATE.format(question=question)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        prompt_len = input_ids.shape[1]
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_ids = gen_ids[0, prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        pred_num = extract_number(generated_text)
        is_correct = (pred_num is not None and abs(pred_num - answer) < 0.5)
        records.append({
            "idx": i, "category": category, "question": question,
            "correct_answer": answer, "predicted_num": pred_num,
            "is_correct": is_correct, "generated_text": generated_text,
            "gen_len": len(generated_ids),
        })
        status = "✓" if is_correct else "✗"
        print(f"  [{i+1:02d}] {status} [{category}] pred={pred_num} ans={answer}  {question[:40]}")
        print(f"        生成: {generated_text[:80].strip()!r}")
    return records

# ─── 统计 ──────────────────────────────────────────────────────────────────────

def compute_stats_mcq(records):
    total = len(records)
    acc = sum(r["is_correct"] for r in records) / total
    by_diff = {}
    for d in [1, 2, 3]:
        recs = [r for r in records if r["difficulty"] == d]
        if recs:
            by_diff[d] = {
                "acc": sum(r["is_correct"] for r in recs) / len(recs),
                "correct_probs": [r["correct_prob"] for r in recs],
                "margins": [r["margin"] for r in recs],
                "n": len(recs),
            }
    return {"acc": acc, "total": total,
            "correct_count": sum(r["is_correct"] for r in records),
            "correct_probs": [r["correct_prob"] for r in records],
            "margins": [r["margin"] for r in records],
            "by_difficulty": by_diff}


def compute_stats_gen(records):
    total = len(records)
    acc = sum(r["is_correct"] for r in records) / total
    by_diff = {}
    for d in [1, 2, 3]:
        recs = [r for r in records if r["difficulty"] == d]
        if recs:
            by_diff[d] = {
                "acc": sum(r["is_correct"] for r in recs) / len(recs),
                "avg_gen_len": np.mean([r["gen_len"] for r in recs]),
                "n": len(recs),
            }
    return {"acc": acc, "total": total,
            "correct_count": sum(r["is_correct"] for r in records),
            "avg_gen_len": np.mean([r["gen_len"] for r in records]),
            "by_difficulty": by_diff}


def compute_stats_geo_mcq(records):
    total = len(records)
    acc = sum(r["is_correct"] for r in records) / total
    cats = sorted(set(r["category"] for r in records))
    by_cat = {}
    for cat in cats:
        recs = [r for r in records if r["category"] == cat]
        by_cat[cat] = {
            "acc": sum(r["is_correct"] for r in recs) / len(recs),
            "correct_probs": [r["correct_prob"] for r in recs],
            "margins": [r["margin"] for r in recs],
            "n": len(recs),
        }
    return {"acc": acc, "total": total,
            "correct_count": sum(r["is_correct"] for r in records),
            "correct_probs": [r["correct_prob"] for r in records],
            "margins": [r["margin"] for r in records],
            "by_category": by_cat}


def compute_stats_geo_gen(records):
    total = len(records)
    acc = sum(r["is_correct"] for r in records) / total
    cats = sorted(set(r["category"] for r in records))
    by_cat = {}
    for cat in cats:
        recs = [r for r in records if r["category"] == cat]
        by_cat[cat] = {
            "acc": sum(r["is_correct"] for r in recs) / len(recs),
            "n": len(recs),
        }
    return {"acc": acc, "total": total,
            "correct_count": sum(r["is_correct"] for r in records),
            "by_category": by_cat}

# ─── 可视化 ───────────────────────────────────────────────────────────────────

def plot_all(mcq_ut, mcq_tr, gen_ut, gen_tr, s_mcq_ut, s_mcq_tr, s_gen_ut, s_gen_tr):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"实验一：数学推理 OOD 测试  ({SAVE_ID})", fontsize=14)

    # ── 1. MCQ 按难度准确率 ───────────────────────────────────────────────────
    ax = axes[0, 0]
    diffs = [1, 2, 3]
    diff_labels = ["D1 基础", "D2 中等", "D3 综合"]
    acc_ut = [s_mcq_ut["by_difficulty"].get(d, {}).get("acc", 0) for d in diffs]
    acc_tr = [s_mcq_tr["by_difficulty"].get(d, {}).get("acc", 0) for d in diffs]
    x = np.arange(len(diffs))
    w = 0.35
    b1 = ax.bar(x - w/2, acc_ut, w, label="Untrained", color="#5B9BD5", alpha=0.85)
    b2 = ax.bar(x + w/2, acc_tr, w, label="Trained",   color="#ED7D31", alpha=0.85)
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="随机基线")
    ax.set_xticks(x); ax.set_xticklabels(diff_labels, fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_ylabel("准确率"); ax.set_title("Part A 多选题 — 按难度准确率")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    for b in list(b1) + list(b2):
        h = b.get_height()
        if h > 0.02:
            ax.text(b.get_x() + b.get_width()/2, h + 0.02, f"{h:.0%}",
                    ha="center", va="bottom", fontsize=8)

    # ── 2. MCQ 总体准确率 ─────────────────────────────────────────────────────
    ax = axes[0, 1]
    vals = [s_mcq_ut["acc"], s_mcq_tr["acc"]]
    bars = ax.bar(["Untrained", "Trained"], vals, color=["#5B9BD5", "#ED7D31"], alpha=0.85, width=0.4)
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="随机基线")
    ax.set_ylim(0, 1.1); ax.set_ylabel("准确率")
    ax.set_title(f"Part A 多选题总体准确率\n"
                 f"Ut:{s_mcq_ut['correct_count']}/{s_mcq_ut['total']}  "
                 f"Tr:{s_mcq_tr['correct_count']}/{s_mcq_tr['total']}")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.1%}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    # ── 3. MCQ P(正确答案) 分布 ───────────────────────────────────────────────
    ax = axes[0, 2]
    parts = ax.violinplot(
        [s_mcq_ut["correct_probs"], s_mcq_tr["correct_probs"]],
        positions=[1, 2], showmedians=True, showextrema=True,
    )
    for pc, c in zip(parts["bodies"], ["#5B9BD5", "#ED7D31"]):
        pc.set_facecolor(c); pc.set_alpha(0.7)
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Untrained", "Trained"])
    ax.set_ylabel("P(正确答案)"); ax.set_title("Part A 正确答案 softmax 概率分布")
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    for pos, probs in zip([1, 2], [s_mcq_ut["correct_probs"], s_mcq_tr["correct_probs"]]):
        ax.text(pos, 0.02, f"μ={np.mean(probs):.3f}", ha="center", fontsize=9)

    # ── 4. Gen 按难度准确率 ───────────────────────────────────────────────────
    ax = axes[1, 0]
    diffs_g = sorted(set(r["difficulty"] for r in gen_ut))
    diff_g_labels = {1: "D1 基础", 2: "D2 中等", 3: "D3 综合"}
    acc_g_ut = [s_gen_ut["by_difficulty"].get(d, {}).get("acc", 0) for d in diffs_g]
    acc_g_tr = [s_gen_tr["by_difficulty"].get(d, {}).get("acc", 0) for d in diffs_g]
    x = np.arange(len(diffs_g))
    b1 = ax.bar(x - w/2, acc_g_ut, w, label="Untrained", color="#5B9BD5", alpha=0.85)
    b2 = ax.bar(x + w/2, acc_g_tr, w, label="Trained",   color="#ED7D31", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([diff_g_labels[d] for d in diffs_g], fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_ylabel("准确率"); ax.set_title("Part B 生成题 — 按难度准确率")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    for b in list(b1) + list(b2):
        h = b.get_height()
        if h > 0.02:
            ax.text(b.get_x() + b.get_width()/2, h + 0.02, f"{h:.0%}",
                    ha="center", va="bottom", fontsize=8)

    # ── 5. Gen 总体准确率 ─────────────────────────────────────────────────────
    ax = axes[1, 1]
    vals_g = [s_gen_ut["acc"], s_gen_tr["acc"]]
    bars_g = ax.bar(["Untrained", "Trained"], vals_g,
                    color=["#5B9BD5", "#ED7D31"], alpha=0.85, width=0.4)
    ax.set_ylim(0, 1.1); ax.set_ylabel("准确率")
    ax.set_title(f"Part B 生成题总体准确率\n"
                 f"Ut:{s_gen_ut['correct_count']}/{s_gen_ut['total']}  "
                 f"Tr:{s_gen_tr['correct_count']}/{s_gen_tr['total']}")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars_g, vals_g):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.1%}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    # ── 6. MCQ logit margin 分布 ──────────────────────────────────────────────
    ax = axes[1, 2]
    parts2 = ax.violinplot(
        [s_mcq_ut["margins"], s_mcq_tr["margins"]],
        positions=[1, 2], showmedians=True, showextrema=True,
    )
    for pc, c in zip(parts2["bodies"], ["#5B9BD5", "#ED7D31"]):
        pc.set_facecolor(c); pc.set_alpha(0.7)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Untrained", "Trained"])
    ax.set_ylabel("Logit 边距（top1 − top2）"); ax.set_title("Part A 决策置信度（logit margin）")
    ax.grid(axis="y", alpha=0.3)
    for pos, margins in zip([1, 2], [s_mcq_ut["margins"], s_mcq_tr["margins"]]):
        y0 = min(s_mcq_ut["margins"] + s_mcq_tr["margins"])
        ax.text(pos, y0 - 0.3, f"μ={np.mean(margins):.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{SAVE_ID}_ood_math.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n保存图表: {path}")


def plot_geometry(geo_mcq_ut, geo_mcq_tr, geo_gen_ut, geo_gen_tr,
                  s_geo_mcq_ut, s_geo_mcq_tr, s_geo_gen_ut, s_geo_gen_tr,
                  s_mcq_ut, s_mcq_tr, s_gen_ut, s_gen_tr):
    cats = list(s_geo_mcq_ut["by_category"].keys())
    x = np.arange(len(cats))
    w = 0.35

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"实验一 Part C/D：坐标几何推理测试  ({SAVE_ID})", fontsize=14)

    # ── 1. Part C MCQ 按类别准确率 ────────────────────────────────────────────
    ax = axes[0, 0]
    acc_ut = [s_geo_mcq_ut["by_category"][c]["acc"] for c in cats]
    acc_tr = [s_geo_mcq_tr["by_category"][c]["acc"] for c in cats]
    b1 = ax.bar(x - w/2, acc_ut, w, label="Untrained", color="#5B9BD5", alpha=0.85)
    b2 = ax.bar(x + w/2, acc_tr, w, label="Trained",   color="#ED7D31", alpha=0.85)
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="随机基线")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9, rotation=12)
    ax.set_ylim(0, 1.15); ax.set_ylabel("准确率")
    ax.set_title("Part C 几何多选 — 按类别准确率")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    for b in list(b1) + list(b2):
        h = b.get_height()
        if h > 0.02:
            ax.text(b.get_x() + b.get_width()/2, h + 0.02, f"{h:.0%}",
                    ha="center", va="bottom", fontsize=8)

    # ── 2. Part C MCQ 总体 + Part D Gen 总体 ─────────────────────────────────
    ax = axes[0, 1]
    labels2 = ["C-MCQ\nUntrained", "C-MCQ\nTrained", "D-Gen\nUntrained", "D-Gen\nTrained"]
    vals2   = [s_geo_mcq_ut["acc"], s_geo_mcq_tr["acc"],
               s_geo_gen_ut["acc"], s_geo_gen_tr["acc"]]
    colors2 = ["#5B9BD5", "#ED7D31", "#70AD47", "#FFC000"]
    bars2   = ax.bar(labels2, vals2, color=colors2, alpha=0.85, width=0.5)
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_ylim(0, 1.15); ax.set_ylabel("准确率")
    ax.set_title("几何题总体准确率（C/D 两部分）")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars2, vals2):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.1%}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # ── 3. Part C MCQ P(正确答案) violin ─────────────────────────────────────
    ax = axes[0, 2]
    parts = ax.violinplot(
        [s_geo_mcq_ut["correct_probs"], s_geo_mcq_tr["correct_probs"]],
        positions=[1, 2], showmedians=True, showextrema=True,
    )
    for pc, c in zip(parts["bodies"], ["#5B9BD5", "#ED7D31"]):
        pc.set_facecolor(c); pc.set_alpha(0.7)
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Untrained", "Trained"])
    ax.set_ylabel("P(正确答案)"); ax.set_title("Part C 正确答案 softmax 概率分布")
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    for pos, probs in zip([1, 2], [s_geo_mcq_ut["correct_probs"], s_geo_mcq_tr["correct_probs"]]):
        ax.text(pos, 0.02, f"μ={np.mean(probs):.3f}", ha="center", fontsize=9)

    # ── 4. Part D Gen 按类别准确率 ────────────────────────────────────────────
    ax = axes[1, 0]
    gen_cats = list(s_geo_gen_ut["by_category"].keys())
    xg = np.arange(len(gen_cats))
    gv_ut = [s_geo_gen_ut["by_category"][c]["acc"] for c in gen_cats]
    gv_tr = [s_geo_gen_tr["by_category"][c]["acc"] for c in gen_cats]
    bg1 = ax.bar(xg - w/2, gv_ut, w, label="Untrained", color="#5B9BD5", alpha=0.85)
    bg2 = ax.bar(xg + w/2, gv_tr, w, label="Trained",   color="#ED7D31", alpha=0.85)
    ax.set_xticks(xg); ax.set_xticklabels(gen_cats, fontsize=9, rotation=12)
    ax.set_ylim(0, 1.15); ax.set_ylabel("准确率")
    ax.set_title("Part D 几何生成 — 按类别准确率")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    for b in list(bg1) + list(bg2):
        h = b.get_height()
        if h > 0.02:
            ax.text(b.get_x() + b.get_width()/2, h + 0.02, f"{h:.0%}",
                    ha="center", va="bottom", fontsize=8)

    # ── 5. 全实验横向对比（A/B/C/D 四部分总体准确率）────────────────────────
    ax = axes[1, 1]
    part_labels = ["A-MCQ\n数学多选", "B-Gen\n数学生成", "C-MCQ\n几何多选", "D-Gen\n几何生成"]
    ut_accs = [s_mcq_ut["acc"], s_gen_ut["acc"], s_geo_mcq_ut["acc"], s_geo_gen_ut["acc"]]
    tr_accs = [s_mcq_tr["acc"], s_gen_tr["acc"], s_geo_mcq_tr["acc"], s_geo_gen_tr["acc"]]
    xp = np.arange(len(part_labels))
    ax.bar(xp - w/2, ut_accs, w, label="Untrained", color="#5B9BD5", alpha=0.85)
    ax.bar(xp + w/2, tr_accs, w, label="Trained",   color="#ED7D31", alpha=0.85)
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="随机基线")
    ax.set_xticks(xp); ax.set_xticklabels(part_labels, fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_ylabel("准确率")
    ax.set_title("实验一全局：四部分准确率对比")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    for xv, ut, tr in zip(xp, ut_accs, tr_accs):
        ax.text(xv - w/2, ut + 0.02, f"{ut:.0%}", ha="center", va="bottom", fontsize=8)
        ax.text(xv + w/2, tr + 0.02, f"{tr:.0%}", ha="center", va="bottom", fontsize=8)

    # ── 6. Part C logit margin violin ────────────────────────────────────────
    ax = axes[1, 2]
    parts2 = ax.violinplot(
        [s_geo_mcq_ut["margins"], s_geo_mcq_tr["margins"]],
        positions=[1, 2], showmedians=True, showextrema=True,
    )
    for pc, c in zip(parts2["bodies"], ["#5B9BD5", "#ED7D31"]):
        pc.set_facecolor(c); pc.set_alpha(0.7)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Untrained", "Trained"])
    ax.set_ylabel("Logit 边距（top1 − top2）")
    ax.set_title("Part C 几何多选决策置信度")
    ax.grid(axis="y", alpha=0.3)
    for pos, margins in zip([1, 2], [s_geo_mcq_ut["margins"], s_geo_mcq_tr["margins"]]):
        y0 = min(s_geo_mcq_ut["margins"] + s_geo_mcq_tr["margins"])
        ax.text(pos, y0 - 0.3, f"μ={np.mean(margins):.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{SAVE_ID}_geo_reasoning.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n保存图表: {path}")


def plot_per_question_scatter(mcq_ut, mcq_tr):
    probs_ut = [r["correct_prob"] for r in mcq_ut]
    probs_tr = [r["correct_prob"] for r in mcq_tr]
    diff_colors = {1: "#2196F3", 2: "#FF9800", 3: "#F44336"}
    colors = [diff_colors[r["difficulty"]] for r in mcq_ut]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(probs_ut, probs_tr, c=colors, s=70, alpha=0.85,
               edgecolors="white", linewidths=0.5)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="y=x")
    ax.axhline(0.25, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(0.25, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=diff_colors[d], label=f"D{d}") for d in [1, 2, 3]]
    legend_els.append(plt.Line2D([0], [0], linestyle="--", color="black", label="y=x"))
    ax.legend(handles=legend_els, fontsize=9)

    ax.set_xlabel("Untrained  P(正确答案)", fontsize=11)
    ax.set_ylabel("Trained  P(正确答案)", fontsize=11)
    ax.set_title("每题正确答案概率散点\n（点在对角线上方 → Trained 置信度更高）", fontsize=11)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{SAVE_ID}_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存图表: {path}")

# ─── 保存 & 摘要 ──────────────────────────────────────────────────────────────

def save_json(mcq_ut, mcq_tr, gen_ut, gen_tr,
              geo_mcq_ut, geo_mcq_tr, geo_gen_ut, geo_gen_tr,
              s_mcq_ut, s_mcq_tr, s_gen_ut, s_gen_tr,
              s_geo_mcq_ut, s_geo_mcq_tr, s_geo_gen_ut, s_geo_gen_tr):
    out = {
        "partA_math_mcq": {
            "untrained": {"acc": s_mcq_ut["acc"], "total": s_mcq_ut["total"],
                          "correct": s_mcq_ut["correct_count"],
                          "by_difficulty": {str(k): v["acc"] for k, v in s_mcq_ut["by_difficulty"].items()},
                          "records": mcq_ut},
            "trained":   {"acc": s_mcq_tr["acc"], "total": s_mcq_tr["total"],
                          "correct": s_mcq_tr["correct_count"],
                          "by_difficulty": {str(k): v["acc"] for k, v in s_mcq_tr["by_difficulty"].items()},
                          "records": mcq_tr},
        },
        "partB_math_gen": {
            "untrained": {"acc": s_gen_ut["acc"], "total": s_gen_ut["total"],
                          "correct": s_gen_ut["correct_count"],
                          "by_difficulty": {str(k): v["acc"] for k, v in s_gen_ut["by_difficulty"].items()},
                          "records": gen_ut},
            "trained":   {"acc": s_gen_tr["acc"], "total": s_gen_tr["total"],
                          "correct": s_gen_tr["correct_count"],
                          "by_difficulty": {str(k): v["acc"] for k, v in s_gen_tr["by_difficulty"].items()},
                          "records": gen_tr},
        },
        "partC_geo_mcq": {
            "untrained": {"acc": s_geo_mcq_ut["acc"], "total": s_geo_mcq_ut["total"],
                          "correct": s_geo_mcq_ut["correct_count"],
                          "by_category": {k: v["acc"] for k, v in s_geo_mcq_ut["by_category"].items()},
                          "records": geo_mcq_ut},
            "trained":   {"acc": s_geo_mcq_tr["acc"], "total": s_geo_mcq_tr["total"],
                          "correct": s_geo_mcq_tr["correct_count"],
                          "by_category": {k: v["acc"] for k, v in s_geo_mcq_tr["by_category"].items()},
                          "records": geo_mcq_tr},
        },
        "partD_geo_gen": {
            "untrained": {"acc": s_geo_gen_ut["acc"], "total": s_geo_gen_ut["total"],
                          "correct": s_geo_gen_ut["correct_count"],
                          "by_category": {k: v["acc"] for k, v in s_geo_gen_ut["by_category"].items()},
                          "records": geo_gen_ut},
            "trained":   {"acc": s_geo_gen_tr["acc"], "total": s_geo_gen_tr["total"],
                          "correct": s_geo_gen_tr["correct_count"],
                          "by_category": {k: v["acc"] for k, v in s_geo_gen_tr["by_category"].items()},
                          "records": geo_gen_tr},
        },
    }
    path = os.path.join(OUTPUT_DIR, f"{SAVE_ID}_ood_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"保存结果: {path}")


def print_summary(s_mcq_ut, s_mcq_tr, s_gen_ut, s_gen_tr,
                  s_geo_mcq_ut, s_geo_mcq_tr, s_geo_gen_ut, s_geo_gen_tr):
    print("\n" + "=" * 60)
    print("实验一  数学推理 + 坐标几何推理  结果摘要")
    print("=" * 60)

    print("\n── Part A/B 数学推理 ──")
    print(f"{'':12} {'总体':>6}  D1      D2      D3")
    print("-" * 50)
    for label, s in [("MCQ Untr.", s_mcq_ut), ("MCQ Train.", s_mcq_tr)]:
        d_accs = "  ".join(
            f"{s['by_difficulty'].get(d, {}).get('acc', float('nan')):.0%}" for d in [1, 2, 3]
        )
        print(f"{label:<12} {s['acc']:>6.1%}  {d_accs}")
    print("-" * 50)
    for label, s in [("Gen Untr.", s_gen_ut), ("Gen Train.", s_gen_tr)]:
        d_accs = "  ".join(
            f"{s['by_difficulty'].get(d, {}).get('acc', float('nan')):.0%}" for d in [1, 2, 3]
        )
        print(f"{label:<12} {s['acc']:>6.1%}  {d_accs}")

    print("\n── Part C/D 坐标几何 ──")
    geo_cats = ["坐标距离", "平行垂直", "中点变换", "面积周长"]
    header = "  ".join(f"{c[:4]:>5}" for c in geo_cats)
    print(f"{'':12} {'总体':>6}  {header}")
    print("-" * 60)
    for label, s in [("GeoMCQ Ut.", s_geo_mcq_ut), ("GeoMCQ Tr.", s_geo_mcq_tr)]:
        cat_accs = "  ".join(
            f"{s['by_category'].get(c, {}).get('acc', float('nan')):>5.0%}" for c in geo_cats
        )
        print(f"{label:<12} {s['acc']:>6.1%}  {cat_accs}")
    print("-" * 60)
    gen_cats = ["坐标距离", "平行垂直", "中点变换", "面积周长"]
    for label, s in [("GeoGen Ut.", s_geo_gen_ut), ("GeoGen Tr.", s_geo_gen_tr)]:
        cat_accs = "  ".join(
            f"{s['by_category'].get(c, {}).get('acc', float('nan')):>5.0%}" for c in gen_cats
        )
        print(f"{label:<12} {s['acc']:>6.1%}  {cat_accs}")

    print("\n── Δ（Trained − Untrained）──")
    for tag, tr, ut in [
        ("A-MCQ", s_mcq_tr["acc"], s_mcq_ut["acc"]),
        ("B-Gen", s_gen_tr["acc"], s_gen_ut["acc"]),
        ("C-GeoMCQ", s_geo_mcq_tr["acc"], s_geo_mcq_ut["acc"]),
        ("D-GeoGen", s_geo_gen_tr["acc"], s_geo_gen_ut["acc"]),
    ]:
        d = tr - ut
        print(f"  {tag:<10} {'+' if d>=0 else ''}{d:.1%}")
    print("=" * 60)

# ─── 主流程 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Part A 数学多选题: {len(MCQ_QUESTIONS)} 题")
    print(f"Part B 数学生成题: {len(GEN_QUESTIONS)} 题")
    print(f"Part C 几何多选题: {len(GEO_MCQ_QUESTIONS)} 题  "
          f"类别: {sorted(set(q[6] for q in GEO_MCQ_QUESTIONS))}")
    print(f"Part D 几何生成题: {len(GEO_GEN_QUESTIONS)} 题")

    ut_tokenizer, ut_model = load_untrained_model()
    tr_tokenizer, tr_model = load_trained_model()

    # ── Part A/B 数学推理 ─────────────────────────────────────────────────────
    mcq_ut = evaluate_mcq(ut_model, ut_tokenizer, "Untrained")
    mcq_tr = evaluate_mcq(tr_model, tr_tokenizer, "Trained")
    gen_ut = evaluate_gen(ut_model, ut_tokenizer, "Untrained")
    gen_tr = evaluate_gen(tr_model, tr_tokenizer, "Trained")

    # ── Part C/D 几何推理 ─────────────────────────────────────────────────────
    geo_mcq_ut = evaluate_geo_mcq(ut_model, ut_tokenizer, "Untrained")
    geo_mcq_tr = evaluate_geo_mcq(tr_model, tr_tokenizer, "Trained")
    geo_gen_ut = evaluate_geo_gen(ut_model, ut_tokenizer, "Untrained")
    geo_gen_tr = evaluate_geo_gen(tr_model, tr_tokenizer, "Trained")

    # ── 统计 ──────────────────────────────────────────────────────────────────
    s_mcq_ut = compute_stats_mcq(mcq_ut)
    s_mcq_tr = compute_stats_mcq(mcq_tr)
    s_gen_ut = compute_stats_gen(gen_ut)
    s_gen_tr = compute_stats_gen(gen_tr)
    s_geo_mcq_ut = compute_stats_geo_mcq(geo_mcq_ut)
    s_geo_mcq_tr = compute_stats_geo_mcq(geo_mcq_tr)
    s_geo_gen_ut = compute_stats_geo_gen(geo_gen_ut)
    s_geo_gen_tr = compute_stats_geo_gen(geo_gen_tr)

    # ── 输出 ──────────────────────────────────────────────────────────────────
    print_summary(s_mcq_ut, s_mcq_tr, s_gen_ut, s_gen_tr,
                  s_geo_mcq_ut, s_geo_mcq_tr, s_geo_gen_ut, s_geo_gen_tr)
    plot_all(mcq_ut, mcq_tr, gen_ut, gen_tr, s_mcq_ut, s_mcq_tr, s_gen_ut, s_gen_tr)
    plot_geometry(geo_mcq_ut, geo_mcq_tr, geo_gen_ut, geo_gen_tr,
                  s_geo_mcq_ut, s_geo_mcq_tr, s_geo_gen_ut, s_geo_gen_tr,
                  s_mcq_ut, s_mcq_tr, s_gen_ut, s_gen_tr)
    plot_per_question_scatter(mcq_ut, mcq_tr)
    save_json(mcq_ut, mcq_tr, gen_ut, gen_tr,
              geo_mcq_ut, geo_mcq_tr, geo_gen_ut, geo_gen_tr,
              s_mcq_ut, s_mcq_tr, s_gen_ut, s_gen_tr,
              s_geo_mcq_ut, s_geo_mcq_tr, s_geo_gen_ut, s_geo_gen_tr)

    print(f"\n全部结果已保存至: {OUTPUT_DIR}")
