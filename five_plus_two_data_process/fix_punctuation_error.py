import json
import re
import sys

# 匹配 <beam>(x1,y1)(x2,y2) 或 <beams>(x1,y1)(x2,y2)（两组坐标之间缺少逗号）
PATTERN = re.compile(r"(<(?:beam|shearwall)>)\(([^)]+)\)\(([^)]+)\)")

def fix_string(s: str) -> str:
    return PATTERN.sub(r"\1(\2),(\3)", s)

def fix_value(v):
    if isinstance(v, str):
        return fix_string(v)
    if isinstance(v, dict):
        return {k: fix_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [fix_value(item) for item in v]
    return v

def fix_jsonl(input_path: str, output_path: str):
    lines_out = []
    total = 0
    changed = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.rstrip("\n")
            if not stripped.strip():
                lines_out.append(stripped)
                continue
            record = json.loads(stripped)
            fixed = fix_value(record)
            total += 1
            if fixed != record:
                changed += 1
            lines_out.append(json.dumps(fixed, ensure_ascii=False))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))
        if lines_out:
            f.write("\n")

    print(f"共 {total} 条记录，修改了 {changed} 条")
    print(f"输出: {output_path}")

if __name__ == "__main__":
    input_path="train_json_data/five_plus_two_train_jsonl_data/design_3.27/base_model_train/test_set_100_actionized_sort_floor_force_auged(3_times).jsonl"
    output_path="train_json_data/five_plus_two_train_jsonl_data/design_3.27/base_model_train/test_set_100_actionized_sort_floor_force_auged(3_times)_fixed.jsonl"

    fix_jsonl(input_path, output_path)
