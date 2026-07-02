import json
import re

input_path="five_plus_two_optimization/train_model_new/GRPO/test/test_jsonl/grpo_fpt_new_6/grpo_fpt_new_6_20000_step.jsonl"

with open(input_path, "r", encoding="utf-8") as f:
    for i,line in enumerate(f):
        print(i)
        record=json.loads(line)
        response_list=record["response"]
        response=response_list[0]
        pattern = r"<(add|remove)><(beam|shearwall)>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
        for m in re.finditer(pattern, response):
            act_type,seg_type,x1, y1, x2, y2 = m.groups() #m代表一个捕获pattern
            text=f"<{act_type}><{seg_type}>({x1},{y1}),({x2},{y2})"
            response=''.join(response.rsplit(text,1))
        print(response)