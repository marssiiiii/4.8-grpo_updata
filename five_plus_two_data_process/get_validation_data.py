import json
import numpy as np
TRAIN_NUM,TEST_NUM=13613,4803
#TRAIN_NUM,TEST_NUM=14651,3295
SAMPLE_NUM=200
train_sample_points = np.linspace(0, TRAIN_NUM-1, SAMPLE_NUM)
train_sample_points = np.round(train_sample_points).astype(int)
test_sample_points = np.linspace(0,TEST_NUM-1,SAMPLE_NUM)
test_sample_points = np.round(test_sample_points).astype(int)

input_path="train_json_data/five_plus_two_train_jsonl_data/design_3.27/grpo_phase_1_train_data/train_set_inbox_force_prompt.jsonl"
output_path=f"train_json_data/five_plus_two_train_jsonl_data/design_3.27/validation_data/grpo_train_200.jsonl"

len=0
with open(input_path, "r", encoding="utf-8") as fin:
    with open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i in test_sample_points:
                record = json.loads(line)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                len+=1
print(len)