import json

name="90017-architectural_human_labeled_Rude_labeled_V3"
input_path="train_json_data/five_plus_two_train_jsonl_data/design_3.27/grpo_phase_1_train_data/train_set_inbox_force_prompt.jsonl"
output_path=f"train_json_data/five_plus_two_train_jsonl_data/design_3.27/grpo_phase_1_train_data/overfitting_data/train_set_inbox_force_prompt_overfitting({name}).jsonl"
with open(input_path, "r", encoding="utf-8") as fin:
    with open(output_path, "a", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            record = json.loads(line)
            house = record["house"]
            if house==name:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")