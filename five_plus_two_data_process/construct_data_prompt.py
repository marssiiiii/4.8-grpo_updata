import json
import sys
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/train_model_new/base_code")
from reward_design_trunc import process_completion_predict_filt #process_completion_predict_filt(context, completion_predict, response)

def construct_prompt(context,completion_predict):
    prompt=""
    prompt=prompt+"Making actions based on context and given structures."
    prompt=prompt+f"context:{context}"
    prompt=prompt+f"structures:{completion_predict}"
    prompt=f"<s>{prompt}</s>"
    return prompt

input_path="design_3.27/initial_data_100_test_set_inbox_shf.jsonl"
output_path="design_3.27/grpo_phase_1_train_data/initial_data_100_test_set_inbox_shf_prompt_filt_valid.jsonl"

with open(input_path,"r",encoding="utf-8") as fin,\
open(output_path, "w", encoding="utf-8") as fout:
    for i,line in enumerate(fin):
        if i%100==0:
            print(i)
        record = json.loads(line) #依据输入构造数据
        house,floor,design,bound,context,completion_predict = record["house"],record["floor"],record["design"],\
        record["bound"],record["context"],record["completion_predict"]
        completion_predict_filt=process_completion_predict_filt(context=context,completion_predict=completion_predict,response="")
        prompt=construct_prompt(context,completion_predict_filt)

        wrapped = {
            "house":house,
            "floor":floor,
            "bound":bound,
            "context":context,
            "completion_predict":completion_predict_filt,
            "prompt":prompt,
        }
        fout.write(json.dumps(wrapped, ensure_ascii=False) + "\n")