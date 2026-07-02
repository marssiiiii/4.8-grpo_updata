import json
import re
import random

def get_force_items_from_text(prompt):
    #print(pre_lineload_text,pre_post_text)
    pattern_lineload = r"<LINELOAD>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\s*(-?\d+)\s*"
    pattern_post = r"<POST>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\s*(-?\d+)\s*"
    pre_lineload_list = [((int(m[0]), int(m[1])), (int(m[2]), int(m[3])), int(m[4]))
                         for m in re.findall(pattern_lineload, prompt)]
    pre_post_list = [((int(m[0]), int(m[1])), int(m[2]))
                     for m in re.findall(pattern_post, prompt)]
    return pre_lineload_list,pre_post_list

if __name__=="__main__":
    INPUT_JSON="train_json_data/five_plus_two_train_jsonl_data/design_3.27/base_model_train/train_set_100_actionized_sort_floor_force_1.jsonl"
    OUTPUT_PATH="train_json_data/five_plus_two_train_jsonl_data/design_3.27/base_model_train/train_set_100_actionized_sort_floor_force_1_filt.jsonl"
    no_pre_force_num,total_num=0,0
    with open(INPUT_JSON, "r", encoding="utf-8") as fin:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                HAS_UNSUPPORTED_ITEM=False
                #print(i)
                record = json.loads(line)
                house,floor,design,prompt = record["house"],record["floor"],record["design"],record["prompt"]
                pre_lineload_list,pre_post_list=get_force_items_from_text(prompt)
            
                rd=random.random()
                if pre_post_list==[] and pre_lineload_list==[]:
                    if rd<0.1:
                        continue
                    no_pre_force_num+=1
                total_num+=1
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"no_pre_force_num:{no_pre_force_num}/{total_num},占比{no_pre_force_num/total_num}")