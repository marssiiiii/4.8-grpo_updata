import json
import sys
sys.path.append("/home/jiuxing_li")
from five_plus_two_optimization.train_model_new.train_base_model.data_process.pre_force_aug import \
get_force_items_from_unprocess_force_text,process_force_info_to_text
from five_plus_two_optimization.train_model_new.train_base_model.cmp_actionized_and_tokenized_api import \
construct_prompt

if __name__=="__main__":
    input_path="train_json_data/five_plus_two_train_jsonl_data/design_3.27/grpo_phase_1_train_data/validation/test_set_inbox_force_prompt_validation_25.jsonl"
    output_path="train_json_data/five_plus_two_train_jsonl_data/design_3.27/grpo_phase_1_train_data/validation/test_set_inbox_force_prompt_validation_25_transfer.jsonl"
    with open(input_path, "r", encoding="utf-8") as f:
        with open(output_path,"w",encoding="utf-8") as fout:
            for i, line in enumerate(f):
                record = json.loads(line)
                #prompt=record["prompt"]
                #print(f"prompt:{prompt}")
                
                pre_post_text,pre_lineload_text=record["pre_post_text"],record["pre_lineload_text"]
                if pre_post_text==None:
                    pre_post_text=""
                if pre_lineload_text==None:
                    pre_lineload_text=""
                #print(f"pre_post_text:{pre_post_text},pre_lineload_text:{pre_lineload_text}")
                pre_lineload_list,pre_post_list=get_force_items_from_unprocess_force_text(pre_post_text+pre_lineload_text)
                
                #pre_lineload_list,pre_post_list=get_force_items_from_unprocess_force_text(prompt)
                pre_lineload_text_transfer,pre_post_text_transfer=process_force_info_to_text(
                    pre_lineload_list=pre_lineload_list,pre_post_list=pre_post_list)
                #print(f"pre_post_text_transfer:{pre_post_text_transfer},pre_lineload_text_transfer:{pre_lineload_text_transfer}")
                prompt_transfer=construct_prompt(pre_post_text=pre_post_text_transfer,pre_lineload_text=pre_lineload_text_transfer,
                        context=record["context"],completion_predict=record["completion_predict"])
                #print(f"prompt_transfer:{prompt_transfer}")
                #if i>50:
                #    break
                record["pre_post_text"]=pre_post_text_transfer
                record["pre_lineload_text"]=pre_lineload_text_transfer
                record["prompt"]=prompt_transfer
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")