import json
import sys
sys.path.append("/home/jiuxing_li")
from five_plus_two_optimization.train_model_new.GRPO.test.test_code.unsupport_force_calculate import \
    process_completion_predict_filt,get_force_items_from_text,get_segments_info_five_plus_two,transfer_force_text_to_dict,\
    visualize_force

if __name__=="__main__":
    INPUT_JSON="train_json_data/five_plus_two_train_jsonl_data/design_3.27/base_model_train/train_set_100_actionized_sort_floor_force_auged(3_times).jsonl"
    with open(INPUT_JSON, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            record = json.loads(line)
            house,design,floor,bound=record["house"],record["design"],record["floor"],record["bound"]
            context,prompt,completion_predict,response = \
                record["context"],record["prompt"],record["completion_predict"],record["response"]
            pre_post_text,pre_lineload_text=record["pre_post_text"],record["pre_lineload_text"]
            answer=process_completion_predict_filt(context=context,completion_predict=completion_predict,response=response)
            print(i,house,design,floor)
            pre_lineload_list,pre_post_list=get_force_items_from_text(pre_post_text=pre_post_text,
                                                                pre_lineload_text=pre_lineload_text)
            house_areas=get_segments_info_five_plus_two(context,["inoutbox"],[],[])

            pre_lineload_dict,pre_post_dict=transfer_force_text_to_dict(pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text)
            visualize_force(context=context,answer=answer,pre_context=context,pre_answer=completion_predict,
            pre_post_data=pre_post_dict,pre_lineload_data=pre_lineload_dict,bound=bound,
            output_path=f"five_plus_two_optimization/train_model_new/train_base_model/test/test_pic/aug_check/aug_check_1/{i+1}_{house}_{design}_{floor}.png",
            #unsolved_lineload_list=unsolved_force_item_result["unsolved_lineload_list"],
            #unsolved_post_list=unsolved_force_item_result["unsolved_post_list"],
            house_areas=house_areas)
            if i>50:
                break