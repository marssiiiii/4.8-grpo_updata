import sys
sys.path.append("/home/jiuxing_li")
from five_plus_two_optimization.train_model_new.GRPO.test.test_code.unsupport_force_calculate import\
get_unsolved_force_item
import json
from five_plus_two_optimization.train_model_new.GRPO.test.test_code.model_perform_visualisation_five_plus_two import\
visualize
from five_plus_two_optimization.train_model_new.base_code.reward_design_trunc import\
process_completion_predict_filt

if __name__=="__main__":
    input_path="train_json_data/five_plus_two_train_jsonl_data/design_3.27/post_train_auged_sft_data/grpo_fpt_api_4_4000step_conditioal_improve_flatten_4_steps.jsonl"
    output_pic_dir="five_plus_two_optimization/five_plus_two_test/test_pic/improve_trunc_check/step_4"
    with open(input_path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            record = json.loads(line)
            house,design,floor,context,completion_predict,pre_post_text,pre_lineload_text,bound,response=\
            record["house"],record["design"],record["floor"],record["context"],\
            record["completion_predict"],record["pre_post_text"],record["pre_lineload_text"],\
            record["bound"],record["response"]
            answer=process_completion_predict_filt(context=context,
            completion_predict=completion_predict,response=response)
            unsolved_force_result_pre=get_unsolved_force_item(context=context,answer=completion_predict,pre_lineload_text=pre_lineload_text,pre_post_text=pre_post_text)
            unsolved_force_result=get_unsolved_force_item(context=context,answer=answer,pre_lineload_text=pre_lineload_text,pre_post_text=pre_post_text)
            visualize(context=context,completion_predict=completion_predict,answer=answer,output_pic_dir=f"{output_pic_dir}/{house}_{floor}_{design}.png",
                                  pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text,bound=bound,
                                  unsolved_force_result=unsolved_force_result,unsolved_force_result_pre=unsolved_force_result_pre)
            if i>50:
                break