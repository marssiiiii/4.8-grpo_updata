import json
import sys
sys.path.append("/home/jiuxing_li")
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/train_model_new/base_code")
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/train_model_new/train_base_model")
from standard_function import get_segments_info_five_plus_two
from five_plus_two_optimization.train_model_new.base_code.reward_design_trunc import process_completion_predict_filt
from cmp_actionized_and_tokenized_api import process_force_info_to_text,initialize_token_embedding
from five_plus_two_optimization.train_model_new.base_code.reward_design_trunc.reward_design_dcr_price_trunc import \
    get_pre_floor_post_and_lineload,get_initial_struct_based_on_response,get_analysis_result_from_design,\
    calculate_price_reward,visualize_force
from transformers import AutoTokenizer,AutoModelForCausalLM
MODEL_ID = "baidu/ERNIE-4.5-0.3B-PT"
import torch

def construct_prompt(context,completion_predict,pre_post_text,pre_lineload_text):
    prompt=""
    prompt=prompt+"Making actions based on context and given structures,and support the upper layer for force transmission."
    prompt=prompt+f"context:{context} "
    prompt=prompt+f"upper layer post:{pre_post_text} "
    prompt=prompt+f"upper layer lineload:{pre_lineload_text} "
    prompt=prompt+f"structures:{completion_predict} "
    prompt=f"<s>{prompt}</s>"
    return prompt

def design_has_error(resp_result,floor,pre_force_load_result,floor_design):
    cls_price_reward=calculate_price_reward(resp_struct=resp_result["resp_floors"][floor]["structs"],
            extra_data=resp_result["extra_data"],analysis_score=resp_result["analysis_score"],
            floor=floor,floor_design=floor_design,
            pre_lineload_dict=pre_force_load_result["pre_lineload_dict"] if pre_force_load_result!=None else {},
            pre_post_dict=pre_force_load_result["pre_post_dict"] if pre_force_load_result!=None else {})
    unsolved_force_item_result=cls_price_reward.get_unsolved_force_item()
    warn_error_result=cls_price_reward.get_warn_error_from_resp()
    error_num=unsolved_force_item_result["unsolved_lineload_num"]+unsolved_force_item_result["unsolved_post_num"]+warn_error_result["error_cnt"]
    return error_num>0

if __name__ == "__main__":
    NEED_OUTPUT=False
    NEED_VISUALIZE=False
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID) #初始化tokenizer
    if tokenizer.pad_token is None: #如果tokenizer中没有pad_token,就用eos_token填充
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto", #自动把模型分配到gpu
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, #使用半精度节省显存，如gpu不支持则用32位浮点
        use_cache=False,  # 禁用KV缓存，与梯度检查点冲突
    )
    initialize_token_embedding(model,tokenizer)

    input_path="train_json_data/five_plus_two_train_jsonl_data/design_3.27/initial_data_100_test_set_inbox_shf_sort_floor.jsonl"
    output_path="train_json_data/five_plus_two_train_jsonl_data/design_3.27/grpo_phase_1_train_data/test_set_inbox_force_prompt_1.jsonl"

    with open(input_path,"r",encoding="utf-8") as fin,\
    open(output_path, "a", encoding="utf-8") as fout:
        INVALID_HOUSE_DESIGN_CODE=""
        house_design_code_pre,floor_design_pre,floor_pre=None,None,None
        designs,floor_pre_list={},[]
        pre_context,pre_answer="",""
        for i,line in enumerate(fin):
            if i<500:
                continue
            #if i<8000 or i>18000:
            #    continue
            #读取输入信息
            record = json.loads(line)
            house,floor,design,bound,context,completion_predict = record["house"],record["floor"],record["design"],\
            record["bound"],record["context"],record["completion_predict"]
            house_areas=get_segments_info_five_plus_two(context,["inoutbox"],[],[])
            house_design_code=str(house)+str(design)
            print(f"{i},{house_design_code},{floor}")
            answer=process_completion_predict_filt(context=context,completion_predict=completion_predict,response="")
            
            #处理force信息
            if house_design_code==INVALID_HOUSE_DESIGN_CODE: #如果有某一层设计无效，那么直接跳过后续楼层的设计
                print("之前层有错误，跳过本楼层")
                house_design_code_pre,floor_design_pre,floor_pre=None,None,None
                designs,floor_pre_list={},[]
                pre_context,pre_answer="",""
                continue

            if house_design_code_pre==None or (house_design_code_pre!=None and house_design_code!=house_design_code_pre): #代表进入一个新的房屋设计
                print("开始新的房屋设计")
                house_design_code_pre,floor_design_pre,floor_pre=None,None,None
                designs,floor_pre_list={},[]
                pre_context,pre_answer="",""
            
            cls_initial_struct=get_initial_struct_based_on_response(floor=floor,context=context, #整合设计，得到返回结果
            answer=answer,floor_design_pre=floor_design_pre)
            floor_design,UNSOLVED_POST_NUM=cls_initial_struct.get_house_struct()
            designs[floor]=floor_design
            resp_result=get_analysis_result_from_design(designs=designs,house=house,
                                    house_design_code=house_design_code,floor=floor,f_index=i+1,NEED_OUTPUT=False)
            
            if floor_pre!=None and resp_result!=None:#得到pre_force
                cls_pre_force_load=get_pre_floor_post_and_lineload(
                    resp_struct=resp_result["resp_floors"][floor_pre]["structs"],bound=bound,house_areas=house_areas)
                pre_force_load_result=cls_pre_force_load.get_pre_post_and_pre_lineload()
                pre_post_text,pre_lineload_text=process_force_info_to_text(pre_force_load_result=pre_force_load_result)
            else:
                pre_force_load_result=None
                pre_post_text,pre_lineload_text=None,None
            cls_force_load=get_pre_floor_post_and_lineload(resp_struct=resp_result["resp_floors"][floor]["structs"],bound=bound,house_areas=house_areas)
            force_load_result=cls_force_load.get_pre_post_and_pre_lineload()
            
            if NEED_VISUALIZE:
                if pre_force_load_result==None: #可视化效果
                    visualize_force(context=context,answer=answer,pre_context=pre_context,pre_answer=pre_answer,
                                    pre_post_data=None,pre_lineload_data=None,bound=bound,
                                    post_data=force_load_result["pre_post_dict"],lineload_data=force_load_result["pre_lineload_dict"],
                                    output_path=f"five_plus_two_optimization/five_plus_two_test/test_pic/data_3.27/api_force_check_2/{i+1}_{house_design_code}_{floor}.png",
                                    house_areas=house_areas)
                else:
                    cls_price_reward=calculate_price_reward(resp_struct=resp_result["resp_floors"][floor]["structs"],extra_data=resp_result["extra_data"],
                                analysis_score=resp_result["analysis_score"],floor=floor,floor_design=floor_design,
                                pre_lineload_dict=pre_force_load_result["pre_lineload_dict"] if pre_force_load_result!=None else {},
                                pre_post_dict=pre_force_load_result["pre_post_dict"] if pre_force_load_result!=None else {})
                    unsolved_force_item_result=cls_price_reward.get_unsolved_force_item()
                    visualize_force(context=context,answer=answer,pre_context=pre_context,pre_answer=pre_answer,
                                    pre_post_data=pre_force_load_result["pre_post_dict"]
                                    ,pre_lineload_data=pre_force_load_result["pre_lineload_dict"],bound=bound,
                                    post_data=force_load_result["pre_post_dict"],lineload_data=force_load_result["pre_lineload_dict"],
                    output_path=f"five_plus_two_optimization/five_plus_two_test/test_pic/data_3.27/api_force_check_2/{i+1}_{house_design_code}_{floor}.png",
                    unsolved_lineload_list=unsolved_force_item_result["unsolved_lineload_list"],
                    unsolved_post_list=unsolved_force_item_result["unsolved_post_list"],
                    house_areas=house_areas)
            
            #整合信息并输出
            prompt=construct_prompt(context=context,completion_predict=answer,
                                    pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text)
            prompt_token_length = len(tokenizer(prompt).input_ids)
            if prompt_token_length>9999:
                print(f"{prompt_token_length}超过长度")

            if prompt_token_length<=9999 and NEED_OUTPUT==True:
                wrapped = {
                    "house":house,
                    "floor":floor,
                    "design":design,
                    "bound":bound,
                    "context":context,
                    "completion_predict":answer,
                    "prompt":prompt,
                    "pre_post_text":pre_post_text,
                    "pre_lineload_text":pre_lineload_text
                }
                fout.write(json.dumps(wrapped, ensure_ascii=False) + "\n")
            #加入token长度限制
            if design_has_error(resp_result=resp_result,floor=floor,
                            pre_force_load_result=pre_force_load_result,floor_design=floor_design)==True:
                print(f"{house_design_code}_{floor}的设计存在error，输出本楼层，终止其后续的楼层的生成!")
                INVALID_HOUSE_DESIGN_CODE=house_design_code
                house_design_code_pre,floor_design_pre,floor_pre=None,None,None
                designs,floor_pre_list={},[]
                pre_context,pre_answer="",""
                continue
            else:
                house_design_code_pre,floor_pre,floor_design_pre=house_design_code,floor,floor_design
                floor_pre_list.append(floor)
                pre_context,pre_answer=context,answer

#数量训练集：16877，测试集：5701