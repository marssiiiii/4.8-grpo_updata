import sys
import json
import re
import random
import torch
import time
import matplotlib.pyplot as plt
sys.path.append("/home/jiuxing_li")
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/train_model_new/base_code")
from standard_function import get_segments_info_five_plus_two
from reward_design_trunc import filter_completion_predict,get_design_score,get_valid_shear_walls
from five_plus_two_optimization.train_model_new.base_code.reward_design_dcr_price_trunc import \
    get_pre_floor_post_and_lineload,get_initial_struct_based_on_response,calculate_price_reward,\
        get_analysis_result_from_design,visualize_force
from transformers import AutoTokenizer,AutoModelForCausalLM
from ernie_base_model_train import initialize_token_embedding
from shapely.geometry import LineString,Point
from five_plus_two_optimization.train_model_new.GRPO.test.test_code.unsupport_force_calculate import get_unsolved_force_item

MODEL_ID = "baidu/ERNIE-4.5-0.3B-PT"
POLYGON_TYPES = ["opening","inoutbox"]
LINE_TYPES = ["wall","beam","exterior_wall","shearwall"]
POINT_TYPES = []

def filter_valid_structures(context,beams,shear_walls):
    #提取beams和shear_walls转为completion_predict
    completion_predict=""
    for beam in beams:
        completion_predict=completion_predict+f"<beam>({int(beam[1][0])},{int(beam[1][1])}),({int(beam[1][0])},{int(beam[1][1])})"
    for shearwall in shear_walls:
        completion_predict=completion_predict+f"<shearwall>({int(shearwall[1][0])},{int(shearwall[1][1])}),({int(shearwall[1][0])},{int(shearwall[1][1])})"
    #
    completion_predict_filtered=filter_completion_predict(context=context,completion_predict=completion_predict)
    #
    beams=get_segments_info_five_plus_two(completion_predict_filtered,[],['beam'],[])
    shear_walls=get_segments_info_five_plus_two(completion_predict_filtered,[],['shearwall'],[])
    return beams,shear_walls

def from_cmp_to_action_context(context,answer_pre,answer_after,completion_predict):
    #
    house_items=get_segments_info_five_plus_two(context,POLYGON_TYPES,LINE_TYPES,POINT_TYPES)
    exterior_walls,walls=[],[]
    for seg in house_items:
        if seg[0]=="exterior_wall":
            exterior_walls.append(seg)
        elif seg[0]=="wall":
            walls.append(seg)
    #
    result=get_design_score(context=context,answer=completion_predict)
    final_valid_beams,final_valid_shear_walls=result['valid_beams'],result['shear_walls_valid'] #从answer中得到最终有效的设计
    #final_valid_beams,final_valid_shear_walls=filter_valid_structures(context=context,beams=final_valid_beams,shear_walls=final_valid_shear_walls)

    #print(f"final_valid_beams:{final_valid_beams},final_valid_shear_walls:{final_valid_shear_walls}")

    result_action_context=""
    #对answer_pre中不合理的结构进行remove
    beams_pre=get_segments_info_five_plus_two(answer_pre,[],['beam'],[])
    shear_walls_pre=get_segments_info_five_plus_two(answer_pre,[],['shearwall'],[])
    #print(f"beams_pre:{beams_pre},shear_walls_pre:{shear_walls_pre}")
    ADD_BEAM,ADD_SHEAR,REMOVE_BEAM,REMOVE_SHEAR=0,0,0,0
    beams_added=[]
    shear_walls_added=[]
    for beam in beams_pre: #移除answer_pre中不合理的beam
        if (beam not in final_valid_beams):
            result_action_context=result_action_context+f"<remove><{beam[0]}>({int(beam[1][0])},{int(beam[1][1])}),({int(beam[2][0])},{int(beam[2][1])})"
            REMOVE_BEAM=REMOVE_BEAM+1
        elif (beam not in beams_added):
            #result_action_context=result_action_context+f"<remove><{beam[0]}>({int(beam[1][0])},{int(beam[1][1])}),({int(beam[2][0])},{int(beam[2][1])})"
            #REMOVE_BEAM=REMOVE_BEAM+1
            beams_added.append(beam)
    for shearwall in shear_walls_pre: #移除answer_pre中不合理的shearwall
        if (shearwall not in final_valid_shear_walls):
            result_action_context=result_action_context+f"<remove><{shearwall[0]}>({int(shearwall[1][0])},{int(shearwall[1][1])}),({int(shearwall[2][0])},{int(shearwall[2][1])})"
            REMOVE_SHEAR=REMOVE_SHEAR+1
        elif (shearwall not in shear_walls_added):
            #result_action_context=result_action_context+f"<remove><{shearwall[0]}>({int(shearwall[1][0])},{int(shearwall[1][1])}),({int(shearwall[2][0])},{int(shearwall[2][1])})"
            #REMOVE_SHEAR=REMOVE_SHEAR+1
            shear_walls_added.append(shearwall)
    
    #将shear_walls_added扩充
    #shear_walls_added=get_valid_shear_walls(walls,exterior_walls,shear_walls_added,beams_added)
    shear_walls_added=get_valid_shear_walls(walls,exterior_walls,shear_walls_added,[])
    #print(f"shear_walls_added:{shear_walls_added}")
    
    #将剩余answer_after中合理的结构加入到result_action_context中
    beams_after=get_segments_info_five_plus_two(answer_after,[],['beam'],[])
    shear_walls_after=get_segments_info_five_plus_two(answer_after,[],['shearwall'],[])
    #print(f"beams_after:{beams_after},shear_walls_after:{shear_walls_after}")
    for beam in beams_after:
        if (beam in final_valid_beams) and (beam not in beams_added):
            result_action_context=result_action_context+f"<add><{beam[0]}>({int(beam[1][0])},{int(beam[1][1])}),({int(beam[2][0])},{int(beam[2][1])})"
            ADD_BEAM=ADD_BEAM+1
            beams_added.append(beam)
    for shearwall in shear_walls_after:
        #if (shearwall in final_valid_shear_walls) and (shearwall not in shear_walls_added):
        if (shearwall in final_valid_shear_walls) and random.random()>0.5:
            result_action_context=result_action_context+f"<add><{shearwall[0]}>({int(shearwall[1][0])},{int(shearwall[1][1])}),({int(shearwall[2][0])},{int(shearwall[2][1])})"
            ADD_SHEAR=ADD_SHEAR+1
            shear_walls_added.append(shearwall)

    return result_action_context,ADD_BEAM,ADD_SHEAR,REMOVE_BEAM,REMOVE_SHEAR

def split_cmp_to_answer(completion_predict,split_point=0.7): #将completion_predict分割为前后两部分（顺序均为先beam后shearwall）
    beams=get_segments_info_five_plus_two(completion_predict,[],['beam'],[])
    shearwalls=get_segments_info_five_plus_two(completion_predict,[],['shearwall'],[])
    answer_pre,answer_after="",""
    #构造answer_pre
    for beam in beams[:int(len(beams)*split_point)]:
        answer_pre=answer_pre+f"<beam>({int(beam[1][0])},{int(beam[1][1])}),({int(beam[2][0])},{int(beam[2][1])})"
    for shearwall in shearwalls[:int(len(shearwalls)*split_point)]:
        answer_pre=answer_pre+f"<shearwall>({int(shearwall[1][0])},{int(shearwall[1][1])}),({int(shearwall[2][0])},{int(shearwall[2][1])})"
    #构造answer_after
    for beam in beams[int(len(beams)*split_point):]:
        answer_after=answer_after+f"<beam>({int(beam[1][0])},{int(beam[1][1])}),({int(beam[2][0])},{int(beam[2][1])})"
    for shearwall in shearwalls[int(len(shearwalls)*split_point):]:
        answer_after=answer_after+f"<shearwall>({int(shearwall[1][0])},{int(shearwall[1][1])}),({int(shearwall[2][0])},{int(shearwall[2][1])})"

    return answer_pre,answer_after

def random_action_context(action_context):
    #获取action_context内的内容并打乱
    pattern = r"<(add|remove)><(beam|shearwall)>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
    results = []
    for m in re.finditer(pattern, action_context):
        act_type,seg_type,x1, y1, x2, y2 = m.groups() #m代表一个捕获pattern
        results.append([act_type,seg_type,x1,y1,x2,y2])
    random.shuffle(results)
    #将打乱后的内容组合成random_action_context
    random_action_context=""
    for result in results:
        random_action_context=random_action_context+f"<{result[0]}><{result[1]}>({result[2]},{result[3]}),({result[4]},{result[5]})"
    return random_action_context

def process_force_info_to_text(pre_force_load_result):
    def point_in_line(pt:Point,line:LineString,BETWEEN_FLOOR_EPS=30):
        return pt.distance(line) < BETWEEN_FLOOR_EPS

    lineload_line_list=[]
    text_lineload=""
    pre_lineload_dict=pre_force_load_result["pre_lineload_dict"]
    for lineload_key in pre_lineload_dict.keys():
        lineload_item=pre_lineload_dict[lineload_key]
        if lineload_item.force>=1:
            text_lineload+=f"<LINELOAD>({lineload_key[0][0]},{lineload_key[0][1]}),({lineload_key[1][0]},{lineload_key[1][1]}),{int(lineload_item.force)}"
            lineload_line_list.append(LineString([lineload_key[0],lineload_key[1]]))
    
    text_post=""
    pre_post_dict=pre_force_load_result["pre_post_dict"]
    for post_key in pre_post_dict.keys():
        post_item=pre_post_dict[post_key]
        if post_item.force>=1:
            POST_IN_LINELOAD=False
            post_point=Point(post_key)
            for lineload_line in lineload_line_list:
                if point_in_line(pt=post_point,line=lineload_line)==True:
                    POST_IN_LINELOAD=True
                    #print("POST_IN_LINELOAD")
                    break
            if POST_IN_LINELOAD==False:
                text_post+=f"<POST>({post_key[0]},{post_key[1]}),{int(post_item.force)}"
    
    return text_post,text_lineload

def construct_prompt(pre_post_text,pre_lineload_text,context,completion_predict):
    prompt=""
    prompt=prompt+"Making actions based on context and given structures,and support the upper layer for force transmission."
    prompt=prompt+f"upper layer lineload:{pre_lineload_text} "
    prompt=prompt+f"upper layer post:{pre_post_text} "
    prompt=prompt+f"context:{context} "
    prompt=prompt+f"structures:{completion_predict}"
    prompt=f"<s>{prompt}</s>"
    return prompt

if __name__=="__main__":
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

    INPUT_JSON="train_json_data/five_plus_two_train_jsonl_data/design_3.27/initial_data_100_train_set_inbox_shf_sort_floor.jsonl"
    output_path="five_plus_two_optimization/train_model_new/train_base_model/test/test_set_100_actionized_sort_floor_force_1_validation.jsonl"
    total_add_beam,total_add_shear,total_remove_beam,total_remove_shear=0,0,0,0

    house_design_code_pre,floor_design_pre,floor_pre=None,None,None
    designs={}
    pre_context,pre_answer="",""
    sta_time=time.time()
    STA,END=2000,2300
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i<STA or i>END:
                continue
            print(f"sta:{STA},end:{END}")
            print(i)
            record = json.loads(line)
            house,floor,design,bound,context,completion_predict = record["house"],record["floor"],record["design"],record["bound"]\
            ,record["context"],record["completion_predict"]
            house_design_code=str(house)+str(design)
            house_areas=get_segments_info_five_plus_two(context,["inoutbox"],[],[])
            print(f"{i+1},{house_design_code},{floor}")
            answer=completion_predict
            HAS_UNSUPPORTED_ITEM=False
            #测试代码
            #answer=filter_completion_predict(context=context,completion_predict=completion_predict)

            #依据context和answer_1构造pre_post和pre_lineload
            if house_design_code_pre==None or (house_design_code_pre!=None and house_design_code!=house_design_code_pre): #代表进入一个新的房屋设计
                print("开始新的房屋设计")
                designs={}
                floor_pre=None
                floor_design_pre=None
            
            cls_initial_struct=get_initial_struct_based_on_response(floor=floor,context=context, #整合设计，得到返回结果
            answer=answer,floor_design_pre=floor_design_pre)
            floor_design,UNSOLVED_POST_NUM=cls_initial_struct.get_house_struct()
            designs[floor]=floor_design
            resp_result=get_analysis_result_from_design(designs=designs,house=house,
                                house_design_code=house_design_code,floor=floor,f_index=i+1,NEED_OUTPUT=False)
            if floor_pre!=None and resp_result!=None: #获取上层的post和lineload
                cls_pre_force_load=get_pre_floor_post_and_lineload(resp_struct=resp_result["resp_floors"][floor_pre]["structs"],bound=bound,house_areas=house_areas)
                pre_force_load_result=cls_pre_force_load.get_pre_post_and_pre_lineload()
                pre_post_text,pre_lineload_text=process_force_info_to_text(pre_force_load_result=pre_force_load_result)
            else:
                pre_force_load_result=None
                pre_post_text,pre_lineload_text=None,None
            
            cls_force_load=get_pre_floor_post_and_lineload(resp_struct=resp_result["resp_floors"][floor]["structs"],bound=bound,house_areas=house_areas)
            force_load_result=cls_force_load.get_pre_post_and_pre_lineload()
            
            cls_price_reward=calculate_price_reward(resp_struct=resp_result["resp_floors"][floor]["structs"],extra_data=resp_result["extra_data"],
                                analysis_score=resp_result["analysis_score"],floor=floor,floor_design=floor_design,
                                pre_lineload_dict=pre_force_load_result["pre_lineload_dict"] if pre_force_load_result!=None else {},
                                pre_post_dict=pre_force_load_result["pre_post_dict"] if pre_force_load_result!=None else {})
            unsolved_force_item_result=cls_price_reward.get_unsolved_force_item()
            warn_error_result=cls_price_reward.get_warn_error_from_resp()
            print(f"warn_error_result:{warn_error_result}")
            print(f"unsolved_force_item_result:{unsolved_force_item_result}")
            if unsolved_force_item_result["unsolved_lineload_num"]+unsolved_force_item_result["unsolved_post_num"]>0:
                HAS_UNSUPPORTED_ITEM=True
                print("有未支撑的上层受力")
            #print(f"pre_post_text:{pre_post_text},pre_lineload_text:{pre_lineload_text}")
            
            if NEED_VISUALIZE:
                if pre_force_load_result==None: #可视化效果
                    visualize_force(context=context,answer=answer,pre_context=pre_context,pre_answer=pre_answer,
                                    pre_post_data=None,pre_lineload_data=None,bound=bound,
                                    post_data=force_load_result["pre_post_dict"],lineload_data=force_load_result["pre_lineload_dict"],
                                    output_path=f"five_plus_two_optimization/five_plus_two_test/test_pic/data_3.27/api_force_check_1/{i+1}_{house_design_code}_{floor}.png",
                                    house_areas=house_areas)
                else:
                    visualize_force(context=context,answer=answer,pre_context=pre_context,pre_answer=pre_answer,
                                    pre_post_data=pre_force_load_result["pre_post_dict"]
                                    ,pre_lineload_data=pre_force_load_result["pre_lineload_dict"],bound=bound,
                                    post_data=force_load_result["pre_post_dict"],lineload_data=force_load_result["pre_lineload_dict"],
                    output_path=f"five_plus_two_optimization/five_plus_two_test/test_pic/data_3.27/api_force_check_1/{i+1}_{house_design_code}_{floor}.png",
                    unsolved_lineload_list=unsolved_force_item_result["unsolved_lineload_list"],
                    unsolved_post_list=unsolved_force_item_result["unsolved_post_list"],
                    house_areas=house_areas)
            
            #切割completion_predict为answer_1,answer_2,依据转化后的结果构造prompt和response
            answer_1,answer_2=split_cmp_to_answer(completion_predict)
            completion_predict_filtered=filter_completion_predict(context=context,completion_predict=completion_predict)
            response,ADD_BEAM,ADD_SHEAR,REMOVE_BEAM,REMOVE_SHEAR=from_cmp_to_action_context(context=context,answer_pre=answer_1,
            answer_after=answer_2,completion_predict=completion_predict_filtered)
            response=random_action_context(response)
            #print(f"completion_predict:{answer_1},response:{response}")

            prompt=construct_prompt(pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text,
                                    context=context,completion_predict=answer_1)
            
            response=f"{response}</s></s>"
            #print(response)
            #print(f"prompt:{prompt},response:{response}")
            prompt_response=prompt+response   #计算prompt+response的总长度
            prompt_response_token_length = len(tokenizer(prompt_response).input_ids)
            if prompt_response_token_length>9999:
                print(f"{prompt_response_token_length}超过长度")
            
            if NEED_OUTPUT==True and prompt_response_token_length<9999 and HAS_UNSUPPORTED_ITEM==False:
                with open(output_path, "a", encoding="utf-8") as f: #追加文件而不是覆盖
                    wrapped = {
                        "house":house,
                        "floor":floor,
                        "design":design,
                        "bound":bound,
                        "context":context,
                        "completion_predict":completion_predict,
                        "prompt":prompt,
                        "response":response,
                        "pre_post_text":pre_post_text,
                        "pre_lineload_text":pre_lineload_text
                    }
                    f.write(json.dumps(wrapped, ensure_ascii=False) + "\n")
            
            #将模型对过去的记忆改为完整的设计
            house_design_code_pre,floor_pre,floor_design_pre=house_design_code,floor,floor_design
            pre_context,pre_answer=context,answer
            #if i>2:
            #    break
            #break
    end_time=time.time()
    total_time,avg_time=end_time-sta_time,(end_time-sta_time)/10
    print(f"总用时：{total_time}，平均用时：{avg_time},10000条计算效率：{10000*avg_time/5}")