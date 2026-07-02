import sys
import json
import re
import random
import torch
sys.path.append("/mnt/efs/jiuxing_li")
sys.path.append("/mnt/efs/jiuxing_li/five_plus_two_optimization/train_model_new/base_code")
from standard_function import get_segments_info_five_plus_two
from reward_design import filter_completion_predict,get_design_score,get_valid_shear_walls
from transformers import AutoTokenizer,AutoModelForCausalLM
from ernie_base_model_train import initialize_token_embedding

MODEL_ID = "baidu/ERNIE-4.5-0.3B-PT"
POLYGON_TYPES = ["opening","inoutbox"]
LINE_TYPES = ["wall","beam","exterior_wall"]
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

if __name__=="__main__":
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
    output_path="train_json_data/five_plus_two_train_jsonl_data/design_3.27/base_model_train/test_set_100_actionized_sort_floor_force.jsonl"
    total_add_beam,total_add_shear,total_remove_beam,total_remove_shear=0,0,0,0
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            #if i<0 or i>6000:
            #    continue
            print(i)
            #将completion_predict转化为action_context
            record = json.loads(line)
            house,floor,design,bound,context,completion_predict = record["house"],record["floor"],record["design"],record["bound"]\
            ,record["context"],record["completion_predict"]

            answer_pre,answer_after=split_cmp_to_answer(completion_predict)
            completion_predict_filtered=filter_completion_predict(context=context,completion_predict=completion_predict)
            
            action_context,ADD_BEAM,ADD_SHEAR,REMOVE_BEAM,REMOVE_SHEAR=from_cmp_to_action_context(context=context,answer_pre=answer_pre,
            answer_after=answer_after,completion_predict=completion_predict_filtered)
            total_add_beam=total_add_beam+ADD_BEAM
            total_add_shear=total_add_shear+ADD_SHEAR
            total_remove_beam=total_remove_beam+REMOVE_BEAM
            total_remove_shear=total_remove_shear+REMOVE_SHEAR
            #print(action_context)

            #依据转化后的结果构造prompt和response
            prompt=""
            prompt=prompt+"Making actions based on context and given structures."
            prompt=prompt+f"context:{context}"
            prompt=prompt+f"structures:{answer_pre}"
            prompt=f"<s>{prompt}</s>"
            response=action_context
            response=random_action_context(response)
            response=f"{response}</s></s>"
            #print(response)
            #print(f"prompt:{prompt},response:{response}")
            prompt_response=prompt+response   #计算prompt+response的总长度
            prompt_response_token_length = len(tokenizer(prompt_response).input_ids)
            if prompt_response_token_length>4999:
                print(f"{prompt_response_token_length}超过长度")
                continue
            
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
                }
                f.write(json.dumps(wrapped, ensure_ascii=False) + "\n")
            
            #break
    print(f"ADD_BEAM操作的数目为{total_add_beam},ADD_SHEAR操作的数目为{total_add_shear},REMOVE_BEAM操作的数目为{total_remove_beam},REMOVE_SHEAR操作的数目为{total_remove_shear}")