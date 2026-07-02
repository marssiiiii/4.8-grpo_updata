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
from reward_design_trunc import process_completion_predict_filt
from five_plus_two_optimization.train_model_new.base_code.reward_design_dcr_price_trunc import \
    get_pre_floor_post_and_lineload,get_initial_struct_based_on_response,calculate_price_reward,\
    get_analysis_result_from_design
from transformers import AutoTokenizer,AutoModelForCausalLM
from ernie_base_model_train import initialize_token_embedding
from shapely.geometry import LineString,Point

MODEL_ID = "baidu/ERNIE-4.5-0.3B-PT"
POLYGON_TYPES = ["opening","inoutbox"]
LINE_TYPES = ["wall","beam","exterior_wall","shearwall"]
POINT_TYPES = []
BETWEEN_FLOOR_EPS,OVERLAP_THRESHOLD=30,0.5

def get_force_items_from_text(pre_lineload_text,pre_post_text):
    #print(pre_lineload_text,pre_post_text)
    pattern_lineload = r"<LINELOAD>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\s*(-?\d+)\s*"
    pattern_post = r"<POST>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\s*(-?\d+)\s*"
    pre_lineload_list = [((int(m[0]), int(m[1])), (int(m[2]), int(m[3])), int(m[4]))
                         for m in re.findall(pattern_lineload, pre_lineload_text)]
    pre_post_list = [((int(m[0]), int(m[1])), int(m[2]))
                     for m in re.findall(pattern_post, pre_post_text)]
    return pre_lineload_list,pre_post_list

def get_unsolved_lineload(beams,walls,pre_lineload_list,BETWEEN_FLOOR_EPS,OVERLAP_THRESHOLD):
    unsolved_lineload_list,unsolved_lineload_num=[],0
    for pre_lineload in pre_lineload_list:
        overlap_list=[]
        total_overlap_length=0
        #print(pre_lineload)
        lineload_line=LineString([pre_lineload[0],pre_lineload[1]])
        if lineload_line.length<BETWEEN_FLOOR_EPS:
            continue
        for beam in beams:
            beam_pt_1,beam_pt_2=beam[1],beam[2]
            beam_line=LineString([beam_pt_1, beam_pt_2])
            if get_initial_struct_based_on_response.line_overlap_line(line_1=lineload_line,line_2=beam_line)==True:
                overlap_length=get_initial_struct_based_on_response.fuzzy_line_overlap_length(
                                                                line_1=lineload_line,line_2=beam_line)
                total_overlap_length+=overlap_length
                overlap_list.append(beam_line)
        for wall in walls:
            wall_pt_1,wall_pt_2=wall[1],wall[2]
            wall_line=LineString([wall_pt_1, wall_pt_2])
            if get_initial_struct_based_on_response.line_overlap_line(line_1=lineload_line,line_2=wall_line)==True:
                overlap_length=get_initial_struct_based_on_response.fuzzy_line_overlap_length(
                                                    line_1=lineload_line,line_2=wall_line)
                total_overlap_length+=overlap_length
                overlap_list.append(wall_line)
        if total_overlap_length/lineload_line.length<OVERLAP_THRESHOLD:
            #print(f"lineload_line:{lineload_line},overlap_list:{overlap_list}")
            unsolved_lineload_list.append((pre_lineload[0],pre_lineload[1],pre_lineload[2]))
            unsolved_lineload_num=unsolved_lineload_num+1
    return unsolved_lineload_list,unsolved_lineload_num

def get_unsolved_post(beams,walls,pre_post_list):
    unsolved_post_list,unsolved_post_num=[],0
    for pre_post in pre_post_list:
        POST_SOLVED=False
        for beam in beams:
            beam_pt_1,beam_pt_2=beam[1],beam[2]
            if get_initial_struct_based_on_response.point_in_line(pt=Point(pre_post[0]),line=LineString([beam_pt_1,beam_pt_2])):
                POST_SOLVED=True
        for wall in walls:
            wall_pt_1,wall_pt_2=wall[1],wall[2]
            if get_initial_struct_based_on_response.point_in_line(pt=Point(pre_post[0]),line=LineString([wall_pt_1,wall_pt_2])):
                POST_SOLVED=True
        if POST_SOLVED==False:
            unsolved_post_list.append((pre_post[0][0],pre_post[0][1],pre_post[1]))
            unsolved_post_num+=1
    return unsolved_post_list,unsolved_post_num

def get_unsolved_force_item(context,answer,pre_lineload_text,pre_post_text,BETWEEN_FLOOR_EPS,OVERLAP_THRESHOLD):
    pre_lineload_list,pre_post_list=get_force_items_from_text(pre_lineload_text,pre_post_text)
    walls=get_segments_info_five_plus_two(context,[],["wall","exterior_wall"],[])
    beams=get_segments_info_five_plus_two(answer,[],["beam"],[])

    unsolved_lineload_list,unsolved_lineload_num=get_unsolved_lineload(beams=beams,
    walls=walls,pre_lineload_list=pre_lineload_list,BETWEEN_FLOOR_EPS=BETWEEN_FLOOR_EPS,OVERLAP_THRESHOLD=OVERLAP_THRESHOLD)
    unsolved_post_list,unsolved_post_num=get_unsolved_post(beams=beams,walls=walls,pre_post_list=pre_post_list)

    return {"unsolved_lineload_list":unsolved_lineload_list,"unsolved_lineload_num":unsolved_lineload_num,
                "unsolved_post_list":unsolved_post_list,"unsolved_post_num":unsolved_post_num}

if __name__=="__main__":
    INPUT_JSON="five_plus_two_optimization/train_model_new/train_base_model/test/test_jsonl/base_model_9_train.jsonl"
    unsolve_force_num,total_num=0,0
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            HAS_UNSUPPORTED_ITEM=False
            print(i)
            record = json.loads(line)
            house,floor,design,bound,context,completion_predict,response_pred,pre_post_text,pre_lineload_text = \
            record["house"],record["floor"],record["design"],record["bound"]\
            ,record["context"],record["completion_predict"],record["response_pred"],\
            record["pre_post_text"],record["pre_lineload_text"]

            if pre_post_text==None and pre_lineload_text==None:
                continue

            house_design_code=str(house)+str(design)
            house_areas=get_segments_info_five_plus_two(context,["inoutbox"],[],[])
            print(f"{i+1},{house_design_code},{floor}")
            answer=process_completion_predict_filt(context=context,
                    completion_predict=completion_predict,response=response_pred)
        
            unsolved_force_item_result=get_unsolved_force_item(context=context,answer=answer,
            pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text,
            BETWEEN_FLOOR_EPS=BETWEEN_FLOOR_EPS,OVERLAP_THRESHOLD=OVERLAP_THRESHOLD)

            print(f"unsolved_force_item_result:{unsolved_force_item_result}")
            if unsolved_force_item_result["unsolved_lineload_num"]+unsolved_force_item_result["unsolved_post_num"]>0:
                HAS_UNSUPPORTED_ITEM=True
                print("有未支撑的上层受力")
                unsolve_force_num+=1
            total_num+=1
    
    print(f"未解决force:{unsolve_force_num}/{total_num}")