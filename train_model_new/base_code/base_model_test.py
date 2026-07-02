import json
import torch
import sys
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from shapely.geometry import LineString,Point
sys.path.append("/mnt/disks/workspace/jiuxing_li")
sys.path.append("/mnt/efs/jiuxing_li/five_plus_two_optimization/train_model_new/base_code")
sys.path.append("code/GRPO/test/test_code")
sys.path.append("five_plus_two_optimization/five_plus_two_data_embedding_3_11")
from reward_design import judge_shearwall_action_is_valid #judge_shearwall_action_is_valid(sh_action,walls,sh_pre_list)
from standard_function import get_segments_info_five_plus_two
from grpo_no_depend_test import point_is_on_line
from Line_Judgement_3_11 import orientation
import time
#思路更新12.3
#在推理过程中加入物理规则
POLYGON_TYPES = ["opening","joist_area","truss_area","diaphragms"]
LINE_TYPES = ["wall","beam"]
POINT_TYPES = []
# ------------------------
# 配置
# ------------------------
#mode_path = "Qwen/Qwen2.5-3B-Instruct"
#mode_path="Qwen/Qwen2.5-72B-Instruct"
#mode_path = "Qwen/Qwen2.5-0.5B-Instruct"
mode_path = "five_plus_two_optimization/train_model_new/GRPO/base_model/ernie_base_model_2"

INPUT_JSON = "train_json_data/five_plus_two_train_jsonl_data/design_3.10/initial_data_train.jsonl"#var
OUTPUT_PATH="train_json_data/five_plus_two_train_jsonl_data/design_3.10/initial_data_train_responsed.jsonl" #var
MAX_LENGTH=5000
import os
os.environ["HF_HOME"] = "/mnt/efs/jiuxing_li/five_plus_two_optimization/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/efs/jiuxing_li/five_plus_two_optimization/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/mnt/efs/jiuxing_li/five_plus_two_optimization/huggingface"
cache_dir = "/mnt/efs/jiuxing_li/five_plus_two_optimization/huggingface/models"
# ------------------------
# 加载模型与tokenizer
# ------------------------

def filter_completion_predict(completion_predict,context): #从给出的初始设计方案中预去除无效的操作
    #读取walls
    walls=[]
    house_items=get_segments_info_five_plus_two(context,POLYGON_TYPES,LINE_TYPES,POINT_TYPES)
    for seg in house_items:
        if seg[0]=="wall":
            walls.append(seg)
    #print(f"walls:{walls}")
    #print(f"completion_predict:{completion_predict}")

    #过滤completion_predict中重复的beam,并将提取到的beam记录在beams中
    pattern = r"<beams?>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
    segs=re.findall(pattern,completion_predict)
    vis_beam={}
    beams=[]
    for seg in segs:
        line=LineString([(int(seg[0]),int(seg[1])),(int(seg[2]),int(seg[3]))])
        if line in vis_beam:
            beam_text=f"<beam>({int(seg[0])},{int(seg[1])}),({int(seg[2])},{int(seg[3])})"
            completion_predict=completion_predict.replace(beam_text,"",1) #注意这里只去除一个beam,不能全部去除
            #print(f"{beam_text}已经存在，移除")
        else:
            beams.append(["beam",(int(seg[0]),int(seg[1])),(int(seg[2]),int(seg[3]))])
            vis_beam[line]=1

    #过滤completion_predict中不合法的shearwall
    pattern = r"<shearwall>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
    shs=re.findall(pattern,completion_predict)
    sh_pre_list=[]

    #利用beams初始化一个sh_pre_list名单
    for wall in walls: #被beam平行支撑的wall,都算作shear_wall
        wall_pt_1,wall_pt_2=(wall[1][0],wall[1][1]),(wall[2][0],wall[2][1])
        for beam in beams:
            beam_pt_1,beam_pt_2=(beam[1][0],beam[1][1]),(beam[2][0],beam[2][1])
            if orientation(wall_pt_1,wall_pt_2,beam_pt_1)==0 and orientation(wall_pt_1,wall_pt_2,beam_pt_2)==0:#beam和wall必须共线
                if ((beam_pt_1==wall_pt_1 or beam_pt_1==wall_pt_2) and point_is_on_line(beam_pt_2,wall_pt_1,wall_pt_2,code=0)==False) or \
                   ((beam_pt_2==wall_pt_1 or beam_pt_2==wall_pt_2) and point_is_on_line(beam_pt_1,wall_pt_1,wall_pt_2,code=0)==False):#beam和wall必须有且仅有一个交点,且beam的令一点不能在wall内部
                    #print(f"依据beam{beam}生成shear_wall{wall}")
                    sh_pre_list.append(LineString([wall_pt_1,wall_pt_2]))
   
    #基于初始的sh_pre_list判断
    for sh in shs:
        sh_action={"coords":(int(sh[0]),int(sh[1]),int(sh[2]),int(sh[3]))}
        sh_line=LineString([(sh[0],sh[1]),(sh[2],sh[3])])
        IS_VALID,ERROR_TYPE=judge_shearwall_action_is_valid(sh_action=sh_action,walls=walls,sh_pre_list=sh_pre_list)
        if IS_VALID==True:
            sh_pre_list.append(sh_line)
        else:
            sh_text=f"<shearwall>({int(sh[0])},{int(sh[1])}),({int(sh[2])},{int(sh[3])})"
            #print(f"{sh_text}不合规")
            if ERROR_TYPE==0:#如果不是wall的子线段，则全部删除
                completion_predict=completion_predict.replace(sh_text,"")
            elif ERROR_TYPE==1:#如果是覆盖错误，则只删除一条线即可
                completion_predict=completion_predict.replace(sh_text,"",1)
    return completion_predict

def process_completion_predict_based_on_response(context,completion_predict,response):
    action_context=completion_predict
    #处理response,将操作按原始含义添加在completion_predict
    pattern = r"<(add|remove)><(.*?)>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
    raw_actions=re.findall(pattern,response)
    for action in raw_actions:
        if action[1]=="beam" or action[1]=="beams":
            if action[0]=="add":
                add_action_text=f"<beam>({int(action[2])},{int(action[3])}),({int(action[4])},{int(action[5])})"
                action_context=action_context+add_action_text
            if action[0]=="remove":
                minus_action_text=f"<beam>({int(action[2])},{int(action[3])}),({int(action[4])},{int(action[5])})"
                if minus_action_text in action_context:
                    action_context=action_context.replace(minus_action_text,"")
        elif action[1]=="shearwall":
            if action[0]=="add":
                add_action_text=f"<shearwall>({int(action[2])},{int(action[3])}),({int(action[4])},{int(action[5])})"
                action_context=action_context+add_action_text
            elif action[0]=="remove":
                minus_action_text=f"<shearwall>({int(action[2])},{int(action[3])}),({int(action[4])},{int(action[5])})"
                if minus_action_text in action_context:
                    action_context=action_context.replace(minus_action_text,"")

    #移除重复的beam和重复或者不在操作域的shearwall
    action_context=filter_completion_predict(action_context,context)
    return action_context

def get_walls_context_from_context(context):
    pattern=r"<wall>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
    walls=re.findall(pattern,context)
    walls_context=""
    for wall in walls:
        walls_context=walls_context+f"<wall>({int(wall[0])},{int(wall[1])}),({int(wall[2])},{int(wall[3])})"
    return walls_context

if __name__ == "__main__":
    start_time = time.time()

    model = AutoModelForCausalLM.from_pretrained(mode_path,device_map="auto",
                                                 cache_dir=cache_dir,torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(mode_path,cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    '''
    #保存base model
    save_dir = "five_plus_two_optimization/train_model/GRPO/base_model/qwen1.5b"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    '''
    model.eval()
    
    # ------------------------
    #构建测试数据
    # ------------------------
    #训练集输出
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(i)
            record = json.loads(line)
            context = record["context"]
            completion_predict = record["completion_predict"]
            completion_predict_filtered=filter_completion_predict(completion_predict=completion_predict,context=context)
            #print(f"completion_predict:{completion_predict}")
            #print(f"completion_predict_filtered:{completion_predict_filtered}")
            house=record["house"]
            floor=record["floor"]
            bound=record["bound"]
            #prompt构造
            prompt="You are an architectural structural assistant.\n"
            #prompt=prompt+"Your task is to modify BEAMS and SHEARWALLS in a given 2D architectural line layout."
            prompt=prompt+"Available operations (STRICT FORMAT):\n"
            prompt=prompt+"<add><beam>(x1,y1),(x2,y2)</beam>\n" 
            prompt=prompt+"<remove><beam>(x1,y1),(x2,y2)</beam>\n" 
            prompt=prompt+"<add><shearwall>(x1,y1),(x2,y2)</shearwall> (the shearwall must in wall)\n"
            prompt=prompt+"<remove><shearwall>(x1,y1),(x2,y2)</shearwall>\n"
            #prompt=prompt+"Your task is to add or remove beams or shearwalls in initial structure, and the shearwalls,beams and exterior walls form the diaphragms to split the whole structure.\n"
            prompt=prompt+"Input format:\n"
            prompt=prompt+"<type>(x1,y1),(x2,y2) where type ∈ {wall,exterior_wall, beam,shearwall, opening, truss_area, joist_area}\n"
            prompt=prompt+"Rules:\n"
            prompt=prompt+"1. ONLY output beam/shearwall modifications using the exact formats above\n"
            prompt=prompt+"2. Do NOT modify walls, exterior_walls, openings, or boundary lines.\n"
            prompt=prompt+"3. Beams: must be supported by walls/beams at both ends.\n"
            prompt=prompt+"4. Shearwalls: can ONLY be added by converting existing walls (coordinates must match an existing wall line)\n"
            prompt=prompt+"5. truss_area/joist_area: the boundary of the polygon\n"
            prompt=prompt+"Output format (STRICT):\n"
            prompt=prompt+"-Only output modified beams using:\n"
            prompt=prompt+"<add><beam>(x1,y1),(x2,y2)</beam>\n" 
            prompt=prompt+"<remove><beam>(x1,y1),(x2,y2)</beam>\n" 
            prompt=prompt+"<add><shearwall>(x1,y1),(x2,y2)</shearwall> (the shearwall must in wall)\n"
            prompt=prompt+"<remove><shearwall>(x1,y1),(x2,y2)</shearwall>\n"
            prompt=prompt+"NO explanations.\n"
            prmompt=prompt+"NO input repetition.\n"
            prompt=prompt+"NO additional text.\n"
            
            prompt=prompt+f"The initial structure:{context}\n"
            prompt=prompt+f"shearwalls and beams that have been added:{completion_predict_filtered}\n"
            
            prompt=f"<s>{prompt}</s>"
            #print(context,completion_predict)
            
            prompt_tokens = len(tokenizer(prompt).input_ids)
            print("Prompt tokens:", prompt_tokens)
            
            inputs = tokenizer(prompt,return_tensors="pt").to("cuda").input_ids  # prefix
            input_len = inputs.shape[1]
            num_tokens = 200 #var
            response=""
            
            #生成回答
            gen = model.generate(inputs, max_new_tokens=num_tokens, do_sample=False)
            #generated_ids = gen[0,input_len:]
            generated_ids = gen[0,input_len:]
            generated_tokens = tokenizer.decode(generated_ids)
            response+=generated_tokens
            
            print(f"response:{response}")
        
            processed_predict=process_completion_predict_based_on_response(context=context,
                completion_predict=completion_predict_filtered,response=response)
            print(f"processed_predict:{processed_predict}")
            
            with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"house":house,"floor":floor,"bound":bound,
                                    "context": context,"completion_predict":completion_predict_filtered,
                                    "prompt":prompt,"response":response,
                                    "processed_predict":processed_predict,}
                                    ,ensure_ascii=False) + "\n")
            #break
                
    end_time = time.time()
    print(f"代码平均执行时间：{(end_time - start_time)/(i+1):.4f} 秒")