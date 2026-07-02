import json
import os
from shapely.geometry import LineString
from typing import List
from embedding_define_five_plus_two_3_11 import StructuralSegment, Segment_Embedding
from house_item_embedding_five_plus_two_3_11 import *
import re
import sys
sys.path.append("/mnt/efs/jiuxing_li")
sys.path.append("/mnt/efs/jiuxing_li/five_plus_two_optimization/train_model_new/base_code")
from standard_function import get_segments_info_five_plus_two,get_bound_from_context_five_plus_two
from reward_design import get_total_area
POLYGON_TYPES = ["opening","inoutbox"]
LINE_TYPES = ["wall","beam","shearwall"]
POINT_TYPES = []
GENERATE_NUM=1
# 生成逐步预测数据
def generate_stepwise_data(context,completion_predict,design_code,output_path,house_name,floor_name):
    #统计context中x,y范围
    bound=get_bound_from_context_five_plus_two(context,POLYGON_TYPES=POLYGON_TYPES,
                LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES)
    # 保存为 JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f: #追加文件而不是覆盖
        wrapped = {
            "house":house_name,
            "floor":floor_name,
            "design":design_code,
            "bound":bound,
            "context":context,
            "completion_predict":completion_predict,
        }
        f.write(json.dumps(wrapped, ensure_ascii=False) + "\n")

    print(f"已将{house_name}_{floor_name}_{design_code} 写入到{output_path}")

def floor_is_end(segments,i):
    pattern = r"\b([fb]\d+)(?=,)"

    match1 = re.search(pattern, segments[i])
    if match1:
        floor1 = match1.group(1)
    else:
        floor1=None

    if i==len(segments)-1:
        return floor1,True
    
    match2 = re.search(pattern, segments[i+1])
    if match2:
        floor2=match2.group(1)
    else:
        floor2=None

    return (floor1,bool(floor1!=floor2))

def total_area_is_zero(context):
    #提取房屋基本元素和设计处理完成的beam
    house_items=get_segments_info_five_plus_two(context,POLYGON_TYPES,LINE_TYPES,POINT_TYPES)
    house_areas=[]
    for seg in house_items:
        if seg[0]=="inoutbox":
            house_areas.append(seg)
    total_area=get_total_area(house_areas=house_areas)
    if total_area==0:
        return True
    else:
        return False

def not_one_inoutbox(context):
    house_items=get_segments_info_five_plus_two(context,POLYGON_TYPES,LINE_TYPES,POINT_TYPES)
    house_areas=[]
    for seg in house_items:
        if seg[0]=="inoutbox":
            house_areas.append(seg)
    if len(house_areas)!=1:
        print(f"{len(house_areas)}含有不唯一的inoutbox")
        return True
    return False

if __name__ == "__main__":
    input_folder_path="house_data/noopening_test_detail"
    output_path=f"train_json_data/five_plus_two_train_jsonl_data/design_3.27/initial_data_1.jsonl"
    for filename in os.listdir(input_folder_path):
        print(filename)
        for design_code in range(GENERATE_NUM):
            input_path_item=f"{input_folder_path}/{filename}/house_items.json"

            segments_item=process_dir(house_item_get_segments,input_path_item) #读取房屋基础数据
            if segments_item==-1:
                print("house_item为空")
                continue
            list_context=house_item_format_segments_grouped(segments_item)

            input_path_stru=f"{input_folder_path}/{filename}/design_{design_code+1}.jsonl" #读取建筑元素数据
            try:
                segments_stru=process_dir(house_item_get_segments,input_path_stru)
            except:
                print("structure数据有问题")
                continue
            if segments_stru==-1:
                print("structure为空")
                continue
            list_predict=house_item_format_segments_grouped(segments_stru)
            
            #组合每一楼层信息
            index_i=0
            index_i,index_j=0,0
            while(index_i<len(list_context) and index_j<len(list_predict)):
                context,completion_predict="",""
                #构造floor中的context
                for i in range(index_i,len(list_context)):
                    floor_1,is_end_floor_1=floor_is_end(list_context,i)
                    index_i=i
                    info=re.sub(r",?[fb]\d+,?", "", list_context[i])
                    #print(info)
                    context+=info
                    #print(context)
                    if is_end_floor_1:
                        break
                #构造floor中的completion_predict
                for j in range(index_j,len(list_predict)):
                    floor_2,is_end_floor_2=floor_is_end(list_predict,j)
                    index_j=j
                    info=re.sub(r",?[fb]\d+,?", "", list_predict[j])
                    #print(info)
                    completion_predict+=info
                    #print(context)
                    if is_end_floor_2:
                        break
                
                if context=="" or completion_predict=="":
                    print("空数据！")
                    if context=="":
                        index_j=index_j+1
                    if completion_predict=="":
                        index_i=index_i+1
                    continue
                if floor_1!=floor_2:
                    print(f"{floor_1}!={floor_2}")
                    index_i=index_i+1
                    index_j=index_j+1
                    continue
                if not_one_inoutbox(context)==True:
                    index_i=index_i+1
                    index_j=index_j+1
                    continue
                if total_area_is_zero(context)==True:
                    print(f"house_{filename},floor_{floor_1},total area is zero")
                    index_i=index_i+1
                    index_j=index_j+1
                    continue
                #构造
                generate_stepwise_data(context=context,completion_predict=completion_predict,design_code=design_code+1,
                                       output_path=output_path,house_name=filename,floor_name=floor_1)
                index_i=index_i+1
                index_j=index_j+1
            #break
        #break