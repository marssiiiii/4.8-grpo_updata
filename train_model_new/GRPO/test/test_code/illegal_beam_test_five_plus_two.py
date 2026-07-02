import json
import re
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("/mnt/efs/jiuxing_li/five_plus_two_optimization/train_model/base_code")
sys.path.append("/mnt/efs/jiuxing_li")
from reward_design import judge_beam_is_valid #judeg_beam_is_valid(beam,house_areas,shear_walls,beams,openings,BD_HAS_DIAG)
from reward_design import process_completion_predict_based_on_response,bd_has_diagnal
from reward_design import get_shear_walls_from_segments # get_shear_walls_from_segments(walls,exterior_walls,beams)
from standard_function import get_segments_info_five_plus_two 

POLYGON_TYPES = ["opening","joist_area","truss_area"]
LINE_TYPES = ["wall","beams","exterior_wall"]
POINT_TYPES = []

if __name__=="__main__":    
    INPUT_JSON=f"five_plus_two_optimization/train_model/GRPO/test/test_jsonl/grpo_fpt_6_3_2000.jsonl"
    OUTPUT_DIR="five_plus_two_optimization/train_model/GRPO/test/illegal_pic"
    
    total_beam_num,illegal_beam_num=0,0
    error_type=["out_bd","wr_dir","crs_long","no_dep","cov_shear","inter_open"]
    error_nums=[0,0,0,0,0,0]
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            #读取信息
            record = json.loads(line)
            house,floor,context,completion_predict,prompt,response = record["house"],record["floor"],record["context"],\
            record["completion_predict"],record['prompt'],record["response"]
            print(f"Record{i},{house},{floor}")
            answer=process_completion_predict_based_on_response(completion_predict,response)
            house_items=get_segments_info_five_plus_two(context,POLYGON_TYPES,LINE_TYPES,POINT_TYPES)
            predict_structures=get_segments_info_five_plus_two(answer,POLYGON_TYPES,LINE_TYPES,POINT_TYPES)
            exterior_walls,walls,beams,openings,house_areas=[],[],[],[],[]
            for seg in house_items:
                if seg[0]=="exterior_wall":
                    exterior_walls.append(seg)
                elif seg[0]=="wall":
                    walls.append(seg)
                elif seg[0]=='opening':
                    openings.append(seg)
                elif seg[0]=="joist_area" or seg[0]=='truss_area':
                    house_areas.append(seg)
            for seg in predict_structures:
                if seg[0]=="beams":
                    beams.append(seg)
            #判断每条beam是否有效
            beams_pre=[]
            for beam in beams:
                total_beam_num=total_beam_num+1
                beams_pre.append(beam)
                shear_walls=get_shear_walls_from_segments(walls,exterior_walls,beams_pre)
                BD_HAS_DIAG=bd_has_diagnal(house_areas=house_areas)
                BEAM_IS_VALID,illustration,ERROR_CODE=judge_beam_is_valid(beam=beam,house_areas=house_areas,openings=openings,
                        shear_walls=shear_walls,beams=beams,BD_HAS_DIAG=BD_HAS_DIAG) #计算的是final illegal
                if BEAM_IS_VALID==False:
                    illegal_beam_num=illegal_beam_num+1
                    print(f"非法beam:{illustration}")
                    error_nums[ERROR_CODE]=error_nums[ERROR_CODE]+1
            
    print(f"非法beam数:{illegal_beam_num},占比为:{illegal_beam_num/total_beam_num}")
    for i in range(6):
        print(f"{error_type[i]}发生{error_nums[i]}次,占总error{error_nums[i]/illegal_beam_num}")

    #绘制各类错误发生次数
    plt.figure()
    x=np.arange(len(error_nums))
    plt.bar(x, error_nums)

    plt.xlabel(error_type)
    plt.ylabel("ERROR NUM")
    plt.title(f"ERROR DISTRIBUTION")

    plt.savefig(f"{OUTPUT_DIR}/grpo_fpt_6_3_error_pic.png")


    #print(orientation((64,64),(0,64),(64,64)))
    #print(point_is_on_line((64,0),(64,64),(64,64),2))
    #print(False+True+True)