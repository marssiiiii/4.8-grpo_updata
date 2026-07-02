import re
import json
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
sys.path.append("/mnt/efs/jiuxing_li")
sys.path.append("/mnt/efs/jiuxing_li/five_plus_two_optimization/train_model_new/base_code")
#----------------------------------
#获取item,structure 的segments信息
#----------------------------------
from standard_function import get_segments_info_five_plus_two
from reward_design import process_completion_predict_based_on_response,get_design_score
POLYGON_TYPES = ["opening","inoutbox"]
LINE_TYPES = ["wall","beams","exterior_wall"]
POINT_TYPES = []

def acquire_rewards(input_dir):
    steps,rewards=[],[]
    with open(input_dir, "r", encoding="utf-8") as f:
        for i,line in enumerate(f):
            record=json.loads(line)
            if "per_token_kl" in record:
                step,reward=record["step"],record["rewards"]
                steps.append(step)
                rewards.append(reward[0][0])
    return steps,rewards

def acquire_kl(input_dir,bottom=0):
    steps,kls=[],[]
    with open(input_dir, "r", encoding="utf-8") as f:
        for i,line in enumerate(f):
            record=json.loads(line)
            if "per_token_kl" in record:
                step,kl=record["step"],record["per_token_kl"]
                if step<bottom:
                    continue
                flat_data=[]  # 提取每个子列表里的数字
                for x in kl:
                    flat_data.append(np.mean(x).item())
                steps.append(step)
                #print(flat_data,np.mean(flat_data))
                kls.append(np.mean(flat_data).item())
    return steps,kls

def acquire_per_token_obj(input_dir,bottom=0):
    steps,objs=[],[]
    with open(input_dir, "r", encoding="utf-8") as f:
        for i,line in enumerate(f):
            record=json.loads(line)
            if "per_token_kl" in record:
                step,kl=record["step"],record["per_token_obj"]
                if step<bottom:
                    continue
                flat_data=[]  # 提取每个子列表里的数字
                for x in kl:
                    flat_data.append(np.mean(x).item())
                steps.append(step)
                #print(flat_data,np.mean(flat_data))
                objs.append(np.mean(flat_data).item())
    return steps,objs

def window_value(steps,values,window=10):
    avg_steps,avg_rewards=[],[]
    for i in range(0, len(values), window):
        chunk_rewards = values[i:i + window]
        chunk_steps = steps[i:i + window]

        if len(chunk_rewards) < window:
            break  # 可选：丢弃不足 window 的尾部

        avg_rewards.append(np.mean(chunk_rewards).item())
        avg_steps.append(int(np.mean(chunk_steps)))
    return avg_steps,avg_rewards

def visualize_promote(initial_list, list1, list2, save_path, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # =======================
    # ax1: 绘制 initial_list
    # =======================
    initial_list=sorted(initial_list)
    x0 = np.arange(len(initial_list))
    ax1.bar(x0, initial_list)
    ax1.set_ylim(-1, 1)
    ax1.set_xlabel("House Demo")
    ax1.set_ylabel("Initial Score")
    ax1.set_title("Initial Score")

    # =======================
    # ax2: 排序后绘制优化前后对比
    # =======================
    sorted_pairs = sorted(zip(list1, list2))
    list1_sorted, list2_sorted = zip(*sorted_pairs)

    list1_sorted = list(list1_sorted)
    list2_sorted = list(list2_sorted)

    x = np.arange(len(list1))
    width = 0.35

    ax2.bar(x - width/2, list1_sorted, width, label='Before')
    ax2.bar(x + width/2, list2_sorted, width, label='After')

    ax2.set_ylim(-1, 1)
    ax2.set_xlabel("House Demo")
    ax2.set_ylabel("House Score")
    ax2.set_title(f"{title} Visualization")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_score_from_data(set_name,load_path,output_path_1=None,output_path_2=None):
    OPT_HOUSE_SCORE_BEFORE_LIST,OPT_HOUSE_SCORE_AFTER_LIST,INITIAL_SCORE=[],[],[]
    DEC_HOUSE_SCORE_BEFORE_LIST,DEC_HOUSE_SCORE_AFTER_LIST=[],[]
    TOTAL_NUM,PROMOTE_NUM,DEC_NUM=0,0,0
    
    for index in range(1):
        with open(load_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if set_name=="train" and i>=50:
                    continue
                elif set_name=="test" and i<50:
                    continue
                elif set_name=="all":
                    print("全集结果")
                record = json.loads(line)
                house,floor,context,completion_predict,prompt,answer = record["house"],record["floor"],record["context"],\
                record["completion_predict"],record['prompt'],record["answer"]
                #answer=process_completion_predict_based_on_response(context=context,completion_predict=completion_predict,response=response)
                print(i+1,house,floor)
    
                #得到设计前的房屋整体设计分数
                result_before=get_design_score(context,completion_predict)
                house_score_before,area_ratio_before,\
                =result_before["house_score"],result_before["area_ratio"]

                #得到设计后的房屋整体设计分数
                result_after=get_design_score(context,answer)
                house_score_after,area_ratio_after,beams_after,shear_walls_after,valid_diaphragms_after,\
                =result_after["house_score"],result_after["area_ratio"],result_after["beams"],\
                result_after["shear_walls"],result_after["valid_diaphragms"]

                #记录分数
                TOTAL_NUM=TOTAL_NUM+1
                INITIAL_SCORE.append(house_score_before)
                if house_score_after>house_score_before:
                    OPT_HOUSE_SCORE_BEFORE_LIST.append(house_score_before)
                    OPT_HOUSE_SCORE_AFTER_LIST.append(house_score_after)
                    PROMOTE_NUM=PROMOTE_NUM+1
                elif house_score_after<house_score_before:
                    DEC_HOUSE_SCORE_BEFORE_LIST.append(house_score_before)
                    DEC_HOUSE_SCORE_AFTER_LIST.append(house_score_after)
                    DEC_NUM=DEC_NUM+1
        
        print(f"{set_name}所有房屋优化前平均分数：{np.mean(INITIAL_SCORE)}")
        print(f"{set_name}被优化的房屋优化前平均分数：{np.mean(OPT_HOUSE_SCORE_BEFORE_LIST)}")
        print(f"{set_name}被优化的房屋优化后平均分数：{np.mean(OPT_HOUSE_SCORE_AFTER_LIST)}")
        print(f"{set_name}优化房屋占比：{PROMOTE_NUM}/{TOTAL_NUM}")
        print(f"{set_name}被降低分数的房屋优化前平均分数：{np.mean(DEC_HOUSE_SCORE_BEFORE_LIST)}")
        print(f"{set_name}被降低分数的房屋优化后平均分数：{np.mean(DEC_HOUSE_SCORE_AFTER_LIST)}")
        print(f"{set_name}降分房屋占比：{DEC_NUM}/{TOTAL_NUM}")
                        
        if PROMOTE_NUM>0 and output_path_1!=None:
            #os.makedirs(os.path.dirname(f"{folder_path}/promote_pic/grpo_fpt_new_3_1_2000/promote.png"), exist_ok=True)
            visualize_promote(INITIAL_SCORE,OPT_HOUSE_SCORE_BEFORE_LIST,OPT_HOUSE_SCORE_AFTER_LIST,
            f"{output_path_1}_{set_name}_promote.png","PROMOTE")
        if DEC_NUM>0 and output_path_2!=None:
            #os.makedirs(os.path.dirname(f"{folder_path}/promote_pic/grpo_fpt_new_3_1_2000/dec.png"), exist_ok=True)
            visualize_promote(INITIAL_SCORE,DEC_HOUSE_SCORE_BEFORE_LIST,DEC_HOUSE_SCORE_AFTER_LIST,
            f"{output_path_2}_{set_name}_dec.png","DECREASE")
        return {"PROMOTE_NUM":PROMOTE_NUM,"DEC_NUM":DEC_NUM}

if __name__=="__main__":
    
    set_name="all" #将base_model中的信息加入
    x,y_pro,y_dec,result_list=[],[],[],[]
    result_base=get_score_from_data(set_name=set_name,
        #load_path=f"five_plus_two_optimization/train_model_new/GRPO/test/test_jsonl/ernie_base_model_10_area_auged_filt.jsonl")
        load_path=f"five_plus_two_optimization/train_model_new/GRPO/test/test_jsonl/grpo_fpt_api_3_3/area_auged_filt/1000_step_auged_filt.jsonl")
    x.append(0)
    y_pro.append(result_base["PROMOTE_NUM"])
    y_dec.append(result_base["DEC_NUM"])
    result_list.append(result_base)
    
    
    # grpo_loss_path="five_plus_two_optimization/train_model_new/GRPO/save_model/grpo_fpt_new_6_2/loss_history.jsonl" #得到loss中的信息
    # kl_steps,kl_values=acquire_kl(grpo_loss_path,bottom=0)
    # kl_10_steps,kl_10_values=window_value(kl_steps,kl_values,10)
    # obj_steps,obj_values=acquire_per_token_obj(grpo_loss_path,bottom=0)
    # obj_10_steps,obj_10_values=window_value(obj_steps,obj_values,10)
    
    #set_name="all"
    #x,y_pro,y_dec,result_list=[],[],[],[]
    
    valid_index=[1,2,3,4,5,6,7,8]
    for i in valid_index: #加入模型结果
        load_path=f"five_plus_two_optimization/train_model_new/GRPO/test/test_jsonl/grpo_fpt_api_4_1/area_auged_filt/{(i)*1000}_step_area_auged_filt.jsonl"
        result=get_score_from_data(set_name=set_name,
        load_path=load_path,
        #,output_path_1=f"five_plus_two_optimization/train_model_new/GRPO/test/promote_pic/grpo_fpt_new_6_2/grpo_fpt_new_6_2_{(i)*1000}_step_auged_filt",
        #output_path_2=f"five_plus_two_optimization/train_model_new/GRPO/test/dec_pic/grpo_fpt_new_6_2/grpo_fpt_new_6_2_{(i)*1000}_step_auged_filt"
        )
        x.append((i)*1000)
        y_pro.append(result["PROMOTE_NUM"])
        y_dec.append(result["DEC_NUM"])
        result_list.append(result)

    #x,y=[1000,2000,3000],[10,20,30]
    print(len(x),len(y_pro),len(y_dec))
    print(result_list)
    print(x,y_pro,y_dec)
    output_path="five_plus_two_optimization/train_model_new/GRPO/save_model/grpo_fpt_api/grpo_fpt_api_4_1"
    
    plt.figure() #绘图
    plt.bar(x,y_pro,color="blue",width=500,label="promote num")
    #plt.plot(kl_10_steps,[kl_10_value*10 for kl_10_value in kl_10_values],color="red",marker='o',label="kl")
    plt.xlabel("STEP")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(f"{output_path}/{set_name}_promote_auged_filt.png")
    
    # plt.figure()
    # plt.bar(x,y_dec,color="red",width=500,label="decrease num")
    # plt.plot(kl_10_steps,[kl_10_value*10 for kl_10_value in kl_10_values],color="blue",marker='o',label="kl")
    # plt.xlabel("STEP")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.savefig(f"{output_path}/{set_name}_dec_kl_bar_auged_filt.png")
    #get_score_from_data(folder_path,version_name,"train")
    #get_score_from_data(folder_path,version_name,"test")
'''
    plt.figure()
    plt.bar(x,y_pro,color="blue",width=500,label="promote num")
    plt.plot(obj_10_steps,[obj_10_value*10+20 for obj_10_value in obj_10_values],color="red",marker='o',label="obj")
    plt.xlabel("STEP")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(f"{output_path}/{set_name}_promote_obj_bar_auged_filt.png")


    plt.figure()
    plt.bar(x,y_dec,color="red",width=500,label="decrease num")
    plt.plot(obj_10_steps,[obj_10_value*10+20 for obj_10_value in obj_10_values],color="blue",marker='o',label="obj")
    plt.xlabel("STEP")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(f"{output_path}/{set_name}_dec_obj_bar_auged_filt.png")
'''
    #get_score_from_data(folder_path,version_name,"train")
    #get_score_from_data(folder_path,version_name,"test")