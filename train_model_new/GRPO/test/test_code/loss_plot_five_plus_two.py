import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def acquire_kl(input_dir):
    steps,kls=[],[]
    with open(input_dir, "r", encoding="utf-8") as f:
        for i,line in enumerate(f):
            record=json.loads(line)
            if "per_token_kl" in record:
                step,kl=record["step"],record["per_token_kl"]
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

def acquire_advantage(input_dir):
    steps,advantages=[],[]
    with open(input_dir, "r", encoding="utf-8") as f:
        for i,line in enumerate(f):
            record=json.loads(line)
            if "advantage" in record:
                step,advantage=record["step"],record["advantage"]
                flat_data=[]  # 提取每个子列表里的数字
                for x in advantage:
                    flat_data.append(x)
                steps.append(step)
                advantages.append(np.mean(flat_data).item())
    return steps,advantages

def acquire_loss(input_dir):
    steps,losses=[],[]
    with open(input_dir, "r", encoding="utf-8") as f:
        for i,line in enumerate(f):
            record=json.loads(line)
            if "loss_val" in record:
                step,loss=record["step"],record["loss_val"]
                steps.append(step)
                losses.append(loss)
    return steps,losses

def plot_and_save_window(steps,values,output_dir,str,window=10):
    avg_steps,avg_rewards=[],[]
    for i in range(0, len(values), window):
        chunk_rewards = values[i:i + window]
        chunk_steps = steps[i:i + window]

        if len(chunk_rewards) < window:
            break  # 可选：丢弃不足 window 的尾部

        avg_rewards.append(np.mean(chunk_rewards).item())
        avg_steps.append(int(np.mean(chunk_steps)))
    plot_and_save(avg_steps,avg_rewards,output_dir,str)
    return avg_steps,avg_rewards

def plot_and_save(steps,values,output_dir,str):
    print(len(steps))
    # 绘制 loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, marker=".",color="red")
    plt.plot()
    plt.xlabel("Steps")
    plt.ylabel(f"{str}")
    plt.title(f"GRPO_Training {str} Curve")
    plt.grid(True)
    plt.savefig(output_dir,dpi=200)

def acquire_rewards(input_dir):
    steps,rewards=[],[]
    with open(input_dir, "r", encoding="utf-8") as f:
        for i,line in enumerate(f):
            record=json.loads(line)
            if "per_token_kl" in record:
                step,reward=record["step"],record["rewards"]
                steps.append(step)
                rewards.append(reward[0])
    return steps,rewards

def acquire_values_dict(input_dir,key_list):
    values_dict={key:[] for key in key_list}
    steps_list=[]
    houses_list=[]
    with open(input_dir, "r", encoding="utf-8") as f:
        for i,line in enumerate(f):
            record=json.loads(line)
            if not any(key in record for key in key_list):
                continue
            steps_list.append(record.get("step", i))
            house=record.get("house","unknown")
            if isinstance(house,list):
                house=house[0] if house else "unknown"
            houses_list.append(str(house))
            for key in key_list:
                if key in record:
                    values=record[key]
                    if key=="floor_price_list":
                        values_dict[key].append(values[0]*1e-3)
                    elif key=="invalid_process_num_list":
                        values_dict[key].append(values[0])
                    elif key=="dcr_delta_list":
                        values_dict[key].append(values[0]*1e3)
                    elif key=="error_num_list":
                        values_dict[key].append(values[0])
                    elif key=="diaphragm_delta_score_list":
                        values_dict[key].append(values[0]*1e1)
                    else:
                        values_dict[key].append(values[0]*1e1)
                else:
                    values_dict[key].append(np.nan)
    return values_dict,steps_list,houses_list

def acquire_values_socres_dict(values_dict,PARAMETERS):
    def get_design_price_score(floor_price_list, diaphragm_area_ratio_list, error_num_list):
        floor_price = np.array(floor_price_list)
        area_ratio = np.array(diaphragm_area_ratio_list)
        error_num = np.array(error_num_list)
        n = len(floor_price)

        sorted_idx = np.argsort(floor_price)
        rank = np.empty(n)
        rank[sorted_idx] = np.arange(n)
        rank_norm = rank / (n - 1) #rank_norm属于[0,1]

        score = (1 - rank_norm) * area_ratio * np.exp(-error_num)
        return score.tolist()

    floor_design_score_list=[]
    design_error_score_list,design_invalid_process_score_list,design_diaphragm_score_list,design_dcr_score_list,design_price_score_list=[],[],[],[],[]
    design_price_score_list=get_design_price_score(values_dict["floor_price_list"], 
        values_dict["diaphragm_area_ratio"], values_dict["error_num_list"])
    for i in range(len(values_dict["diaphragm_delta_score_list"])):
        delta_diaphragm, delta_dcr, error_num, invalid_process_num,diaphragm_area_ratio,response_len_ratio = \
            values_dict["diaphragm_delta_score_list"][i], values_dict["dcr_delta_list"][i],\
            values_dict["error_num_list"][i], values_dict["invalid_process_num_list"][i],\
            values_dict["diaphragm_area_ratio"][i], 0.25
        design_error_score=-0.5*error_num
        design_invalid_process_score=-0.2*invalid_process_num
        if delta_diaphragm!=0:
            design_diaphragm_score=2*delta_diaphragm/response_len_ratio
        else:
            design_diaphragm_score=-0.5*response_len_ratio
        design_dcr_score=delta_dcr*diaphragm_area_ratio*math.exp(-error_num)
        design_price_score=design_price_score_list[i]
        floor_design_score=PARAMETERS["lambda_1"]*design_error_score+PARAMETERS["lambda_2"]*design_invalid_process_score+\
            PARAMETERS["lambda_3"]*design_diaphragm_score+PARAMETERS["lambda_4"]*design_dcr_score+\
            PARAMETERS["lambda_5"]*design_price_score
        floor_design_score=max(-1*PARAMETERS["SCORE_RANGE"],min(PARAMETERS["SCORE_RANGE"],floor_design_score))

        floor_design_score_list.append(floor_design_score)
        design_error_score_list.append(design_error_score)
        design_invalid_process_score_list.append(design_invalid_process_score)
        design_diaphragm_score_list.append(design_diaphragm_score)
        design_dcr_score_list.append(design_dcr_score)
    
    values_scores_dict={
        "floor_design_score_list": floor_design_score_list,
        "design_error_score_list": design_error_score_list,
        "design_invalid_process_score_list": design_invalid_process_score_list,
        "design_diaphragm_score_list": design_diaphragm_score_list,
        "design_dcr_score_list": design_dcr_score_list,
        "design_price_score_list": design_price_score_list
    }
    return values_scores_dict
    
def plot_values_dict(steps, values_dict,output_path,window=1):
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(values_dict)))
    for idx, (key, values) in enumerate(values_dict.items()):
        #print(key,len(values))
        avg_steps,avg_values=[],[]
        for i in range(0, len(values), window):
            chunk_rewards = values[i:i + window]
            chunk_steps = steps[i:i + window]
            if len(chunk_rewards) < window:
                break
            avg_values.append(np.mean(chunk_rewards).item())
            avg_steps.append(int(np.mean(chunk_steps)))
        plt.plot(avg_steps,avg_values,label=key,color=colors[idx],linewidth=2)
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Values")
    plt.title(f"Values over Steps_window_{window}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)

def plot_top6_houses(steps,values_dict,houses,output_path,window=1):
    from collections import Counter
    house_counts=Counter(houses)
    top6=[h for h,_ in house_counts.most_common(6)]

    steps_arr=np.array(steps)
    houses_arr=np.array(houses)
    colors=plt.cm.tab10(np.linspace(0,1,len(values_dict)))

    fig,axes=plt.subplots(2,3,figsize=(18,10))
    axes=axes.flatten()

    for plot_idx,house in enumerate(top6):
        ax=axes[plot_idx]
        mask=houses_arr==house
        house_steps=steps_arr[mask].tolist()

        for key_idx,(key,vals) in enumerate(values_dict.items()):
            vals_arr=np.array(vals,dtype=float)
            house_vals=vals_arr[mask].tolist()

            avg_steps,avg_vals=[],[]
            for i in range(0,len(house_vals),window):
                chunk=house_vals[i:i+window]
                chunk_steps=house_steps[i:i+window]
                if len(chunk)<window:
                    break
                avg_vals.append(np.nanmean(chunk))
                avg_steps.append(int(np.mean(chunk_steps)))

            if avg_steps:
                ax.plot(avg_steps,avg_vals,label=key,color=colors[key_idx],linewidth=1.5)

        ax.set_title(f"House: {house}  (n={house_counts[house]})")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Values")
        ax.legend(fontsize=6)
        ax.grid(True,linestyle="--",alpha=0.5)

    plt.suptitle(f"Top 6 Houses — Values over Steps (window={window})",fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path,dpi=200)
    plt.close()

def correlation_analysis(values_dict,output_path):
    df = pd.DataFrame({k:[x[0] if isinstance(x,list) else x for x in v]
                    for k,v in values_dict.items()})
    corr = df.corr()
    plt.figure(figsize=(8,6))
    plt.imshow(corr)
    plt.xticks(range(len(corr)), corr.columns, rotation=45)
    plt.yticks(range(len(corr)), corr.columns)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)

def correlation_analysis_1(values_dict,output_path):
    df = pd.DataFrame(values_dict)
    trend_corr = pd.DataFrame(index=df.columns, columns=df.columns)
    for c1 in df.columns:
        for c2 in df.columns:
            s1 = np.sign(np.diff(df[c1]))
            s2 = np.sign(np.diff(df[c2]))
            trend_corr.loc[c1, c2] = np.mean(s1 == s2)
    corr = trend_corr.astype(float)
    plt.figure(figsize=(8,6))
    plt.imshow(corr)
    plt.xticks(range(len(corr)), corr.columns, rotation=45)
    plt.yticks(range(len(corr)), corr.columns)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)

if __name__=="__main__":
    input_dir="five_plus_two_optimization/train_model_new/GRPO/save_model/grpo_fpt_api/grpo_fpt_api_4_1/loss_history.jsonl"
    output_dir="five_plus_two_optimization/train_model_new/GRPO/save_model/grpo_fpt_api/grpo_fpt_api_4_1"

    steps,kl=acquire_kl(input_dir)
    print(steps[:5],kl[:5])
    plot_and_save(steps,kl,f"{output_dir}/loss_curve_kl.png","kl")
    plot_and_save_window(steps,kl,f"{output_dir}/loss_curve_kl_window_10.png","kl",10)
    plot_and_save_window(steps,kl,f"{output_dir}/loss_curve_kl_window_50.png","kl",50)

    steps,objs=acquire_per_token_obj(input_dir)
    print(steps[:5],objs[:5])
    plot_and_save(steps,objs,f"{output_dir}/loss_curve_objs.png","objs")
    objs_10_steps,objs_10_values=plot_and_save_window(steps,objs,f"{output_dir}/loss_curve_objs_window_10.png","objs",10)
    plot_and_save_window(steps,objs,f"{output_dir}/loss_curve_objs_window_50.png","objs",50)

    values_dict,steps,houses=acquire_values_dict(input_dir,["diaphragm_delta_score_list","dcr_delta_list",
                "error_num_list","invalid_process_num_list","floor_price_list","diaphragm_area_ratio"])
    values_dict_1,steps_1,houses_1=acquire_values_dict(input_dir,["design_error_score_list","design_invalid_process_score_list",
                "design_diaphragm_score_list","deisgn_dcr_score_list","design_price_score_list"])
    PARAMETERS={"lambda_1": 1, "lambda_2": 0.5, "lambda_3": 2, "lambda_4": 10,"lambda_5": 1, #var
                "MAX_TOKEN_NUM": 200,"SCORE_RANGE":1.5,"TEMPERATURE":0.8}
    values_scores_dict=acquire_values_socres_dict(values_dict=values_dict,PARAMETERS=PARAMETERS)
    for key,values in values_scores_dict.items():
        print(key,len(values),values[:5])
    plot_values_dict(steps,values_dict,f"{output_dir}/loss_curve_values_dict.png")
    plot_values_dict(steps,values_dict,f"{output_dir}/loss_curve_values_dict_window_5.png",window=5)
    plot_values_dict(steps,values_dict,f"{output_dir}/loss_curve_values_dict_window_10.png",window=10)
    plot_values_dict(steps,values_dict,f"{output_dir}/loss_curve_values_dict_window_50.png",window=50)
    plot_values_dict(steps_1,values_dict_1,f"{output_dir}/loss_curve_values_dict_std_score.png")
    plot_values_dict(steps_1,values_dict_1,f"{output_dir}/loss_curve_values_dict_std_score_window_5.png",window=5)
    plot_values_dict(steps_1,values_dict_1,f"{output_dir}/loss_curve_values_dict_std_score_window_10.png",window=10)
    plot_values_dict(steps_1,values_dict_1,f"{output_dir}/loss_curve_values_dict_std_score_window_50.png",window=50)
    plot_top6_houses(steps,values_dict,houses,f"{output_dir}/top6_houses_values_dict.png")
    plot_top6_houses(steps,values_dict,houses,f"{output_dir}/top6_houses_values_dict_window_5.png",window=5)
    plot_top6_houses(steps,values_dict,houses,f"{output_dir}/top6_houses_values_dict_window_10.png",window=10)
    plot_top6_houses(steps_1,values_dict_1,houses_1,f"{output_dir}/top6_houses_values_dict_1.png")
    plot_top6_houses(steps_1,values_dict_1,houses_1,f"{output_dir}/top6_houses_values_dict_1_window_5.png",window=5)
    plot_top6_houses(steps_1,values_dict_1,houses_1,f"{output_dir}/top6_houses_values_dict_1_window_10.png",window=10)
    correlation_analysis(values_scores_dict,f"{output_dir}/correlation_analysis.png")
    correlation_analysis_1(values_scores_dict,f"{output_dir}/correlation_analysis_1.png")