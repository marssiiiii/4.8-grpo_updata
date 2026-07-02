import json
import sys
sys.path.append("/home/jiuxing_li")
from five_plus_two_optimization.train_model_new.base_code.reward_design_trunc import \
    process_completion_predict_filt,get_segments_info_five_plus_two,orientation,point_is_on_line,\
    judge_shearwall_action_is_valid,del_list_last_element,get_actions_and_index_info_index
from five_plus_two_optimization.train_model_new.base_code.reward_design_dcr_price_trunc import\
construct_prompt
from standard_function import shuffle_jsonl_simple
from shapely import LineString
import re

def filter_response(completion_predict,response):#对response进行过滤
    #读取walls
    walls=get_segments_info_five_plus_two(context,[],["wall"],[])
    action_results=get_actions_and_index_info_index(response=response)

    #过滤completion_predict中重复的beam,并将提取到的beam记录在beams中
    beams=get_segments_info_five_plus_two(completion_predict,[],["beam"],[])
    for beam_action_result in action_results:
        if beam_action_result["seg_type"]!="beam":
            continue
        act_type,coords,act_char_span=beam_action_result["act_type"],\
            beam_action_result["coords"],beam_action_result["act_char_span"]
        act_beam=["beam",(int(coords[0]),int(coords[1])),(int(coords[2]),int(coords[3]))]

        if act_type=="add":
            if act_beam in beams:
                response=response[:act_char_span[0]]+"*"*(act_char_span[1]-act_char_span[0])+response[act_char_span[1]:]
                print(f"要添加的{act_beam}已存在，删除该add操作")
            else:
                beams.append(["beam",(int(coords[0]),int(coords[1])),(int(coords[2]),int(coords[3]))])
        elif act_type=="remove":
            if act_beam not in beams:
                response=response[:act_char_span[0]]+"*"*(act_char_span[1]-act_char_span[0])+response[act_char_span[1]:]
                print(f"要删除的{act_beam}不存在，删除该remove操作")
            else:
                beams=del_list_last_element(lst=beams,elm=act_beam)

    #过滤completion_predict中不合法的shearwall
    sh_pre_list=[] #初始化sh_pre_list
    shearwalls=get_segments_info_five_plus_two(completion_predict,[],["shearwall"],[])
    for shearwall in shearwalls:
        sh_pre_list.append(LineString([shearwall[1],shearwall[2]]))
    for wall in walls: #被beam平行支撑的wall,都算作shear_wall
        wall_pt_1,wall_pt_2=wall[1],wall[2]
        for beam in beams:
            beam_pt_1,beam_pt_2=beam[1],beam[2]
            if orientation(wall_pt_1,wall_pt_2,beam_pt_1)==0 and orientation(wall_pt_1,wall_pt_2,beam_pt_2)==0:#beam和wall必须共线
                if ((beam_pt_1==wall_pt_1 or beam_pt_1==wall_pt_2) and point_is_on_line(beam_pt_2,wall_pt_1,wall_pt_2,code=0)==False) or \
                   ((beam_pt_2==wall_pt_1 or beam_pt_2==wall_pt_2) and point_is_on_line(beam_pt_1,wall_pt_1,wall_pt_2,code=0)==False):#beam和wall必须有且仅有一个交点,且beam的令一点不能在wall内部
                    #print(f"依据beam{beam}生成shear_wall{wall}")
                    sh_pre_list.append(LineString([wall_pt_1,wall_pt_2]))
    
    for shearwall_action_result in action_results:
        if shearwall_action_result["seg_type"]!="shearwall":
            continue
        act_type,coords,act_char_span=shearwall_action_result["act_type"],shearwall_action_result["coords"],\
            shearwall_action_result["act_char_span"]
        sh_line=LineString([(coords[0],coords[1]),(coords[2],coords[3])])
        act_shearwall=["shearwall",(int(coords[0]),int(coords[1])),(int(coords[2]),int(coords[3]))]
        if act_type=="add":
            IS_VALID,ERROR_TYPE=judge_shearwall_action_is_valid(sh_action=shearwall_action_result,walls=walls,sh_pre_list=sh_pre_list)
            if IS_VALID==True:
                sh_pre_list.append(sh_line)
                shearwalls.append(act_shearwall)
            else:
                print(f"要添加的{act_shearwall}非法，非法类行为{ERROR_TYPE},删除该非法add操作")
                response=response[:act_char_span[0]]+"*"*(act_char_span[1]-act_char_span[0])+response[act_char_span[1]:]
        elif act_type=="remove":
            if act_shearwall in shearwalls:
                shearwalls=del_list_last_element(lst=shearwalls,elm=act_shearwall)
            else:
                print(f"要删除的{act_shearwall}不存在，删除该remove操作")
                response=response[:act_char_span[0]]+"*"*(act_char_span[1]-act_char_span[0])+response[act_char_span[1]:]
    #对标注后的response按原顺序提取有效信息
    result_actions,result_response=get_actions_and_index_info_index(response),""
    for action in result_actions:
        act_type,seg_type,coords=action["act_type"],action["seg_type"],action["coords"]
        result_response+=f"<{act_type}><{seg_type}>({coords[0]},{coords[1]}),({coords[2]},{coords[3]})"
    return result_response

if __name__=="__main__":
    AGG_STEP=1
    input_path=f"train_json_data/five_plus_two_train_jsonl_data/design_3.27/post_train_auged_sft_data/grpo_fpt_api_5_1000step/error_auged/q_13.jsonl"
    output_path=f"train_json_data/five_plus_two_train_jsonl_data/design_3.27/post_train_auged_sft_data/sft_data_5_1000step/error_auged/q_13.jsonl"
    with open(input_path, "r", encoding="utf-8") as fin:
        with open(output_path, "a", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                print(i)
                record = json.loads(line)
                house,design,floor,context,completion_predict,pre_post_text,pre_lineload_text,bound=\
                    record["house"],record["design"],record["floor"],record["context"],\
                    record["completion_predict"],record["pre_post_text"],record["pre_lineload_text"],\
                    record["bound"]
                response_list=record["improved_response_list"]
                round_completion_predict=completion_predict
                agg_response=""
                for index_response,response in enumerate(response_list):
                    agg_response+=response
                    if (index_response+1)%AGG_STEP==0:
                        print(f"过滤前response:{agg_response}")
                        agg_response_filt=filter_response(completion_predict=round_completion_predict,response=agg_response)
                        print(f"过滤后response:{agg_response_filt}")
                        prompt=construct_prompt(context=context,completion_predict=round_completion_predict,
                                pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text)
                        fout.write(json.dumps({"house":house,"floor":floor,"design":design,"bound":bound,"context":context,
                                        "prompt": prompt,"completion_predict":round_completion_predict,
                                        "pre_post_text":pre_post_text,
                                        "pre_lineload_text":pre_lineload_text,
                                        "response":agg_response_filt}, ensure_ascii=False) + "\n")
                        round_completion_predict=process_completion_predict_filt(context=context,
                                    completion_predict=round_completion_predict,response=agg_response_filt)
                        agg_response=""
    shuffle_jsonl_simple(output_path,output_path)