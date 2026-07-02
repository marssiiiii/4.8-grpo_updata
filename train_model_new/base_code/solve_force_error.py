import json
from shapely import LineString,Point
import random
import re
import sys
sys.path.append("/home/jiuxing_li/Genia-structural-copilot-saas-server")
sys.path.append("/home/jiuxing_li")
from gen_pipeline_v2.src.utils.resolve_support import resolve_segments_with_supports
from five_plus_two_optimization.train_model_new.GRPO.test.test_code.unsupport_force_calculate import \
get_unsolved_force_item
from five_plus_two_optimization.train_model_new.base_code.reward_design_trunc import \
process_completion_predict_filt,get_segments_info_five_plus_two,process_completion_predict_based_on_response
from five_plus_two_optimization.train_model_new.GRPO.test.test_code.model_perform_visualisation_five_plus_two import \
visualize

def divide_prompt_into_parts(prompt):
    pre_lineload_char = prompt.find("upper layer lineload:")
    pre_post_char = prompt.find("upper layer post:")
    ctx_char = prompt.find("context:")
    cmp_char = prompt.find("structures:")
    eos_char = prompt.find("</s>")

    context=prompt[ctx_char:cmp_char]
    completion_predict=prompt[cmp_char:eos_char]
    pre_lineload_text=prompt[pre_lineload_char:pre_post_char]
    pre_post_text=prompt[pre_post_char:ctx_char]
    return context,completion_predict,pre_lineload_text,pre_post_text

def visualize_gen_sample_process(prompt_list,response_list,response_after_list,num_pre_Q):
    OUTPUT_DIR="/home/jiuxing_li/five_plus_two_optimization/five_plus_two_test/test_pic/grpo_error_aug_check"
    for idx in range(len(prompt_list)):
        prompt=prompt_list[idx]
        response=response_list[idx]
        response_after=response_after_list[idx]
        #print(f"response_pre:{response}")
        #print(f"response_after:{response_after}")
        context,completion_predict,pre_lineload_text,pre_post_text=divide_prompt_into_parts(prompt)
        answer=process_completion_predict_filt(context=context,completion_predict=completion_predict,response=response)
        answer_after=process_completion_predict_filt(context=context,completion_predict=completion_predict,response=response_after)
        #print(f"answer_pre:{answer}")
        #print(f"answer_after:{answer_after}")
        
        unsolved_force_result=get_unsolved_force_item(context=context,answer=answer,
        pre_lineload_text=pre_lineload_text,pre_post_text=pre_post_text)
        unsolved_force_result_after=get_unsolved_force_item(context=context,answer=answer_after,
        pre_lineload_text=pre_lineload_text,pre_post_text=pre_post_text)

        if unsolved_force_result["unsolved_lineload_num"]+unsolved_force_result["unsolved_post_num"]>0:
            output_path=f"{OUTPUT_DIR}/{int(idx/num_pre_Q)}_{idx+1}.png"
            visualize(context=context,completion_predict=answer,answer=answer_after,output_pic_dir=output_path,
                        pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text,
                        unsolved_force_result=unsolved_force_result_after,unsolved_force_result_pre=unsolved_force_result,
                        bound=None)

def get_resolve_support_result(pre_post_text,pre_lineload_text,context,answer):
    #得到未解决的线和点
    unsolved_force_result=get_unsolved_force_item(context=context,answer=answer,pre_lineload_text=pre_lineload_text
        ,pre_post_text=pre_post_text)
    unsolved_lineload_list,unsolved_post_list=unsolved_force_result["unsolved_lineload_list"],unsolved_force_result["unsolved_post_list"]
    unsolved_line_list,unsolved_point_list=[],[]
    for lineload in unsolved_lineload_list:
        lineload_line=LineString([lineload[0],lineload[1]])
        unsolved_line_list.append(lineload_line)
    for post in unsolved_post_list:
        post_point=Point((post[0],post[1]))
        unsolved_point_list.append(post_point)
    #得到可以承重的线
    support_lines=[]
    walls=get_segments_info_five_plus_two(context,[],["wall","exterior_wall"],[])
    beams=get_segments_info_five_plus_two(answer,[],["beam"],[])
    for wall in walls:
        support_lines.append(LineString([wall[1],wall[2]]))
    for beam in beams:
        support_lines.append(LineString([beam[1],beam[2]]))
    #调用接口解决上层受力
    resolve_result=resolve_segments_with_supports(segments=unsolved_line_list,initial_supports=support_lines,
    overlap_tol=10,buffer_distance=10,max_extension_distance=1e5,point_loads=unsolved_point_list,point_supported_radius=10)
    return unsolved_lineload_list,unsolved_post_list,unsolved_force_result,resolve_result

def get_resolve_support_response(add_segments,select_beam_rd):
    response=""
    #print(f"add_segments:{add_segments}")
    for idx,seg in add_segments:
        if random.random()<select_beam_rd:
            pt_1,pt_2=seg.coords[0],seg.coords[-1]
            response=response+f"<add><beam>({int(pt_1[0])},{int(pt_1[1])}),({int(pt_2[0])},{int(pt_2[1])})"
    return response

def get_resolve_support_response_from_prompt(prompt,response="",select_beam_rd=0.8):
    context,completion_predict,pre_lineload_text,pre_post_text=divide_prompt_into_parts(prompt)
    answer=process_completion_predict_filt(context=context,completion_predict=completion_predict,response=response)
    _,_,_,resolve_result=get_resolve_support_result(pre_post_text=pre_post_text,
                                                    pre_lineload_text=pre_lineload_text,context=context,answer=answer)
    response=get_resolve_support_response(add_segments=resolve_result.accepted_segments,select_beam_rd=select_beam_rd)
    return response

def combine_resp(resp_pre,added_resp):
    pattern=r"<(add|remove)><(beam|shearwall)>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
    list_pre=[m.groups() for m in re.finditer(pattern, resp_pre)]
    list_added=[m.groups() for m in re.finditer(pattern,added_resp)]

    # 将 list_added 的每个元素随机插入 list_pre
    combined = list_pre[:]
    for item in list_added:
        pos = random.randint(0, len(combined))
        combined.insert(pos, item)
    # 重建 response 字符串
    result = ""
    for act, seg, x1, y1, x2, y2 in combined:
        result += f"<{act}><{seg}>({x1},{y1}),({x2},{y2})"
    return result

if __name__=="__main__":
    resp_pre="<add><beam>(2899,2751),(2899,2659)<add><shearwall>(2899,2061),(3088,2061)<add><beam>(2299,2901),(2299,2793)<add><shearwall>(3190,3201),(3205,3201)<add><beam>(2299,2751),(2899,2751)<add><beam>(2629,2907),(2809,2907)<add><beam>(2899,2997),(2899,3029)<add><shearwall>(3424,2751),(3424,2767)<add><shearwall>(3025,3201),(310"
    added_resp="<add><beam>(2899,2557),(3349,2557)"
    combine_resp(resp_pre,added_resp)
    '''
    DATA_PATH="train_json_data/five_plus_two_train_jsonl_data/design_3.27/grpo_phase_1_train_data/train_set_inbox_force_prompt.jsonl"
    OUTPUT_DIR="five_plus_two_optimization/five_plus_two_test/test_pic/api_solve_check"
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            #if i!=40:
            #    continue
            #输入初始数据
            record = json.loads(line) #依据response生成house_struct
            house,floor,design,bound,context,completion_predict,pre_post_text,pre_lineload_text,bound,prompt = record["house"],record["floor"],\
            record["design"],record['bound'],record["context"],record["completion_predict"],\
            record["pre_post_text"],record["pre_lineload_text"],record["bound"],record["prompt"]
            answer=process_completion_predict_filt(context=context,completion_predict=completion_predict,response="")
            #print(f"answer:{answer}")
            house_design_code=str(house)+str(design)
            house_areas=get_segments_info_five_plus_two(context,["inoutbox"],[],[])
            print(f"{i},{house_design_code},{floor}")
            #得到解决结果
            unsolved_lineload_list,unsolved_post_list,unsolved_force_result,resolve_result=\
            get_resolve_support_result(pre_post_text=pre_post_text,
            pre_lineload_text=pre_lineload_text,context=context,answer=answer)

            if unsolved_lineload_list==[] and unsolved_post_list==[]:
                continue

            print(f"resolve_result:")
            print(f"accepted_segments:{resolve_result.accepted_segments}")
            print(f"remaining_segments:{resolve_result.remaining_segments}")
            print(f"unsupported_points:{resolve_result.unsupported_points}")
            print(f"selection_pending:{resolve_result.selection_pending}") #需要解的lineload
            print(f"selection_accept_count:{resolve_result.selection_accept_count}")

            #response=get_resolve_support_response(add_segments=resolve_result.accepted_segments)
            response=get_resolve_support_response_from_prompt(prompt=prompt)
            print(f"response:{response}")
            answer_resolved=process_completion_predict_filt(context=context,completion_predict=completion_predict,response=response)
            #print(f"answer_resolved:{answer_resolved}")
            unsolved_force_result_after=get_unsolved_force_item(context=context,answer=answer_resolved,
            pre_lineload_text=pre_lineload_text,pre_post_text=pre_post_text)
            
            output_path=f"{OUTPUT_DIR}/{i+1}_{house_design_code}_{floor}.png"
            visualize(context=context,completion_predict=answer,answer=answer_resolved,output_pic_dir=output_path,
                        pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text,
                        unsolved_force_result=unsolved_force_result_after,unsolved_force_result_pre=unsolved_force_result,
                        bound=bound)
            if i>50:
                break
    '''