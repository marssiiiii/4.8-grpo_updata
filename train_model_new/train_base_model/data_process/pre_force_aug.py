import json
import re
import random
import sys
sys.path.append("/home/jiuxing_li")
from shapely import LineString,Point
from five_plus_two_optimization.train_model_new.base_code.reward_design_trunc import \
    process_completion_predict_filt,get_segments_info_five_plus_two
from five_plus_two_optimization.train_model_new.base_code.reward_design_dcr_price_trunc import \
    get_initial_struct_based_on_response,transfer_force_text_to_dict
from five_plus_two_optimization.train_model_new.train_base_model.cmp_actionized_and_tokenized_api import \
split_cmp_to_answer,from_cmp_to_action_context,filter_completion_predict,random_action_context,visualize_force,construct_prompt

def order_of_magnitude(N):
    mag = 0
    n=1
    while n * 10 <= N:
        mag += 1
        n*=10
    return mag

def get_force_items_from_unprocess_force_text(prompt):
    #print(pre_lineload_text,pre_post_text)
    pattern_lineload = r"<LINELOAD>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\s*(-?\d+)\s*"
    pattern_post = r"<POST>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\s*(-?\d+)\s*"
    pre_lineload_list = [((int(m[0]), int(m[1])), (int(m[2]), int(m[3])), int(m[4]))
                         for m in re.findall(pattern_lineload, prompt)]
    pre_post_list = [((int(m[0]), int(m[1])), int(m[2]))
                     for m in re.findall(pattern_post, prompt)]
    return pre_lineload_list,pre_post_list

def get_load_structure_from_text(text,pre_lineload_list,pre_post_list):
    structs=get_segments_info_five_plus_two(text,[],["beam","shearwall"],[])
    structs_load_force=[]
    for struct in structs:
        STRUCT_LOAD_FORCE=False
        struct_line=LineString([struct[1],struct[2]])
        for pre_lineload in pre_lineload_list:
            pre_lineload_line=LineString([pre_lineload[0],pre_lineload[1]])
            if get_initial_struct_based_on_response.line_overlap_line(line_1=struct_line,line_2=pre_lineload_line):
                structs_load_force.append(struct)
                STRUCT_LOAD_FORCE=True
                break
        if STRUCT_LOAD_FORCE==False:
            for pre_post in pre_post_list:
                pre_post_point=Point(pre_post[0])
                if get_initial_struct_based_on_response.point_in_line(pt=pre_post_point,line=struct_line):
                    structs_load_force.append(struct)
                    STRUCT_LOAD_FORCE=True
                    break
    return structs_load_force

def sort_completion_predict(completion_predict): #保证completion_predict的排列方式始终是先beam后shearwall
    text=""
    beams=get_segments_info_five_plus_two(completion_predict,[],["beam"],[])
    shearwalls=get_segments_info_five_plus_two(completion_predict,[],["shearwall"],[])
    for beam in beams:
        text+=f"<{beam[0]}>({int(beam[1][0])},{int(beam[1][1])}),({int(beam[2][0])},{int(beam[2][1])})"
    for shearwall in shearwalls:
        text+=f"<{shearwall[0]}>({int(shearwall[1][0])},{int(shearwall[1][1])}),({int(shearwall[2][0])},{int(shearwall[2][1])})"
    return text

def swap_load_force_items(answer_pre,answer_after,pre_lineload_list,pre_post_list):
    structs_pre=get_segments_info_five_plus_two(answer_pre,[],["beam","shearwall"],[])
    structs_after=get_segments_info_five_plus_two(answer_after,[],["beam","shearwall"],[])
    structs_load_force_pre=get_load_structure_from_text(text=answer_pre,
                                        pre_lineload_list=pre_lineload_list,pre_post_list=pre_post_list)
    structs_load_force_after=get_load_structure_from_text(text=answer_after,
                                        pre_lineload_list=pre_lineload_list,pre_post_list=pre_post_list)
    #将answer_pre中的承力元素等量加到answer_after中并打乱
    structs_after=structs_after+structs_load_force_pre
    for struct in structs_load_force_pre:
        if struct in structs_pre:
            structs_pre.remove(struct)
        else:
            print("对应错误！")
    #将answer_after中的不属于承力元素的结构加回到structs
    add_num,total_num=0,len(structs_load_force_pre)
    for struct in structs_after:
        if (struct in structs_load_force_pre) or (struct in structs_load_force_after):
            continue
        structs_pre.append(struct)
        add_num+=1
        if add_num>=total_num:
            break

    answer_pre=""
    for struct in structs_pre:
        answer_pre=answer_pre+\
            f"<{struct[0]}>({int(struct[1][0])},{int(struct[1][1])}),({int(struct[2][0])},{int(struct[2][1])})"
    answer_pre=sort_completion_predict(answer_pre)
    answer_after=""
    for struct in structs_after:
        answer_after=answer_after+\
            f"<{struct[0]}>({int(struct[1][0])},{int(struct[1][1])}),({int(struct[2][0])},{int(struct[2][1])})"
    return answer_pre,answer_after

def process_force_info_to_text(pre_lineload_list,pre_post_list):
    def point_in_line(pt:Point,line:LineString,BETWEEN_FLOOR_EPS=30):
        return pt.distance(line) < BETWEEN_FLOOR_EPS
    
    lineload_line_list=[]
    pre_lineload_text=""
    for pre_lineload in pre_lineload_list:
        if pre_lineload[2]>=1:
            pre_lineload_text+=f"<LINELOAD>({pre_lineload[0][0]},{pre_lineload[0][1]}),({pre_lineload[1][0]},{pre_lineload[1][1]}),WEIGHT_{order_of_magnitude(int(pre_lineload[2]))}"
            lineload_line_list.append(LineString([pre_lineload[0],pre_lineload[1]]))
    
    pre_post_text=""
    for pre_post in pre_post_list:
        if pre_post[1]>=1:
            POST_IN_LINELOAD=False
            post_point=Point(pre_post[0])
            for lineload_line in lineload_line_list:
                if point_in_line(pt=post_point,line=lineload_line)==True:
                    POST_IN_LINELOAD=True
                    #print("POST_IN_LINELOAD")
                    break
            if POST_IN_LINELOAD==False:
                pre_post_text+=f"<POST>({pre_post[0][0]},{pre_post[0][1]}),WEIGHT_{order_of_magnitude(int(pre_post[1]))}"
    return pre_lineload_text,pre_post_text

def extract_completion_predict_from_prompt(prompt):
    struct_char = prompt.find("structures:")
    end=prompt.find("</s>")
    completion_predict=prompt[struct_char+11:end]
    return completion_predict

if __name__=="__main__":
    INPUT_JSON="ware_house/test_set_100_actionized_sort_floor_force.jsonl"
    OUTPUT_PATH="train_json_data/five_plus_two_train_jsonl_data/design_3.27/base_model_train/test_set_100_actionized_sort_floor_force_auged(3_times)_transfer.jsonl"
    pre_force_num,total_num=0,0
    NEED_VISUALIZE=False
    with open(INPUT_JSON, "r", encoding="utf-8") as fin:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                record = json.loads(line)
                house,design,floor,bound=record["house"],record["design"],record["floor"],record["bound"]
                context,prompt,completion_predict,response = \
                    record["context"],record["prompt"],record["completion_predict"],record["response"]
                print(i,house,design,floor)
                pre_lineload_list,pre_post_list=get_force_items_from_unprocess_force_text(prompt)
                house_areas=get_segments_info_five_plus_two(context,["inoutbox"],[],[])
                HAS_PRE_FORCE=False
                
                if pre_post_list!=[] or pre_lineload_list!=[]:
                    HAS_PRE_FORCE=True
                    pre_lineload_text,pre_post_text=process_force_info_to_text(pre_lineload_list=pre_lineload_list,
                                                            pre_post_list=pre_post_list)
                    #print(f"pre_lineload_text:{pre_lineload_text},pre_post_text:{pre_post_text}")
                    pre_force_num+=1
                    answer_1,answer_2=split_cmp_to_answer(completion_predict)
                    answer_1,answer_2=swap_load_force_items(answer_pre=answer_1,answer_after=answer_2,
                                        pre_lineload_list=pre_lineload_list,pre_post_list=pre_post_list)
                    
                    completion_predict_filtered=filter_completion_predict(context=context,completion_predict=completion_predict)
                    response,ADD_BEAM,ADD_SHEAR,REMOVE_BEAM,REMOVE_SHEAR=from_cmp_to_action_context(context=context,answer_pre=answer_1,
                    answer_after=answer_2,completion_predict=completion_predict_filtered)
                    response=random_action_context(response)
                    #print(f"completion_predict:{answer_1},response:{response}")

                    prompt=construct_prompt(pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text,context=context
                                            ,completion_predict=answer_1)
                    response=f"{response}</s></s>"
                    #print(f"answer_1:{answer_1}")
                    #print(f"response:{response}")
                    #可视化
                    if NEED_VISUALIZE==True:
                        answer=process_completion_predict_filt(context=context,completion_predict=answer_1,response=response)
                        pre_lineload_dict,pre_post_dict=transfer_force_text_to_dict(pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text)
                        #print(f"pre_lineload_dict:{pre_lineload_dict},pre_post_dict:{pre_post_dict}")
                        visualize_force(context=context,answer=answer,pre_context=context,pre_answer=answer_1,
                                        pre_post_data=pre_post_dict,pre_lineload_data=pre_lineload_dict,bound=record["bound"],
                        output_path=f"five_plus_two_optimization/train_model_new/train_base_model/test/test_pic/aug_check/{i+1}_{house}_{design}_{floor}.png",
                        house_areas=house_areas)
                else:
                    pre_post_text,pre_lineload_text="",""
                    answer_1=extract_completion_predict_from_prompt(prompt)
                    prompt=construct_prompt(pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text,context=context,
                                    completion_predict=answer_1)
                    #print(answer_1)

                total_num+=1
                wrapped = {
                        "house":house,
                        "design":design,
                        "floor":floor,
                        "bound":bound,
                        "context":context,
                        "completion_predict_pre":completion_predict,
                        "completion_predict":answer_1,
                        "prompt":prompt,
                        "response":response,
                        "pre_post_text":pre_post_text,
                        "pre_lineload_text":pre_lineload_text,
                    }
                fout.write(json.dumps(wrapped, ensure_ascii=False) + "\n")
                if HAS_PRE_FORCE==True:
                    fout.write(json.dumps(wrapped, ensure_ascii=False) + "\n")
                    fout.write(json.dumps(wrapped, ensure_ascii=False) + "\n")
                    #fout.write(json.dumps(wrapped, ensure_ascii=False) + "\n")
                    pre_force_num+=2
                    total_num+=2
                #if i>10:
                #    break
    
    with open(OUTPUT_PATH) as f:
      lines = f.readlines()
    random.shuffle(lines)
    with open(OUTPUT_PATH, "w") as f:
        f.writelines(lines)
    
    print(f"no_pre_force_num:{pre_force_num}/{total_num},占比{pre_force_num/total_num}")