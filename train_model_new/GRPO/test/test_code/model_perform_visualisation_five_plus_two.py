import re
import json
import matplotlib.pyplot as plt
import sys
import os
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/five_plus_two_test")
sys.path.append("/home/jiuxing_li")
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/train_model_new/base_code")
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/train_model_new/GRPO/test/test_code")
#----------------------------------
#获取item,structure 的segments信息
#----------------------------------
from standard_function import get_segments_info_five_plus_two
from five_plus_two_visualisation_single_floor import plot_diaphragms,plot_segments,plot_segments_with_force_1
from reward_design_trunc import get_design_score,process_completion_predict_based_on_response
from model_test_new_token_five_plus_two import process_completion_predict_filt
from five_plus_two_optimization.train_model_new.base_code.reward_design_dcr_price_trunc import \
    visualize_force,transfer_force_text_to_dict
from unsupport_force_calculate import get_unsolved_force_item
POLYGON_TYPES = ["opening","inoutbox"]
LINE_TYPES = ["wall","shearwall","beam","exterior_wall"]
POINT_TYPES = []

def visualize_seven_axes(context,house_areas,beams_before,beams_valid_before,beams_after,
        beams_valid_after,shear_walls_before,shear_walls_valid_before,shear_walls_after,shear_walls_valid_after,
        diaphragms_before,diaphragms_after,output_path,bound,pre_post_data=None,pre_lineload_data=None,
        unsolved_lineload_list=None,unsolved_post_list=None,unsolved_lineload_list_pre=None,unsolved_post_list_pre=None):

    pre_post_list=[]
    for pre_post_key in pre_post_data.keys():
        pre_post_item=pre_post_data[pre_post_key]
        pre_post_list.append((pre_post_key[0],pre_post_key[1],pre_post_item.force))
    
    pre_lineload_list=[]
    for pre_lineload_key in pre_lineload_data.keys():
        pre_lineload_item=pre_lineload_data[pre_lineload_key]
        pre_lineload_list.append((pre_lineload_key[0],pre_lineload_key[1],pre_lineload_item.force))
    
    fig, axes = plt.subplots(2,3, figsize=(24, 16))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]
    initial_structures=get_segments_info_five_plus_two(context,POLYGON_TYPES,LINE_TYPES,POINT_TYPES)
    #print(answer_predict_beams+shear_walls_answer)
    #plot_segments(ax=ax1,segment_list=initial_structures,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,cut_line_list=None,title="Initial Structure",bound=bound)
    #plot_segments(ax=ax2,segment_list=initial_structures,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,cut_line_list=beams_before+shear_walls_before,title="Design Before",bound=bound)
    plot_segments(ax=ax2,segment_list=initial_structures,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,cut_line_list=beams_valid_before+shear_walls_valid_before,title="Design Before Valid",bound=bound)
    plot_segments_with_force_1(ax=ax2,segment_list=initial_structures,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,cut_line_list=beams_valid_before+shear_walls_valid_before,title="Design Before Valid",bound=bound,
    unsolved_lineload_list=unsolved_lineload_list_pre,unsolved_post_list=unsolved_post_list_pre)
    plot_diaphragms(ax=ax3,house_areas=house_areas,poly_list=diaphragms_before,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,title='Diaphragm Before Diaphragm',bound=bound)
    plot_segments_with_force_1(ax=ax4,segment_list=initial_structures,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,cut_line_list=beams_valid_after+shear_walls_valid_after,title="Design After VALID",bound=bound,
    unsolved_lineload_list=unsolved_lineload_list,unsolved_post_list=unsolved_post_list)
    plot_segments_with_force_1(ax=ax5,segment_list=initial_structures,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,cut_line_list=beams_valid_after+shear_walls_valid_after,title="Design After VALID",bound=bound,
            pre_post_list=pre_post_list,pre_lineload_list=pre_lineload_list)
    plot_diaphragms(ax=ax6,house_areas=house_areas,poly_list=diaphragms_after,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,title='Diaphragm After Diaphragm',bound=bound,code=1)

    plt.tight_layout()
    plt.show()
    plt.savefig(output_path)
    plt.close(fig)

def visualize(context,bound,completion_predict,answer,output_pic_dir,pre_post_text=None,pre_lineload_text=None,unsolved_force_result=None,unsolved_force_result_pre=None):
    #得到设计前的房屋整体设计分数
    result_before=get_design_score(context,completion_predict)
    house_score_before,area_ratio_before,beams_before,shear_walls_before,\
    beams_valid_before,shear_walls_valid_before,\
    valid_diaphragms_before,\
    =result_before["house_score"],result_before["area_ratio"],result_before["beams"],result_before["shear_walls"],\
    result_before["valid_beams"],result_before["shear_walls_valid"],\
    result_before["valid_diaphragms"]

    #得到设计后的房屋整体设计分数
    result_after=get_design_score(context,answer)
    house_score_after,area_ratio_after,beams_after,shear_walls_after,\
    beams_valid_after,shear_walls_valid_after,\
    valid_diaphragms_after,\
    =result_after["house_score"],result_after["area_ratio"],result_after["beams"],result_after["shear_walls"],\
    result_after["valid_beams"],result_after["shear_walls_valid"],\
    result_after["valid_diaphragms"]

    #print(f"unsolved_force_result:{unsolved_force_result},unsolved_force_result_pre:{unsolved_force_result_pre}")
    
    pre_lineload_dict,pre_post_dict=transfer_force_text_to_dict(pre_post_text=pre_post_text,
                                pre_lineload_text=pre_lineload_text)
    visualize_seven_axes(context=context,house_areas=result_before["house_areas"],beams_before=beams_before,beams_after=beams_after,
                    shear_walls_before=shear_walls_before,shear_walls_after=shear_walls_after,
                    beams_valid_before=beams_valid_before,beams_valid_after=beams_valid_after,
                shear_walls_valid_before=shear_walls_valid_before,shear_walls_valid_after=shear_walls_valid_after,
            diaphragms_before=[list(poly.exterior.coords) for poly in valid_diaphragms_before],
            diaphragms_after=[list(poly.exterior.coords) for poly in valid_diaphragms_after],output_path=output_pic_dir,bound=bound,
            pre_post_data=pre_post_dict,pre_lineload_data=pre_lineload_dict,
            unsolved_lineload_list=unsolved_force_result["unsolved_lineload_list"],
            unsolved_post_list=unsolved_force_result["unsolved_post_list"],
            unsolved_lineload_list_pre=unsolved_force_result_pre["unsolved_lineload_list"],
            unsolved_post_list_pre=unsolved_force_result_pre["unsolved_post_list"],)

if __name__=="__main__":
    for index in range(1):
        SUPER_INDEX=34
        load_path=f"five_plus_two_optimization/train_model_new/GRPO/test/test_jsonl/grpo_fpt_api_4/error_auged_filt/4000_step_error_super_{SUPER_INDEX}_auged_filt.jsonl"
        with open(load_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            total_lines = len(lines)
            for path_index, line in enumerate(lines):
                if path_index>100:
                    break
                #if path_index!=2:
                #    continue
                print(f"Visualizing {path_index}")
                record = json.loads(line)
        
                bound=record["bound"]
                house,floor,context,completion_predict,prompt,answer,response_list,improved_result_list = record["house"],record["floor"],record["context"],\
                record["completion_predict"],record['prompt'],record["answer"],record["improved_response_list"],record["improved_result_list"]
                pre_post_text,pre_lineload_text=record["pre_post_text"],record["pre_lineload_text"]
                #improve_step,improved_value=[0]+record['improve_step'],[]
                print(path_index+1,house,floor)
                if pre_post_text!="" or pre_lineload_text!="":
                    unsolved_force_result_pre=get_unsolved_force_item(context=context,answer=completion_predict,
                        pre_lineload_text=pre_lineload_text,pre_post_text=pre_post_text)
                    unsolved_force_num_pre=unsolved_force_result_pre["unsolved_lineload_num"]+unsolved_force_result_pre["unsolved_post_num"]
                else:
                    print(f"无上层受力")
                    unsolved_force_result_pre=None
                    unsolved_force_num_pre=0
                    continue
                #print(f"context:{context}")
                #print(f"completion_predict:{completion_predict}")
                #print(f"response:{response}")
                round_completion_predict=completion_predict
                output_pic_dir="five_plus_two_optimization/train_model_new/GRPO/test/test_pic/grpo_fpt_api_4/super_auged_4000step"
                if len(response_list)!=0:
                    for response_index, response in enumerate(response_list):
                        print(f"Response {response_index}")
                        print(f"context:{context}")
                        print(f"completion_predict:{round_completion_predict}")
                        print(f"response:{response}")
                        answer=process_completion_predict_filt(context=context,completion_predict=round_completion_predict,response=response)
                        print(f"answer:{answer}")
                        improved_result=improved_result_list[response_index]
                        output_pic_path=f'{output_pic_dir}/sample_{SUPER_INDEX}_{house}_{floor}_{response_index+1}_{SUPER_INDEX}_{round(improved_result["error_num_before"],2)}_{round(improved_result["error_num_after"],2)}.png'
                        
                        unsolved_force_result=get_unsolved_force_item(context=context,answer=answer,
                                        pre_lineload_text=pre_lineload_text,pre_post_text=pre_post_text)
                        unsolved_force_result_pre=get_unsolved_force_item(context=context,answer=round_completion_predict,
                        pre_lineload_text=pre_lineload_text,pre_post_text=pre_post_text)

                        visualize(context=context,completion_predict=round_completion_predict,answer=answer,output_pic_dir=output_pic_path,
                                  pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text,bound=bound,
                                  unsolved_force_result=unsolved_force_result,unsolved_force_result_pre=unsolved_force_result_pre)
                        round_completion_predict=answer
                else:
                    output_pic_path=f'{output_pic_dir}/sample_{SUPER_INDEX}_{house}_{floor}_{unsolved_force_num_pre}.png'
                    visualize(context=context,completion_predict=round_completion_predict,answer=answer,output_pic_dir=output_pic_path,
                    pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text,bound=bound,
                    unsolved_force_result=unsolved_force_result_pre,unsolved_force_result_pre=unsolved_force_result_pre)