import re
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("/home/jiuxing_li/code/text_generate_from_given_structure")
from structure_identify_function import get_rect_from_design
from standard_function import get_segments_info

def get_avg_room_num_error_from_path(load_path):
    num_error_list_train,num_error_list_test=[],[]
    with open(load_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            text=""
            #读取数据
            record = json.loads(line)
            prompt = record["prompt"]
            completion_predict,completion_ground = record['completion_predict'],record['completion_ground']
            house,floor=record['house'],record['floor']
            #wall_segments = get_segments_info(prompt)
            wall_segments=get_segments_info(prompt)
            cut_line_segments_predict,cut_line_segments_ground =\
                    get_segments_info(completion_ground),get_segments_info(completion_predict)
            segments_predict,segments_ground=wall_segments+cut_line_segments_predict,\
                wall_segments+cut_line_segments_ground
            rect_list_pred,rect_list_gt=get_rect_from_design(segments_predict),\
                get_rect_from_design(segments_ground)
            if i<50:
                num_error_list_train.append(abs(len(rect_list_pred)-(len(rect_list_gt))))
            else:
                num_error_list_test.append(abs(len(rect_list_pred)-(len(rect_list_gt))))
    return np.mean(num_error_list_train),np.mean(num_error_list_test)

if __name__=="__main__":
    error_train_list,error_test_list=[],[]
    for i in range(1):
        path=f"/mnt/disks/workspace/jiuxing_li/code/GRPO/test/test_jsonl/grpo_10_1_1k.jsonl"
        print(f"path:{path}")
        error_train,error_test=get_avg_room_num_error_from_path(path)
        error_train_list.append(error_train)
        error_test_list.append(error_test)
        print(f"训练集误差:{error_train},测试集误差:{error_test}")
    '''
    code=np.array(code)
    #output_dir="ERNIE_train_results/split_line_predict/generate_no_phase1_1"
    plt.figure()
    bar_width = 0.035
    plt.bar(code - bar_width/2, error_train_list, width=bar_width, label='Train')
    plt.bar(code + bar_width/2, error_test_list,  width=bar_width, label='Test')
    plt.ylabel('avg_num_error')
    plt.legend()
    plt.grid(axis='y')

    plt.show()
    plt.savefig(f"ERNIE_train_results/split_line_predict/generate_phase1_lora_9(10+2)/room_num_error.png")
    '''