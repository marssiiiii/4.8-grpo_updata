import json
from unsupport_force_calculate import get_unsolved_force_item

if __name__=="__main__":
    index_list=[1]
    #index_list=[1,2,3,4,5,6,7,8]
    opt_num_list,total_num_list=[],[]
    for index in index_list:
        load_path=f"five_plus_two_optimization/train_model_new/train_base_model/test/test_jsonl/post_train_2/area_q_13.jsonl"
        #load_path=f"five_plus_two_optimization/train_model_new/GRPO/test/test_jsonl/ernie_base_model_10_error_auged_filt.jsonl"
        with open(load_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            opt_num,total_num=0,0
            for path_index, line in enumerate(lines):
                if path_index<50:
                    continue
                record = json.loads(line)
                
                '''
                #error测试过滤条件
                context,completion_predict,pre_post_text,pre_lineload_text=record["context"],record["completion_predict"],\
                record["pre_post_text"],record["pre_lineload_text"]
                if pre_post_text=="" or pre_lineload_text=="":
                    continue
                unsolved_force_result=get_unsolved_force_item(context=context,answer=completion_predict,
                                pre_lineload_text=pre_lineload_text,pre_post_text=pre_post_text)
                if unsolved_force_result["unsolved_lineload_num"]+unsolved_force_result["unsolved_post_num"]==0:
                    print("unsolved_force_items为0")
                    continue
                '''
                
                bound=record["bound"]
                house,floor,improved_result_list = record["house"],record["floor"],record["improved_result_list"]
                print(path_index+1,house,floor)
                if len(improved_result_list)>0:
                    opt_num+=1
                total_num+=1
            opt_num_list.append(opt_num)
            total_num_list.append(total_num)
    
    print(f"优化数量：{opt_num_list}/{total_num_list}，占比{[opt_num/total_num for opt_num,total_num in zip(opt_num_list,total_num_list)]}")