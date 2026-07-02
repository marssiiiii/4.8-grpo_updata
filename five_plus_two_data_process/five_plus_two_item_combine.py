import sys
import json
import random
sys.path.append("/mnt/efs/jiuxing_li")
sys.path.append("five_plus_two_optimization/train_model_new/base_code")
sys.path.append("five_plus_two_optimization/train_model_new/train_base_model")
from standard_function import get_segments_info_five_plus_two,shuffle_jsonl_simple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from ernie_base_model_train import initialize_token_embedding
from base_model_test import filter_completion_predict #filter_completion_predict(completion_predict,context)
from reward_design import get_design_score #get_design_score(context,answer)
POLYGON_TYPES = ["opening","joist_area","truss_area"]
LINE_TYPES = ["wall","beam","shearwall"]
POINT_TYPES = []
GENERATE_DATA_NUM=10

input_path="train_json_data/five_plus_two_train_jsonl_data/design_3.10/initial_data_test_set.jsonl"
output_path="train_json_data/five_plus_two_train_jsonl_data/design_3.10/initial_data_test_set_combined(0.5rd+0.1rd)_less_than_7k.jsonl"
MODEL_ID = "baidu/ERNIE-4.5-0.3B-PT" #训练模型
MAX_TOKEN_LENGTH=7000
#倒入tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID) #自动加载Gemma3对应tokenizer
if tokenizer.pad_token is None: #如果tokenizer中没有pad_token,就用eos_token填充
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto", #自动把模型分配到gpu
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, #使用半精度节省显存，如gpu不支持则用32位浮点
    use_cache=False,  # 禁用KV缓存，与梯度检查点冲突
)
initialize_token_embedding(model,tokenizer)

def from_structure_list_to_text(structure_list):
    text=""
    for seg in structure_list:
        text+=f"<{seg[0]}>({int(seg[1][0])},{int(seg[1][1])}),({int(seg[2][0])},{int(seg[2][1])})"
    return text

def construct_prompt(context,completion_predict):#旧版construct_prompt
    prompt="You are an architectural structural assistant.\n"
    prompt=prompt+"Available operations (STRICT FORMAT):\n"
    prompt=prompt+"<add><beam>(x1,y1),(x2,y2)</beam>\n" 
    prompt=prompt+"<remove><beam>(x1,y1),(x2,y2)</beam>\n" 
    prompt=prompt+"<add><shearwall>(x1,y1),(x2,y2)</shearwall> (the shearwall must in wall)\n"
    prompt=prompt+"<remove><shearwall>(x1,y1),(x2,y2)</shearwall>\n"
    prompt=prompt+"Input format:\n"
    prompt=prompt+"<type>(x1,y1),(x2,y2) where type ∈ {wall,exterior_wall, beam,shearwall, opening, truss_area, joist_area}\n"
    prompt=prompt+"Rules:\n"
    prompt=prompt+"1. ONLY output beam/shearwall modifications using the exact formats above\n"
    prompt=prompt+"2. Do NOT modify walls, exterior_walls, openings, or boundary lines.\n"
    prompt=prompt+"3. Beams: must be supported by walls/beams at both ends.\n"
    prompt=prompt+"4. Shearwalls: can ONLY be added by converting existing walls (coordinates must match an existing wall line)\n"
    prompt=prompt+"5. truss_area/joist_area: the boundary of the polygon\n"
    prompt=prompt+"Output format (STRICT):\n"
    prompt=prompt+"-Only output modified beams using:\n"
    prompt=prompt+"<add><beam>(x1,y1),(x2,y2)</beam>\n" 
    prompt=prompt+"<remove><beam>(x1,y1),(x2,y2)</beam>\n" 
    prompt=prompt+"<add><shearwall>(x1,y1),(x2,y2)</shearwall> (the shearwall must in wall)\n"
    prompt=prompt+"<remove><shearwall>(x1,y1),(x2,y2)</shearwall>\n"
    prompt=prompt+"NO explanations.\n"
    prompt=prompt+"NO input repetition.\n"
    prompt=prompt+"NO additional text.\n"        
        
    prompt=prompt+f"The initial structure:{context}\n"
    prompt=prompt+f"shearwalls and beams that have been added:{completion_predict}"
    
    prompt=f"<s>{prompt}</s>"
    return prompt

def combine_item_and_result(input_path,output_path):
    with open(input_path, "r", encoding="utf-8") as fin:
        for i,line in enumerate(fin):
            print(f"第{i}条原数据")
            record = json.loads(line)
            context,completion_predict,house,floor,bound = record["context"],record["completion_predict"],\
                record['house'],record['floor'],record['bound']
            completion_predict_filtered=filter_completion_predict(completion_predict=completion_predict,context=context)

            #得到原始以及过滤到的completion_predict内的预测结果，并将其随机打乱
            structure_list = get_segments_info_five_plus_two(completion_predict,POLYGON_TYPES=POLYGON_TYPES,
                                            LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES)
            structure_list_filtered=get_segments_info_five_plus_two(completion_predict_filtered,POLYGON_TYPES=POLYGON_TYPES,
                                            LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES)
            
            for _ in range(GENERATE_DATA_NUM):
                #将structure数据随机打乱
                structure_shuffled = random.sample(structure_list, len(structure_list))
                structure_filtered_shuffled = random.sample(structure_list_filtered, len(structure_list_filtered))
                #构造completion_predict的噪声部分
                k1 = int(0.1*random.random() * len(structure_filtered_shuffled)) #k1代表引入噪声的数量
                structure_new_list = structure_shuffled[:k1]
                structure_combined_context = from_structure_list_to_text(structure_new_list)
                #print(f"completion_predict 噪声部分：{structure_combined_context}")

                #构造completion_predict的有效线段部分
                
                k2 = int((0.5+random.random()/2) * len(structure_filtered_shuffled)) #k2代表保留valid structure的数量
                structure_filtered_new_list=structure_filtered_shuffled[:k2]
                structure_filtered_text = from_structure_list_to_text(structure_filtered_new_list)
                
                #structure_filtered_text=completion_predict_filtered
                result=get_design_score(context,structure_filtered_text)
                valid_beams,valid_shearwalls=result['valid_beams'],result['shear_walls_valid']
                valid_segs=valid_shearwalls+valid_beams
                valid_filtered_text=from_structure_list_to_text(valid_segs)
                
                
                #print(f"completion_predict 有效线段部分：{valid_filtered_text}")
                #组装completion_predict,并依据其构造prompt
                combined_completion_predict=valid_filtered_text+structure_combined_context
                combined_completion_predict=filter_completion_predict(completion_predict=combined_completion_predict,context=context)
                #print(f"组装后的completion_predict:{combined_completion_predict}")
                prompt=construct_prompt(context=context,completion_predict=combined_completion_predict)

                prompt_token_length=len(tokenizer(prompt).input_ids)
                if prompt_token_length>MAX_TOKEN_LENGTH:
                    print(f"{prompt_token_length}超出限制！")
                    continue
                
                with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"house":house,"floor":floor,"bound":bound,
                                            "context": context,"prompt":prompt,
                                            "completion_predict":combined_completion_predict}
                                            , ensure_ascii=False) + "\n")
    shuffle_jsonl_simple(output_path,output_path)
            #break

if __name__=="__main__":
    combine_item_and_result(input_path=input_path,output_path=output_path)