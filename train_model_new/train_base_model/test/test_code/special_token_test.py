import sys
import torch
sys.path.append("five_plus_two_optimization/train_model_new/train_base_model")
from ernie_base_model_train import initialize_token_embedding
from transformers import AutoTokenizer,AutoModelForCausalLM
MODEL_ID = "baidu/ERNIE-4.5-0.3B-PT"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID) #初始化tokenizer
if tokenizer.pad_token is None: #如果tokenizer中没有pad_token,就用eos_token填充
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto", #自动把模型分配到gpu
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, #使用半精度节省显存，如gpu不支持则用32位浮点
    use_cache=False,  # 禁用KV缓存，与梯度检查点冲突
)
initialize_token_embedding(model,tokenizer)

text_1="<POST>(2796,2405),87,<LINELOAD>(2635,3133),(2676,3133),2189"
text_2="<post>(2796,2405),87,<lineload>(2635,3133),(2676,3133),2189"
len_1,len_2 = len(tokenizer(text_1).input_ids),len(tokenizer(text_2).input_ids)
print(len_1,len_2)