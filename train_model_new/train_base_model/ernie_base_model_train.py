import json
from peft import PeftModel,LoraConfig,get_peft_model
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForCausalLM,TrainerCallback
import matplotlib.pyplot as plt
import gc
import numpy as np
import os

# ------------------------
# 配置
# ------------------------
# Set HF_TOKEN in the environment before running this script.
OUTPUT_DIR = "five_plus_two_optimization/train_model_new/train_base_model/TRAIN_RESULTS/grpo_post_train_2/area_q_13"
TRAIN_JSONL = "train_json_data/five_plus_two_train_jsonl_data/design_3.27/post_train_auged_sft_data/sft_data_5_1000step/area_auged/q_13.jsonl"
TEST_JSONL = "train_json_data/five_plus_two_train_jsonl_data/design_3.27/base_model_train/test_set_100_actionized_sort_floor_force_auged(3_times).jsonl"
#MODEL_ID =  "baidu/ERNIE-4.5-0.3B-PT" #训练模型
#MODEL_ID = "google/gemma-3-1b-it"
MODEL_PATH="five_plus_two_optimization/train_model_new/GRPO/save_model/grpo_fpt_api/grpo_fpt_api_5/step_1000"
BATCH_SIZE,GRAD_ACCUM = 1,2
LEARNING_RATES = [1e-3,9e-4,6e-4,3e-4,1e-4,9e-5,6e-5,3e-5,1e-5,9e-6,6e-6,3e-6,1e-6]
#LEARNING_RATE=LEARNING_RATES[4] #学习率
LEARNING_RATE=5e-5
EPOCH = 10 #epoch
LORA_R,LORA_ALPHA = 8,8 #lora参数
MAX_LENGTH=10000 #训练序列最大token数
TARGET_MODULES = [ #进行LoRA adapter的模块
    "q_proj", #self-Attention Query
    "v_proj", #self-Attention Value
    "k_proj", #self-Attention Key
    "o_proj", #self-Attention attention输出线性变换
    "gate_proj", #FFN前半部分门控部分
    "up_proj", #FFN前半部分扩张变换部分
    "down_proj", #FFN后半部分降维部分
]
#TRAIN_NUM,TEST_NUM=14651,3295
TRAIN_NUM,TEST_NUM=9582,3295
#初始化函数

def initialize_token_embedding(model,tokenizer):
    #得到现有token的emb
    existing_tokens,existing_embs=["wall","exterior_wall","opening","boundary","beam","shearwall","post","lineload","add","remove"],[]
    for i in range(256):
        existing_tokens.append(f'{i}')
    for token in existing_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        #print(ids)
        emb=model.get_input_embeddings().weight.data[ids]
        emb = emb.mean(dim=0)
        existing_embs.append(emb)

    #初始化新加入的token
    new_tokens,new_ids = ['<wall>','<exterior_wall>','<opening>','<inoutbox>','<beam>','<shearwall>','<POST>','<LINELOAD>','<add>','<remove>'],[]

    tokenizer.add_special_tokens({
        "additional_special_tokens": new_tokens
    })

    model.resize_token_embeddings(len(tokenizer)) #将新的token加入tokenizer和model

    for token in new_tokens:#得到新词的id
        tid = tokenizer.convert_tokens_to_ids(token)
        new_ids.append(tid)

    for i,new_id in enumerate(new_ids):
        with torch.no_grad():
            model.get_input_embeddings().weight[new_id] = existing_embs[i]

class MyTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)

        # 1. 获取 embedding 权重（Gemma / LLaMA 架构）
        try:
            emb = self.model.base_model.model.model.embed_tokens.weight.data.detach().cpu()
        except:
            # 有些模型路径不一样
            emb = self.model.get_input_embeddings().weight.data.detach().cpu()

        # 2. 保存到 checkpoint 目录
        emb_path = os.path.join(output_dir, "embedding.safetensors")
        torch.save(emb, emb_path)
        print(f"\n[Embedding Saved] → {emb_path}\n")

        # 3. 保存 tokenizer（可选，但通常是需要的）
        if hasattr(self, "tokenizer") and self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)

class SaveLossCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        # 当前 checkpoint 目录
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        # 确保目录存在
        os.makedirs(ckpt_dir, exist_ok=True)

        # 只保存 loss 相关的日志
        loss_logs = [entry for entry in state.log_history if "loss" in entry or "eval_loss" in entry]

        # 写入 JSON
        with open(os.path.join(ckpt_dir, "loss_log.json"), "w", encoding="utf8") as f:
            json.dump(loss_logs, f, ensure_ascii=False, indent=2)

        print(f"Saved loss log at {ckpt_dir}/loss_log.json")
        return control

def clean_memory():
    """在训练前彻底清理 GPU / CPU 缓存。"""
    
    # 1) 清理 Python 垃圾对象
    gc.collect()

    # 2) 清理 CUDA 预留内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("✔ Memory cleaned.")

def test_token_length():
    #从零初始化模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID) #自动加载Gemma3对应tokenizer
    if tokenizer.pad_token is None: #如果tokenizer中没有pad_token,就用eos_token填充
        tokenizer.pad_token = tokenizer.eos_token
 
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto", #自动把模型分配到gpu
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, #使用半精度节省显存，如gpu不支持则用32位浮点
        use_cache=False,  # 禁用KV缓存，与梯度检查点冲突
    )
    
    #text="Making actions based on context and given structures.context:<exterior_wall>(2648,8306),(2706,8306)<exterior_wall>(2648,8306),(2648,8447)<exterior_wall>(2648,8537),(2648,8616)<exterior_wall>(2648,8706),(2648,8756)<wall>(2648,8756),(2824,8756)<exterior_wall>(2648,8756),(2648,8906)<wall>(2648,8906),(2824,8906)<exterior_wall>(2648,8906),(2648,9116)<exterior_wall>(2648,9116),(2723,9116)<exterior_wall>(2826,8306),(3398,8306)<exterior_wall>(2843,9116),(2918,9116)<wall>(2894,8756),(2918,8756)<wall>(2894,8906),(2918,8906)<wall>(2918,8756),(2942,8756)<wall>(2918,8756),(2918,8906)<wall>(2918,8906),(2918,9116)<exterior_wall>(2918,9116),(3023,9116)<wall>(3032,8756),(3278,8756)<exterior_wall>(3173,9116),(3278,9116)<exterior_wall>(3278,8576),(3398,8576)<exterior_wall>(3278,8576),(3278,8600)<exterior_wall>(3278,8720),(3278,8756)<exterior_wall>(3278,8756),(3278,9116)<exterior_wall>(3398,8306),(3398,8576)<truss_area>(3455,8249),(2591,8249),(2591,9173),(3335,9173),(3335,8633),(3455,8633),(3455,8249)structures:<beam>(2648,8447),(2648,8537)<beam>(2648,8616),(2648,8706)<beam>(2706,8306),(2826,8306)<beam>(2723,9116),(2843,9116)<beam>(2824,8906),(2894,8906)<beam>(2894,8756),(2824,8756)<beam>(2918,8906),(3278,8906)<shearwall>(2648,8306),(2706,8306)<shearwall>(2648,8306),(2648,8447)<shearwall>(2648,8537),(2648,8616)<shearwall>(2648,8706),(2648,8756)<shearwall>(2648,8756),(2824,8756)<shearwall>(2648,8756),(2648,8906)<shearwall>(2648,8906),(2824,8906)<shearwall>(2648,8906),(2648,9116)<shearwall>(2648,9116),(2723,9116)<shearwall>(2826,8306),(3398,8306)<shearwall>(2843,9116),(2918,9116)<shearwall>(2894,8756),(2918,8756)<shearwall>(2894,8906),(2918,8906)<shearwall>(2918,8756),(2942,8756)<shearwall>(2918,9116),(3023,9116)"
    text="<remove><beam>(2648,8306),(2706,8306)"
    prompt_tokens = len(tokenizer(text).input_ids)
    print(f"引入新token前prompt_tokens:{prompt_tokens}")

    initialize_token_embedding(model,tokenizer)
    prompt_tokens = len(tokenizer(text).input_ids)
    print(f"引入新token后prompt_tokens:{prompt_tokens}")
    
    
    '''
    #测试token对比
    input_path="train_json_data/five_plus_two_train_jsonl_data/design_3.10/initial_data_train_set_actionized.jsonl"
    list=[]
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            prompt=record['prompt']
            prompt_tokens = len(tokenizer(prompt).input_ids)
            list.append(prompt_tokens)

    list = np.array(list)
    print(f"之前prompt平均token数：{np.mean(list)},最小：{np.min(list)},最大：{np.max(list)},\
        3000超过了{np.sum(list < 3000) / len(list)}的数据")
    
    initialize_token_embedding(model,tokenizer)

    input_path="train_json_data/five_plus_two_train_jsonl_data/design_3.10/initial_data_train_set_actionized.jsonl"
    list=[]
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            prompt=record['prompt']
            prompt_tokens = len(tokenizer(prompt).input_ids)
            list.append(prompt_tokens)

    list = np.array(list)
    print(f"现在prompt平均token数：{np.mean(list)},最小：{np.min(list)},最大：{np.max(list)},\
        3000超过了{np.sum(list < 3000) / len(list)}的数据")
    '''

if __name__ == "__main__":
    clean_memory()
    # ------------------------
    # 定义模型与tokenizer
    # ------------------------
    #test_token_length()
    
    #直接加载模式
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto",torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()
    '''
    #从零初始化模型
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
    '''
    
    #定义训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        max_steps=-1, #按照epoch进行训练
        num_train_epochs=EPOCH,
        logging_steps=2, #日志记录频率

        prediction_loss_only=True,
        #save_strategy="epoch",
        save_strategy="steps",
        eval_strategy="steps", #var
        eval_steps=20, #var
        per_device_eval_batch_size=1,
        #eval_strategy="no",  # 关闭evaluation #var
        save_steps=1000, #依据step和epoch的对应关系设置save_step大小 #var
        save_total_limit=50,#最多保存checkpoint个数（checkpoint可以生成快照，方便继续训练） #var

        bf16=torch.cuda.is_available(),
        fp16=False,
        optim="adamw_torch", #优化器类型
        report_to="none", #不汇报日志
        lr_scheduler_type="constant"
        #weight_decay=0.0
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=0.1, #lora层dropoout，防止过拟合
        bias="none", #lora不会改变原模型的偏置参数
        task_type="CAUSAL_LM" #GPT模型选择类型
    )

    #定义数据
    def preprocess_function(example):
        prompt = example["prompt"]
        response = example["response"]

        full_text = f"{prompt}{response}" #将prompt和completion拼接成Chat风格
        encodings = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, return_tensors=None) #
        prompt_ids = tokenizer(f"{prompt}", truncation=True, max_length=MAX_LENGTH)["input_ids"] 
        labels = [-100] * len(prompt_ids) + encodings["input_ids"][len(prompt_ids):]
        
        return {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"], "labels": labels}
    
    print("Loading dataset...")
    train_data_list=[]
    with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                train_data_list.append(json.loads(line))
    train_dataset=Dataset.from_list(train_data_list)
    train_dataset = train_dataset.map(preprocess_function, batched=False,\
                                    remove_columns=train_dataset.column_names)
    
    #输入测试数据
    test_data_list=[]
    test_sample_point=np.linspace(0,TEST_NUM-1,10,dtype=int) #var
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i not in test_sample_point:   #指定读取内容
                continue
            if line.strip():
                test_data_list.append(json.loads(line))
    test_dataset = Dataset.from_list(test_data_list)
    test_dataset = test_dataset.map(preprocess_function, batched=False,\
                                      remove_columns=test_dataset.column_names)
    print(f"test Dataset size: {len(test_dataset)}")
    
    # --- 注入 LoRA ---
    model = get_peft_model(model, lora_config)

    # --- 冻结主模型参数（不包括 embedding 和 LoRA 参数） ---
    for name, p in model.named_parameters():
        if "embed_tokens" in name:  # 新 embedding 参数
            p.requires_grad = True
        elif "lora" in name:  # LoRA 参数
            p.requires_grad = True
        else:
            p.requires_grad = False
    trainer=MyTrainer( #模型，训练参数，训练数据，collator
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset, #var
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[SaveLossCallback()],
    )
    trainer.train()

    loss_dir=f"{OUTPUT_DIR}/loss_history.json"
    with open(loss_dir, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"Done! Model saved to {OUTPUT_DIR}")

    trainer.model=trainer.model.merge_and_unload() #合并参数后保存模型
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)