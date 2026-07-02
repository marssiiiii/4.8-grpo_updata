import bottle, queue, torch, io, json, threading, time
from bottle import request
from queue import Queue
import threading
import os
from transformers import AutoTokenizer
#base_model_path="/home/jiuxing_li/five_plus_two_optimization/train_model_new/GRPO/save_model/grpo_fpt_api/grpo_fpt_api_3_2/step_2000" #var
base_model_path="/home/jiuxing_li/five_plus_two_optimization/train_model_new/GRPO/base_model/ernie_base_model_10"
#tokenizer = AutoTokenizer.from_pretrained(base_model_path)

raw_queue = queue.Queue()
result_queue = queue.Queue()
app = bottle.Bottle()

# ✅ 新增：模型版本管理 + 更新锁
model_version_info = {
    "version": 0,
    "path": None,
    "train_is_updating": False,  # ✅ 训练过程更新标志
    "gen_worker_is_updating":False, # gen_worker更新标志
}
model_version_lock = threading.Lock()

# train开始更新
@app.route('/lock_train_update', method='POST')
def lock_train_update():
    global model_version_info
    try:
        data = json.loads(request.body.read().decode())
        with model_version_lock:
            model_version_info['train_is_updating'] = True
        print(f"[SERVER] Model update LOCKED by train at step {data.get('step', 'unknown')}")
        return json.dumps({'status': 'locked'})
    except Exception as e:
        print(f"[SERVER] Error in lock_train_update: {e}")
        return json.dumps({'status': 'error', 'message': str(e)})

# train完成更新
@app.route('/unlock_train_update', method='POST')
def unlock_train_update():
    global model_version_info
    try:
        data = json.loads(request.body.read().decode())
        with model_version_lock:
            model_version_info['version'] = data['version']
            model_version_info['path'] = data['path']
            model_version_info['train_is_updating'] = False
        print(f"[SERVER] Model update UNLOCKED: version={data['version']}, path={data['path']}")
        return json.dumps({'status': 'unlocked'})
    except Exception as e:
        print(f"[SERVER] Error in unlock_model_update: {e}")
        return json.dumps({'status': 'error', 'message': str(e)})

@app.route('/lock_gen_worker_update', method='POST')
def lock_gen_worker_update():
    """train进程开始更新模型前调用，设置更新锁"""
    global model_version_info
    try:
        data = json.loads(request.body.read().decode())
        with model_version_lock:
            model_version_info['gen_worker_is_updating'] = True
        print(f"[SERVER] Model update LOCKED by gen worker at iteration {data.get('it', 'unknown')}")
        return json.dumps({'status': 'locked'})
    except Exception as e:
        print(f"[SERVER] Error in lock_gen_worker_update: {e}")
        return json.dumps({'status': 'error', 'message': str(e)})

@app.route('/unlock_gen_worker_update', method='POST')
def unlock_gen_worker_update():
    global model_version_info
    try:
        data = json.loads(request.body.read().decode())
        with model_version_lock:
            model_version_info['gen_worker_is_updating'] = False
        print(f"[SERVER] Gen Worker update UNLOCKED: version={data['version']}, path={data['path']}")
        return json.dumps({'status': 'unlocked'})
    except Exception as e:
        print(f"[SERVER] Error in unlock_model_update: {e}")
        return json.dumps({'status': 'error', 'message': str(e)})

#通知有新模型可用
@app.route('/notify_model', method='POST')
def notify_model():
    """接收训练进程的模型更新通知"""
    global model_version_info
    try:
        data = json.loads(request.body.read().decode())
        with model_version_lock:
            model_version_info['version'] = data['version']
            model_version_info['path'] = data['path']
        print(f"[SERVER] Model updated: version={data['version']}, path={data['path']}")
        return json.dumps({'status': 'ok'})
    except Exception as e:
        print(f"[SERVER] Error in notify_model: {e}")
        return json.dumps({'status': 'error', 'message': str(e)})

#查询最新模型版本
@app.route('/model_version', method='GET')
def get_model_version():
    """返回当前最新的模型版本信息"""
    bottle.response.content_type = 'application/json'
    with model_version_lock:
        return json.dumps(model_version_info)

def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t.detach().cpu(), buffer)  # ⭐ 核心
    return buffer.getvalue()

def bytes_to_tensor(b, device=None):
    t = torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)
    if device is not None:
        t = t.to(device)
    return t

def make_bytes_list(blist):#把一个字节对象列表压缩成一个连续的字节流
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()
def bytes_list_to_list(b): #把make_bytes_list打包的字节流拆回原来的字节对象列表。
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist

@app.route('/upload', method='POST') #Actor发数据，server把原始batch放入raw_queue等待处理
def do_upload():
    print("上传！")
    dd = request.body.read()
    raw_queue.put(dd)
    print('[SERVER] received batch! Queue size:', raw_queue.qsize())
    return b'tensor'

@app.route('/get', method='GET') #Learner拉取batch,如果队列为空则返回empty,若队列有数据则返回处理好的数据
def do_get():
    if result_queue.empty():
        return b'empty'
    print('[SERVER] send batch from result_queue, size:', result_queue.qsize()-1)
    return result_queue.get()

# ✅ 新增：查询队列大小接口
@app.route('/queue_size', methods=['GET'])
def queue_size():
    """返回当前队列中的数据量"""
    bottle.response.content_type = 'application/json'
    size = result_queue.qsize()
    print(f"[SERVER] Queue size requested: {size}")  # 添加日志
    return json.dumps({'queue_size': size})

def batch_worker():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_path = base_model_path
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16).to('cuda:0') #加载裸模型，为一个只读模型不参与梯度更新 #var
    ref_model.eval()
    ref_model.requires_grad_(False)

    # latest model (最新训练检查点) tracking
    latest_model = None
    latest_model_version = -1

    def get_per_token_logps(model, input_ids): #返回指定模型输出的token概率分布
        logits = model(input_ids).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V)
        ids = input_ids[:, 1:]  # (B, L-1)
        per_token_logps = []
        for logits_row, ids_row in zip(logits, ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def try_load_latest_model():
        nonlocal latest_model, latest_model_version
        with model_version_lock:
            version = model_version_info['version']
            path = model_version_info['path']
            is_updating = model_version_info['train_is_updating']
        if is_updating or path is None or version <= latest_model_version:
            return
        try:
            new_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to('cuda:0')
            new_model.eval()
            new_model.requires_grad_(False)
            latest_model = new_model
            latest_model_version = version
            print(f'[BATCH_WORKER] Loaded latest model version {version} from {path}')
        except Exception as e:
            print(f'[BATCH_WORKER] Failed to load latest model: {e}')

    while True:
        try:
            try_load_latest_model()

            d = raw_queue.get()
            dd = bytes_list_to_list(d)
            meta = json.loads(dd[0])
            inputs = bytes_to_tensor(dd[1])
            rewards_std = bytes_to_tensor(dd[2])
            rewards=bytes_to_tensor(dd[3])

            diaphragm_delta_score_list=bytes_to_tensor(dd[4])
            dcr_delta_list=bytes_to_tensor(dd[5])
            error_num_list=bytes_to_tensor(dd[6])
            invalid_process_num_list=bytes_to_tensor(dd[7])
            floor_price_list=bytes_to_tensor(dd[8])
            diaphragm_area_ratio_list=bytes_to_tensor(dd[9])

            design_error_score_list=bytes_to_tensor(dd[10])
            design_invalid_process_score_list=bytes_to_tensor(dd[11])
            design_diaphragm_score_list=bytes_to_tensor(dd[12])
            deisgn_dcr_score_list=bytes_to_tensor(dd[13])
            design_price_score_list=bytes_to_tensor(dd[14])

            #print(f"here in ref_client:{tokenizer.decode(inputs[0].tolist())}")
            #print(f"rewards:{rewards}")
            prompt_length = meta['plen']
            with torch.inference_mode():
                per_token_logps = get_per_token_logps(ref_model, inputs.to(ref_model.device)) #计算裸模型per_token_logps
                per_token_logps = per_token_logps[:, prompt_length-1:]

                if latest_model is not None:
                    latest_per_token_logps = get_per_token_logps(latest_model, inputs.to(latest_model.device))
                    latest_per_token_logps = latest_per_token_logps[:, prompt_length-1:]
                else:
                    # 尚无最新模型，以ref_model的logps作为占位
                    latest_per_token_logps = per_token_logps.clone()

            data = [json.dumps(meta).encode(), tensor_to_bytes(inputs),
                    tensor_to_bytes(rewards_std), tensor_to_bytes(per_token_logps),tensor_to_bytes(rewards),
                    tensor_to_bytes(diaphragm_delta_score_list), tensor_to_bytes(dcr_delta_list), tensor_to_bytes(error_num_list),
                    tensor_to_bytes(invalid_process_num_list), tensor_to_bytes(floor_price_list), tensor_to_bytes(diaphragm_area_ratio_list),
                    tensor_to_bytes(design_error_score_list),tensor_to_bytes(design_invalid_process_score_list),tensor_to_bytes(design_diaphragm_score_list),
                    tensor_to_bytes(deisgn_dcr_score_list),tensor_to_bytes(design_price_score_list),
                    tensor_to_bytes(latest_per_token_logps),]  # [16] latest model logps

            xdata = make_bytes_list(data)
            result_queue.put(xdata)
            print('[SERVER] processed batch, result_queue size:', result_queue.qsize())
        except Exception as e:
            print('[SERVER] Batch process error:', e)
            time.sleep(1)

if __name__ == '__main__':
    worker = threading.Thread(target=batch_worker, daemon=True) #表示开一个线程不断执行batch_worker任务
    worker.start()
    #print("Launching HTTP server on 59875 ...")
    #bottle.run(app, host='127.0.0.1', port=59875, server='tornado')
    print("Launching HTTP server on 59874 ...")
    bottle.run(app, host='127.0.0.1', port=59874, server='tornado')