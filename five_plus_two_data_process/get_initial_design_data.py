import requests
import time
import os
import json

url = "https://generate-service-552099422773.us-central1.run.app/ai/generate"
GENERATE_NUM=300
def filter_and_count_jsonl(file_path):
    keyword = "code"
    filtered_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if keyword not in data.keys():
                    print(data.keys())
                # 判断是否包含关键词
                if keyword in data:
                    filtered_data.append(data)
            except:
                continue
    
    if len(filtered_data)>GENERATE_NUM:
        filtered_data=filtered_data[-GENERATE_NUM:]
    # 覆盖写回文件
    with open(file_path, "w", encoding="utf-8") as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 统计过滤后的条数
    print("过滤后条数:", len(filtered_data))
    return len(filtered_data)

if __name__=="__main__":
    input_folder_path="/mnt/efs/jiuxing_li/house_data/noopening_test_detail"
    for i,filename in enumerate(os.listdir(input_folder_path)):#遍历房子
        #if i<91 or i>100:
        #    continue
        sta_time = time.time()
        input_path = f"{input_folder_path}/{filename}/house_items.json"
        output_path=f"{input_folder_path}/{filename}/design.jsonl"
        
        if not os.path.exists(input_path):
            print(f"{input_path}不存在")
            continue
        if os.path.exists(output_path):
            print(f"{output_path}已存在")
            cnt=filter_and_count_jsonl(output_path)
            if cnt>=GENERATE_NUM:
                continue
            else:
                repeat_times=GENERATE_NUM-cnt
                print(f"还需要生成{repeat_times}条")
        for repeat_time in range(min(repeat_times,GENERATE_NUM)):
            print(f"filename:{filename},repeat_time:{repeat_time}")
            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    house_items = json.load(f)
                data = {
                    "house_items": json.dumps(house_items),
                }
                response = requests.post(url,data=data)
                #response="ok"
                #response = requests.post(url, files=files)
                print(response.status_code)
                print(response.text)
                
                if response.status_code==200:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, "a", encoding="utf-8") as f: #追加文件而不是覆盖
                        wrapped = {
                            "house":filename,
                            "code":repeat_time,
                            "design":response.text,
                        }
                        f.write(json.dumps(wrapped, ensure_ascii=False) + "\n")
            except:
                continue
                
        end_time = time.time()
        print(f"用时 {(end_time-sta_time)/60} 分钟")
        #break