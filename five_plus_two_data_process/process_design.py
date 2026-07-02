import os
import json

if __name__=="__main__":
    input_folder_path="house_data/noopening_test_detail"
    MAX_SOLUTIONS=300
    for file_index,filename in enumerate(os.listdir(input_folder_path)):#遍历房子
        file_path=f"{input_folder_path}/{filename}/design.jsonl"
        print(file_path)
        with open(file_path, "r", encoding="utf-8") as fin:
            for SOLUTION_ID, line in enumerate(fin):
                output_path=f"{input_folder_path}/{filename}/design_{SOLUTION_ID+1}.jsonl"
                print(f"output_path:{output_path}")
                try:
                    decoder = json.JSONDecoder()
                    record, _ = decoder.raw_decode(line)
                    #record=json.loads(line)
                    design=record['design']
                    design=json.loads(design)
                    data=design['data']
                    designs=data['designs']
                    designs=designs[0]
                    #floor = next(iter(designs))
                    #print(floor)
                    with open(output_path, "w", encoding="utf-8") as fout:
                        json.dump(designs, fout, ensure_ascii=False,indent=2)
                    if SOLUTION_ID>=MAX_SOLUTIONS-1:
                        break
                except:
                    print(f"{output_path} 数据读取失败！")
                    if SOLUTION_ID>=MAX_SOLUTIONS-1:
                        break
        #break