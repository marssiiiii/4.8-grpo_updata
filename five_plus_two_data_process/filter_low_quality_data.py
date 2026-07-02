import json

input_file="train_json_data/five_plus_two_train_jsonl_data/design_3.4/initial_data.jsonl"
output_file="train_json_data/five_plus_two_train_jsonl_data/design_3.4/initial_data_filtered.jsonl"

invalid_list=[1,2,3,4,22,23,24,25,26,53,59,60,61,62,68,69,70,71,72,74,75,76,77]

with open(input_file, 'r', encoding='utf-8') as infile, \
    open(output_file, 'w', encoding='utf-8') as outfile:
    for i, line in enumerate(infile):
        if (i+1) not in invalid_list:
            data = json.loads(line.strip())
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
