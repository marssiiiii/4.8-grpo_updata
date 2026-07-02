import json
import sys
sys.path.append("/mnt/efs/jiuxing_li")
sys.path.append("/mnt/efs/jiuxing_li/five_plus_two_optimization/train_model_new/base_code")
from standard_function import get_segments_info_five_plus_two
from shapely.geometry import Polygon,LineString
from reward_design import fix_polygon

POLYGON_TYPES = ["opening","inoutbox"]
LINE_TYPES = ["wall","exterior_wall","beam","shearwall"]
POINT_TYPES = []

input_path="train_json_data/five_plus_two_train_jsonl_data/design_3.27/initial_data_100_test_set.jsonl"
output_path="train_json_data/five_plus_two_train_jsonl_data/design_3.27/initial_data_100_test_set_inbox.jsonl"

def filter_lines_in_box(lines,inoutbox_poly):
    result_lines=[]
    for line in lines:
        other_line=LineString([(line[1][0],line[1][1]),(line[2][0],line[2][1])])
        if other_line.covered_by(inoutbox_poly)==True:
            result_lines.append(line)
    return result_lines

def filter_openings_in_box(openings,inoutbox_poly):
    result_openings=[]
    for opening in openings:
        points = [(p[0], p[1]) for p in opening[1:]]
        #保证points首位相连
        if points[0] != points[-1]:
            points.append(points[0])
        polygon = Polygon(points)
        if polygon.is_valid==False:
            IS_FIXED,polygon=fix_polygon(polygon)
            if IS_FIXED==False:
                continue
        if polygon.covered_by(inoutbox_poly)==True:
            result_openings.append(opening)
    return result_openings

with open(input_path, "r", encoding="utf-8") as fin,\
    open(output_path,"w",encoding="utf-8") as fout:
    for index,line in enumerate(fin):
        print(f"进度：{index+1}")
        #if index!=10:
        #    continue
        record=json.loads(line)
        house,floor,design,bound,context,completion_predict=record["house"],record["floor"],record["design"],\
        record["bound"],record["context"],record["completion_predict"]
        
        inoutbox_list=get_segments_info_five_plus_two(context,["inoutbox"],[],[]) #得到inoutbox所围成的多边形
        if len(inoutbox_list)!=1:
            print(f"{house},{floor},{design},错误，inoutbox不唯一！")
            continue
        inoutbox=inoutbox_list[0]
        inoutbox_pts = [(p[0], p[1]) for p in inoutbox[1:]]

        if inoutbox_pts[0] != inoutbox_pts[-1]: #保证inoutbox是有效的
            inoutbox_pts.append(inoutbox_pts[0])
        inoutbox_poly = Polygon(inoutbox_pts)
        if inoutbox_poly.is_valid==False:
            IS_FIXED,inoutbox_poly=fix_polygon(inoutbox_poly)
            if IS_FIXED==False:
                print(f"{house},{floor},{design},错误，inoutbox无法修复！")
                continue
        
        context_lines=get_segments_info_five_plus_two(context,[],LINE_TYPES,[]) #提取context中在边界内部的线
        context_lines_filt=filter_lines_in_box(context_lines,inoutbox_poly)
        openings=get_segments_info_five_plus_two(context,["opening"],[],[]) #提取context中的在边界内的opening
        openings_filt=filter_openings_in_box(openings,inoutbox_poly)
        completion_predict_lines=get_segments_info_five_plus_two(completion_predict,[],LINE_TYPES,[]) #提取completion_predict在边界内的线
        completion_predict_lines_filt=filter_lines_in_box(completion_predict_lines,inoutbox_poly)

        context_filtered=""
        for wall in context_lines_filt: #加入exteriro_wall和wall
            context_filtered=context_filtered+f"<{wall[0]}>({wall[1][0]},{wall[1][1]}),({wall[2][0]},{wall[2][1]})"
        for opening in openings_filt: #加入opening
            context_filtered=context_filtered+f"<{opening[0]}>"
            for i in range(1,len(opening)):
                context_filtered=context_filtered+f"({opening[i][0]},{opening[i][1]})"
                if i!=len(opening)-1:
                    context_filtered=context_filtered+","
        context_filtered=context_filtered+f"<{inoutbox[0]}>" #加入inoutbox
        for i in range(1,len(inoutbox)):
            context_filtered=context_filtered+f"({inoutbox[i][0]},{inoutbox[i][1]})"
            if i!=len(inoutbox)-1:
                context_filtered=context_filtered+","
        
        completion_predict_filtered=""
        for seg in completion_predict_lines_filt:
            completion_predict_filtered=completion_predict_filtered+f"<{seg[0]}>({seg[1][0]},{seg[1][1]}),({seg[2][0]},{seg[2][1]})"
        
        #print(context_filtered,completion_predict_filtered)
        fout.write(json.dumps({"house":house,"floor":floor,"design":design,"bound":bound,
                        "context": context_filtered,"completion_predict":completion_predict_filtered}
                        , ensure_ascii=False) + "\n")