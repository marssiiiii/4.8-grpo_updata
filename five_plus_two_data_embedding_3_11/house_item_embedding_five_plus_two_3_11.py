import json
import os
from shapely.geometry import LineString,Point
from typing import List, Set, Any
from pathlib import Path
from embedding_define_five_plus_two_3_11 import StructuralSegment,Segment_Embedding
from collections import defaultdict, deque
from Line_Judgement_3_11 import *

POLYGON_TYPES = ["opening","inoutbox"]
LINE_TYPES = ["beams","beam","wall","shearwalls",'shearwall']
POINT_TYPES = []
ELEMENT_TYPES = ["wall","opening","inoutbox","beams","beam","shearwalls",'shearwall']
#ELEMENT_TYPES=POINT_TYPES

TRANSFER={"opening":"opening","inoutbox":"inoutbox",
          "beams":"beam","wall":"wall","shearwalls":"shearwall",}

def house_item_get_segments(file_path:str)->List["StructuralSegment"]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments=[]
    if data is None:
        return segments
    for floor_name, floor_data in data.items(): # 遍历每一层
        if not floor_name.startswith(("f", "b")):
            continue
        for code_index,elem_type in enumerate(ELEMENT_TYPES): #遍历每一层中的元素
            if elem_type not in ELEMENT_TYPES:
                continue
            items = floor_data.get(elem_type, [])
            for item in items:
                if elem_type=="beams" or elem_type=="shearwalls":#提取元素的点集到points_list
                    points_list=[]
                    beam_line=item.get("line")
                    st_pt,end_pt=beam_line["start"],beam_line["end"]
                    if not st_pt or not end_pt:
                        continue
                    points_list.append(st_pt)
                    points_list.append(end_pt)
                else:
                    points_list = item.get("points", [])
                if not points_list:
                    print(f"{elem_type}{item}未检测出point")
                    continue
                
                wall_is_exterior=False #提取元素的exterior属性
                if elem_type=="wall":
                    wall_type=item.get("type",[])
                    if wall_type!=[] and wall_type[0]=="exterior":
                        wall_is_exterior=True

                if elem_type in POLYGON_TYPES: #构造segment
                    if len(points_list)<3:
                        print("POLYGON 点数不足！")
                        continue
                    poly=[]
                    for i in range(len(points_list)):
                        poly.append(Point(points_list[i]))

                    segment = StructuralSegment(
                        floor=floor_name,
                        polygon=poly,
                        type=TRANSFER[elem_type],
                        general_type="poly",
                        build_order=-1,
                    )
                elif elem_type in LINE_TYPES:
                    if len(points_list)!=2:
                        continue
                    if wall_is_exterior==True:
                        attr_type="exterior"
                    else:
                        attr_type=None
                    
                    line = LineString(points_list)
                    segment = StructuralSegment(
                            floor=floor_name,
                            line_string=line,
                            type=TRANSFER[elem_type],
                            general_type="line",
                            attribute_type=attr_type,
                            build_order=-1,
                    )
                elif elem_type in POINT_TYPES:
                    if len(points_list)!=1:
                        continue
                    pt=Point(points_list[0])
                    segment = StructuralSegment(
                        floor=floor_name,
                        point=pt,
                        type=TRANSFER[elem_type],
                        general_type="point",
                        build_order=-1
                    )
                segments.append(segment)
    return segments

# 批量处理目录下所有 JSON 文件
def process_dir(get_segments,file_path: str):
    segments=get_segments(file_path)
    if segments==[]:
        return -1
    return segments

def sort_and_extract_lines(lines,segments): #输入单层楼的line类型,提取整合信息放入segments
    lines_sorted = sorted(lines,
                key=lambda s: (s.line_string.xy[0][0],s.line_string.xy[1][0]))
    encoder=Segment_Embedding(lines_sorted)
    codes=encoder.encode_all()
    # 编号
    for code in codes:
        line = f"<{code[1]}>,{code[0]},({int(code[3].x)},{int(code[3].y)}),({int(code[4].x)},{int(code[4].y)})"
        segments.append(line)
    return segments

def sort_and_extract_polygons(polygons,segments): #输入单层楼的line类型,提取整合信息放入segments
    polygons_sorted = sorted(polygons,
                key=lambda s: (s.polygon[0].x,s.polygon[0].y))
    encoder=Segment_Embedding(polygons_sorted)
    codes=encoder.encode_all()
    # 编号
    for code in codes:
        line = f"<{code[1]}>,{code[0]},"
        for i in range(3,len(code)):
            line=line+f"({int(code[i].x)},{int(code[i].y)})"
            if i!=len(code)-1:
                line=line+","
        segments.append(line)
    return segments

def sort_and_extract_points(points,segments): #输入单层楼的line类型,提取整合信息放入segments
    points_sorted = sorted(points,
                key=lambda s: (s.point.x,s.point.y))
    encoder=Segment_Embedding(points_sorted)
    codes=encoder.encode_all()
    # 编号
    for code in codes:
        line = f"<{code[1]}>,{code[0]},({int(code[3].x)},{int(code[3].y)})"
        segments.append(line)
    return segments

def house_item_format_segments_grouped(segments:List[StructuralSegment]):
    floor_groups = defaultdict(list) #按照floor进行分组
    for seg in segments:
        floor_groups[seg.floor].append(seg)
    
    def floor_sort_key(floor):
        if floor.startswith("f"):
            return -int(floor[1:])  # f2 < f1
        elif floor.startswith("b"):
            return int(floor[1:])   # b1 < b2
        else:
            return 0

    segments = []

    for floor in sorted(floor_groups.keys(), key=floor_sort_key): #按照楼层进行排序
        segs_in_floor = floor_groups[floor]
        type_groups = defaultdict(list) #每层楼的线段
        for seg in segs_in_floor:
            type_groups[seg.type].append(seg)
        
        #print(type_groups)

        for typ in type_groups: #遍历每种类型的线段,提取信息加入segments
            segs_of_type = type_groups[typ]
            if typ in LINE_TYPES:
                segments=sort_and_extract_lines(segs_of_type,segments)
            elif typ in POLYGON_TYPES:
                segments=sort_and_extract_polygons(segs_of_type,segments)
            elif typ in POINT_TYPES:
                segments=sort_and_extract_points(segs_of_type,segments)
    
    return segments

if __name__ == "__main__":
    file_path="house_data/noopening_test_detail/23-05-31Millerd/house_items.json"
    segments=process_dir(house_item_get_segments,file_path)
    #print(segments)
    context="The structure of the house:\n"
    for line in house_item_format_segments_grouped(segments):
        context=context+line
    print(context)
    
    file_path_stru="house_data/noopening_test_detail/23-05-31Millerd/design_4.jsonl"
    segments_stru=process_dir(house_item_get_segments,file_path_stru)
    #print(segments)
    structures="The structure of the house:\n"
    for line in house_item_format_segments_grouped(segments_stru):
        structures=structures+line
    print(structures)
    
    #results = process_house_item_dir("测试数据/house_items")