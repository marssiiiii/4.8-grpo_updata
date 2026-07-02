import json
import sys
sys.path.append("/home/jiuxing_li")
sys.path.append("code/GRPO/test/test_code")
sys.path.append("five_plus_two_optimization/five_plus_two_data_embedding_3_11")
sys.path.append("five_plus_two_optimization/five_plus_two_test")
from standard_function import get_segments_info_five_plus_two
from five_plus_two_visualisation_single_floor import plot_diaphragms,plot_segments
from shapely.geometry import LineString,Point
from shapely.ops import polygonize
import matplotlib.pyplot as plt
from grpo_no_depend_test import point_is_on_line
from Line_Judgement_3_11 import orientation
import re
from transformers import AutoTokenizer
from shapely.geometry import Polygon
import numpy as np
from shapely.ops import unary_union
#from five_plus_two_optimization.train_model_new.base_code.reward_design_dcr_price_trunc import get_initial_struct_based_on_response

POLYGON_TYPES = ["opening","inoutbox"]
LINE_TYPES = ["wall","beam","exterior_wall"]
POINT_TYPES = []

def visualize_six_axes(context,house_areas,cp_beams,cp_shearwalls,beams,shearwalls,valid_shear_walls,
    valid_beams,valid_diaphragms_before,diaphragms,valid_diaphragms,bound,output_path):
    #print(f"valid_diaphragms:{valid_diaphragms_return}")
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    ax1, ax2, ax3 = axes[0] #初始结构，初始结构+初始方案(含invalid beams),初始结构+设计方案(含invalid beams)
    ax4, ax5, ax6 = axes[1] #初始结构+设计方案(只有valid bemas),初始结构+设计方案(diaphragms,只含valid_beams),初始结构+设计方案(valid_diaphragms,只含valid_beams)
    initial_structures=get_segments_info_five_plus_two(context,POLYGON_TYPES,LINE_TYPES,POINT_TYPES)
    #print(answer_predict_beams+shear_walls_answer)
    plot_segments(ax1,initial_structures,POLYGON_TYPES,LINE_TYPES,POINT_TYPES,cut_line_list=None,title="IS",bound=bound)
    plot_segments(ax2,initial_structures,POLYGON_TYPES,["wall","beam","exterior_wall"],POINT_TYPES,cut_line_list=cp_beams+cp_shearwalls,title="IS+CP",bound=bound)
    plot_segments(ax3,initial_structures,POLYGON_TYPES,["wall","beam","exterior_wall","shearwall"],POINT_TYPES,cut_line_list=beams+shearwalls,title="IS+answer_IV",bound=bound)
    plot_segments(ax4,initial_structures,POLYGON_TYPES,["wall","beam","exterior_wall","shearwall"],POINT_TYPES,cut_line_list=valid_beams+valid_shear_walls,title="IS+answer_V",bound=bound)
    plot_diaphragms(ax=ax5,house_areas=house_areas,poly_list=valid_diaphragms_before,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,title='valid_diaphragms_before',bound=bound)
    plot_diaphragms(ax=ax6,house_areas=house_areas,poly_list=valid_diaphragms,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,title='answer_valid_diaphragms',code=1,bound=bound)

    plt.tight_layout()
    plt.show()
    plt.savefig(output_path)
    plt.close(fig)

def get_lw_ratio_from_rectangle(polygon): #获得一个矩形的长宽比
    if hasattr(polygon, "exterior"):
        coords = list(polygon.exterior.coords)
        coords = coords[:-1]  # 去掉首尾重复点
    else:
        coords = polygon
    
    xs = [p[0] for p in coords]    
    ys = [p[1] for p in coords]
    min_x,max_x,min_y,max_y=min(xs),max(xs),min(ys),max(ys)
    length,width=max_x-min_x,max_y-min_y
    return max(length,width)/min(length,width)

def fix_polygon(poly):
    poly=poly.buffer(0)
    if poly.is_valid==True:
        #print("修复成功")
        return True,poly
    else:
        print("修复失败")
        polygon=unary_union(polygon)
        if polygon.is_valid==True:
            print("再次修复成功")
            return True,poly
        else:
            print("再次修复失败")
    return False,poly

def del_list_last_element(lst,elm):
    for i in range(len(lst)-1, -1, -1):
        if lst[i] == elm:
            lst.pop(i)
    return lst

def judge_is_rectangle(polygon): #判断一个多边形是否为正矩形（与坐标轴平行）
    if hasattr(polygon, "exterior"):
        # shapely Polygon
        coords = list(polygon.exterior.coords)
        coords = coords[:-1]  # 去掉首尾重复点
    else:
        # 普通 list / tuple
        coords = polygon
    
    if len(coords) < 4: #确保点数>=4
        return False
    xs = [p[0] for p in coords]    
    ys = [p[1] for p in coords]
    min_x,max_x,min_y,max_y=min(xs),max(xs),min(ys),max(ys)
    area=((max_x-min_x)*(max_y-min_y)) #确保面积大于0
    if area<=0:
        return False
    if (min_x,min_y) not in coords or (min_x,max_y) not in coords or (max_x,min_y) not in coords or (max_x,max_y) not in coords: 
        return False       #确保矩形的四个端点都出现
    for coord in coords: #确保所有点只能出现在矩形的四条边上
        x,y=coord[0],coord[1]
        if (x==min_x or x==max_x) and y>=min_y and y<=max_y:
            continue
        elif (y==min_y or y==max_y) and x>=min_x and x<=max_x:
            continue
        else:
            return False
    return True

def judge_is_boundary(poly,segments): #检查一个polygon是否是边界多边形
    x_min,x_max,y_min,y_max=min(min(seg[1][0],seg[2][0]) for seg in segments),max(max(seg[1][0],seg[2][0]) for seg in segments),\
    min(min(seg[1][1],seg[2][1]) for seg in segments),max(max(seg[1][1],seg[2][1]) for seg in segments)
    poly_cords=list(poly.exterior.coords)
    x_min_1,x_max_1,y_min_1,y_max_1=min(cor[0] for cor in poly_cords),max(cor[0] for cor in poly_cords),\
        min(cor[1] for cor in poly_cords),max(cor[1] for cor in poly_cords)
    if x_min_1==x_min and x_max_1==x_max and y_min_1==y_min and y_max_1==y_max:
        return True
    return False

class tr_rddp:
    def __init__(self):
        1==1
    @classmethod
    def parallel(line_1:LineString, line_2:LineString):
        return get_initial_struct_based_on_response.is_parallel(line_1,line_2)
    @classmethod
    def pt_ovrlap_pt(pt_1:Point,pt_2:Point):
        return get_initial_struct_based_on_response.point_overlap_point(pt_1,pt_2)
    @classmethod
    def pt_in_line(pt:Point,line:LineString):
        return get_initial_struct_based_on_response.point_in_line(pt,line)
    @classmethod
    def line_overlap_line(line_1:LineString, line_2:LineString):
        return get_initial_struct_based_on_response.line_overlap_line(line_1,line_2)

def is_beam_generate_shearwall(wall_line,beam_line,wall_pt_1,wall_pt_2,beam_pt_1,beam_pt_2):
    if tr_rddp.line_overlap_line(wall_line,beam_line):
        if ((tr_rddp.pt_ovrlap_pt(beam_pt_1,wall_pt_1,eps=1) or tr_rddp.pt_ovrlap_pt(beam_pt_1,wall_pt_2,eps=1)) and (tr_rddp.pt_in_line(beam_pt_2,wall_line,eps=1)==False)) or \
        ((tr_rddp.pt_ovrlap_pt(beam_pt_2,wall_pt_1,eps=1) or tr_rddp.pt_ovrlap_pt(beam_pt_2,wall_pt_2,eps=1)) and (tr_rddp.pt_in_line(beam_pt_1,wall_line,eps=1)==False)):
            return True
    return False

def get_shear_walls(walls,exterior_walls,shear_walls_added,beams):
    shear_walls=[]
    #将exterior wall都添加到shearwall中
    for wall in exterior_walls: #在外围的wall都算作shear_wall
        shear_walls.append(["shearwall",wall[1],wall[2]])

    #将shear_walls_added中属于walls的子线段的部分添加到valid_shear_walls中
    for shear_wall in shear_walls_added: #去除不是wall的子线段的shearwall
        shear_walls.append(shear_wall)

    #将beams中关联到的wall添加到valid_shear_walls中
    for wall in walls: #被beam平行支撑的wall,都算作shear_wall
        wall_pt_1,wall_pt_2=wall[1],wall[2]
        for beam in beams:
            beam_pt_1,beam_pt_2=beam[1],beam[2]
            if orientation(wall_pt_1,wall_pt_2,beam_pt_1)==0 and orientation(wall_pt_1,wall_pt_2,beam_pt_2)==0:#beam和wall必须共线
                if ((beam_pt_1==wall_pt_1 or beam_pt_1==wall_pt_2) and point_is_on_line(beam_pt_2,wall_pt_1,wall_pt_2,code=0)==False) or \
                   ((beam_pt_2==wall_pt_1 or beam_pt_2==wall_pt_2) and point_is_on_line(beam_pt_1,wall_pt_1,wall_pt_2,code=0)==False):#beam和wall必须有且仅有一个交点,且beam的令一点不能在wall内部
                    #print(f"依据beam{beam}生成shear_wall{wall}")
                    shear_walls.append(["shearwall",wall[1],wall[2]])
            '''
            if is_beam_generate_shearwall(wall_line=wall_line,beam_line=beam_line,
                            wall_pt_1=wall_pt_1,wall_pt_2=wall_pt_2,beam_pt_1=beam_pt_1,beam_pt_2=beam_pt_2):
                    #print(f"依据beam{beam}生成shear_wall{wall}")
                    shear_walls.append(["shearwall",wall[1],wall[2]])
            '''
    return shear_walls

def get_valid_shear_walls(walls,exterior_walls,shear_walls_added,beams): #依据walls和shear_walls确定valid shear wall(beam不再参与)
    shear_walls=[]
    #将exterior wall都添加到shearwall中
    for wall in exterior_walls: #在外围的wall都算作shear_wall
        shear_walls.append(["shearwall",wall[1],wall[2]])

    #将shear_walls_added中属于walls的部分添加到valid_shear_walls中
    for shear_wall in shear_walls_added: #去除不是wall的子线段的shearwall
        line1=LineString([(shear_wall[1][0],shear_wall[1][1]),(shear_wall[2][0],shear_wall[2][1])])
        for wall in walls:
            line2=LineString([(wall[1][0],wall[1][1]),(wall[2][0],wall[2][1])])
            if line1.equals(line2):
                shear_walls.append(shear_wall)
                break

    #将beams中关联到的wall添加到valid_shear_walls中
    for wall in walls: #被beam平行支撑的wall,都算作shear_wall
        wall_pt_1,wall_pt_2=wall[1],wall[2]
        for beam in beams:
            beam_pt_1,beam_pt_2=beam[1],beam[2]
            if orientation(wall_pt_1,wall_pt_2,beam_pt_1)==0 and orientation(wall_pt_1,wall_pt_2,beam_pt_2)==0:#beam和wall必须共线
                if ((beam_pt_1==wall_pt_1 or beam_pt_1==wall_pt_2) and point_is_on_line(beam_pt_2,wall_pt_1,wall_pt_2,code=0)==False) or \
                   ((beam_pt_2==wall_pt_1 or beam_pt_2==wall_pt_2) and point_is_on_line(beam_pt_1,wall_pt_1,wall_pt_2,code=0)==False):#beam和wall必须有且仅有一个交点,且beam的令一点不能在wall内部
                    #print(f"依据beam{beam}生成shear_wall{wall}")
                    shear_walls.append(["shearwall",wall[1],wall[2]])
            '''
            if is_beam_generate_shearwall(wall_line=wall_line,beam_line=beam_line,
                            wall_pt_1=wall_pt_1,wall_pt_2=wall_pt_2,beam_pt_1=beam_pt_1,beam_pt_2=beam_pt_2):
                    #print(f"依据beam{beam}生成shear_wall{wall}")
                    shear_walls.append(["shearwall",wall[1],wall[2]])
            '''

    #去除shearwall中重复的或可以被其他shearwall代替的线段
    valid_shear_walls=[]
    for i,shear_wall in enumerate(shear_walls):
        line1=LineString([(shear_wall[1][0],shear_wall[1][1]),(shear_wall[2][0],shear_wall[2][1])])
        IS_VALID=True
        for j in range(0,i):
            sh_pre=shear_walls[j]
            line2=LineString([(sh_pre[1][0],sh_pre[1][1]),(sh_pre[2][0],sh_pre[2][1])])
            if line1.covered_by(line2):
                IS_VALID=False
                break
            if line1.overlaps(line2)==True:
                line1=line1.difference(line2)
        if IS_VALID==True:
            valid_shear_walls.append(shear_wall)
    #
    return valid_shear_walls

def get_diaphragm_from_segments(segments): #从segments中抽取多边形
    lines = [LineString([(seg[1][0],seg[1][1]),(seg[2][0],seg[2][1])]) for seg in segments]
    merged = unary_union(lines)
    polygons = list(polygonize(merged))

    if len(polygons)==0:
        return []

    polygons_sorted = sorted(polygons, key=lambda p: p.area, reverse=True)
    if len(polygons_sorted)>1 and judge_is_boundary(polygons_sorted[0],segments)==True:#如果切分出其他矩形，则除去外部最大矩形
        polygons_inner = polygons_sorted[1:]
    else:
        polygons_inner = polygons_sorted
    return polygons_inner 

def extract_diaphragm_from_design(beams,shear_walls):
    segments=shear_walls+beams
    diaphragms=get_diaphragm_from_segments(segments)
    return diaphragms

def get_total_area(house_areas):
    total_area=0
    HOUSE_AREA_VALID=False
    for house_area in house_areas: #提取所有area围成的面积和
        points = [(p[0], p[1]) for p in house_area[1:]]
        #保证points首位相连
        if points[0] != points[-1]:
            points.append(points[0])
        polygon = Polygon(points)
        if polygon.is_valid==False:
            IS_FIXED,polygon=fix_polygon(polygon)
            if IS_FIXED==False:
                continue
        #print(f"house_area:{polygon}")
        total_area=total_area+polygon.area
        HOUSE_AREA_VALID=True
    if HOUSE_AREA_VALID==False:
        print(f"{house_areas}无法围成多边形！")
    return total_area

def poly_overlap_opening(poly,openings): #(暂时弃用)判断一个diaphragm是否和opening有交集
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
        if poly.overlaps(polygon) or poly.equals(polygon):
            return True
    return False

def poly_overlap_and_not_cover_opening(poly,openings):#判断一个diaphragm是否和opening有重叠但又不完全包含
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
        if (poly.overlaps(polygon) or poly.equals(polygon))==True and (polygon.covered_by(poly)==False):
            return True
    return False

def poly_out_of_boundary(poly,house_areas):
    for house_area in house_areas: #提取所有area围成的面积和
        points = [(p[0], p[1]) for p in house_area[1:]]
        #保证points首位相连
        if points[0] != points[-1]:
            points.append(points[0])
        polygon = Polygon(points)
        if polygon.is_valid==False:
            IS_FIXED,polygon=fix_polygon(polygon)
            if IS_FIXED==False:
                continue
        if poly.covered_by(polygon):
            return False
    return True

def poly_contain_invalid_beams(poly,invalid_beams):
    for beam in invalid_beams:
        other_line=LineString([(beam[1][0],beam[1][1]),(beam[2][0],beam[2][1])])
        if other_line.intersects(poly) and not other_line.touches(poly):
            return True
    return False

def char_span_to_token_span(offset_mapping, char_start, char_end):
    """
    将字符区间 [char_start, char_end)
    映射到 token 区间 [token_start, token_end)
    """
    token_start = None
    token_end = None

    for i, (s, e) in enumerate(offset_mapping):
        if e <= char_start:
            continue
        if s >= char_end:
            break

        if token_start is None:
            token_start = i
        token_end = i + 1

    return token_start, token_end

def get_actions_and_index_info_index(response):
    #
    pattern = r"<(add|remove)><(beam|shearwall)>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
    results = []

    for m in re.finditer(pattern, response):
        act_type,seg_type,x1, y1, x2, y2 = m.groups() #m代表一个捕获pattern
        if act_type!="add" and act_type!="remove":
            continue
        if seg_type!="beam" and seg_type!="beams" and seg_type!="shearwall":
            continue
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        seg_char_span = m.span()
        
        results.append({
            "act_type":act_type,
            "seg_type":seg_type,
            "coords": (x1, y1, x2, y2),
            "act_char_span": seg_char_span,
        })
    return results

def filter_completion_predict(completion_predict,context): #（暂时弃用，对answer的提取尽量保持原语义进行提取）从给出的初始设计方案中预去除无效的操作
    #读取walls
    walls=[]
    house_items=get_segments_info_five_plus_two(context,POLYGON_TYPES,LINE_TYPES,POINT_TYPES)
    for seg in house_items:
        if seg[0]=="wall":
            walls.append(seg)
    #print(f"walls:{walls}")
    #print(f"completion_predict:{completion_predict}")

    #过滤completion_predict中重复的beam,并将提取到的beam记录在beams中
    pattern = r"<beams?>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
    segs=re.findall(pattern,completion_predict)
    vis_beam={}
    beams=[]
    for seg in segs:
        line=LineString([(int(seg[0]),int(seg[1])),(int(seg[2]),int(seg[3]))])
        if line in vis_beam:
            beam_text=f"<beam>({int(seg[0])},{int(seg[1])}),({int(seg[2])},{int(seg[3])})"
            completion_predict=completion_predict.replace(beam_text,"",1) #注意这里只去除一个beam,不能全部去除
            #print(f"{beam_text}已经存在，移除")
        else:
            beams.append(["beam",(int(seg[0]),int(seg[1])),(int(seg[2]),int(seg[3]))])
            vis_beam[line]=1

    #过滤completion_predict中不合法的shearwall
    pattern = r"<shearwall>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
    shs=re.findall(pattern,completion_predict)
    sh_pre_list=[]

    #利用beams初始化一个sh_pre_list名单
    for wall in walls: #被beam平行支撑的wall,都算作shear_wall
        wall_pt_1,wall_pt_2=(wall[1][0],wall[1][1]),(wall[2][0],wall[2][1])
        for beam in beams:
            beam_pt_1,beam_pt_2=(beam[1][0],beam[1][1]),(beam[2][0],beam[2][1])
            if orientation(wall_pt_1,wall_pt_2,beam_pt_1)==0 and orientation(wall_pt_1,wall_pt_2,beam_pt_2)==0:#beam和wall必须共线
                if ((beam_pt_1==wall_pt_1 or beam_pt_1==wall_pt_2) and point_is_on_line(beam_pt_2,wall_pt_1,wall_pt_2,code=0)==False) or \
                   ((beam_pt_2==wall_pt_1 or beam_pt_2==wall_pt_2) and point_is_on_line(beam_pt_1,wall_pt_1,wall_pt_2,code=0)==False):#beam和wall必须有且仅有一个交点,且beam的令一点不能在wall内部
                    #print(f"依据beam{beam}生成shear_wall{wall}")
                    sh_pre_list.append(LineString([wall_pt_1,wall_pt_2]))
   
    #基于初始的sh_pre_list判断
    for sh in shs:
        sh_action={"coords":(int(sh[0]),int(sh[1]),int(sh[2]),int(sh[3]))}
        sh_line=LineString([(sh[0],sh[1]),(sh[2],sh[3])])
        IS_VALID,ERROR_TYPE=judge_shearwall_action_is_valid(sh_action=sh_action,walls=walls,sh_pre_list=sh_pre_list)
        if IS_VALID==True:
            sh_pre_list.append(sh_line)
        else:
            sh_text=f"<shearwall>({int(sh[0])},{int(sh[1])}),({int(sh[2])},{int(sh[3])})"
            #print(f"{sh_text}不合规")
            if ERROR_TYPE==0:#如果不是wall的子线段，则全部删除
                completion_predict=completion_predict.replace(sh_text,"")
            elif ERROR_TYPE==1:#如果是覆盖错误，则只删除一条线即可
                completion_predict=completion_predict.replace(sh_text,"",1)
    return completion_predict

def sort_completion_predict(completion_predict): #保证completion_predict的排列方式始终是先beam后shearwall
    text=""
    beams=get_segments_info_five_plus_two(completion_predict,[],["beam"],[])
    shearwalls=get_segments_info_five_plus_two(completion_predict,[],["shearwall"],[])
    for beam in beams:
        text+=f"<{beam[0]}>({int(beam[1][0])},{int(beam[1][1])}),({int(beam[2][0])},{int(beam[2][1])})"
    for shearwall in shearwalls:
        text+=f"<{shearwall[0]}>({int(shearwall[1][0])},{int(shearwall[1][1])}),({int(shearwall[2][0])},{int(shearwall[2][1])})"
    return text

def process_completion_predict_based_on_response(completion_predict,response):
    action_context=completion_predict
    #处理response,将操作按原始含义添加在completion_predict
    pattern = r"<(add|remove)><(beam|shearwall)>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
    raw_actions=re.findall(pattern,response)

    for action in raw_actions:
        if action[1]=="beam" or action[1]=="beams":
            if action[0]=="add":
                add_action_text=f"<beam>({int(action[2])},{int(action[3])}),({int(action[4])},{int(action[5])})"
                action_context=action_context+add_action_text
            if action[0]=="remove":
                minus_action_text=f"<beam>({int(action[2])},{int(action[3])}),({int(action[4])},{int(action[5])})"
                if minus_action_text in action_context:
                    action_context=''.join(action_context.rsplit(minus_action_text,1)) #删除从后往前的第一个匹配项
        elif action[1]=="shearwall":
            if action[0]=="add":
                add_action_text=f"<shearwall>({int(action[2])},{int(action[3])}),({int(action[4])},{int(action[5])})"
                action_context=action_context+add_action_text
            elif action[0]=="remove":
                minus_action_text=f"<shearwall>({int(action[2])},{int(action[3])}),({int(action[4])},{int(action[5])})"
                if minus_action_text in action_context:
                    action_context=''.join(action_context.rsplit(minus_action_text,1)) #删除从后往前的第一个匹配项

    #移除重复的beam和重复或者不在操作域的shearwall
    #action_context=filter_completion_predict(action_context,context)
    action_context=sort_completion_predict(action_context)
    return action_context

def from_valid_segments_to_completion_predict(valid_beams,valid_shearwalls):
    completion_predict=""
    for beam in valid_beams:
        completion_predict+=f"<beam>({beam[1][0]},{beam[1][1]}),({beam[2][0]},{beam[2][1]})"
    for shear_wall in valid_shearwalls:
        completion_predict+=f"<shearwall>({shear_wall[1][0]},{shear_wall[1][1]}),({shear_wall[2][0]},{shear_wall[2][1]})"
    return completion_predict

def process_completion_predict_filt(context, completion_predict, response):
    #print(f"completion_predict:{completion_predict},response:{response}")
    response_processed=process_completion_predict_based_on_response(completion_predict=completion_predict,response=response)
    #print(f"response_processed:{response_processed}")

    response_processed=filter_completion_predict(context=context,completion_predict=response_processed)
    
    result=get_design_score(context,response_processed)
    
    valid_beams,valid_shearwalls=result["valid_beams"],result["shear_walls_valid"]

    completion_predict_filt=from_valid_segments_to_completion_predict(valid_beams=valid_beams,valid_shearwalls=valid_shearwalls)
    return completion_predict_filt

def get_intersection_points(l1, l2): #得到两条线的交点
    inter = l1.intersection(l2)
    if inter.is_empty:
        return []
    if inter.geom_type == "Point":
        return [(inter.x, inter.y)]
    if inter.geom_type == "MultiPoint":
        return [(p.x, p.y) for p in inter.geoms]
    return []

def action_cross_too_long(action,shear_walls):#返回一个beam的跨度长度
    pt_l,pt_r=(action['coords'][0],action['coords'][1]),(action['coords'][2],action['coords'][3])
    beam=LineString([(action['coords'][0],action['coords'][1]),(action['coords'][2],action['coords'][3])])
    wall_pts=[]
    wall_pts.append(pt_l)
    wall_pts.append(pt_r)
    for shear_wall in shear_walls:
        other_line=LineString([(shear_wall[1][0],shear_wall[1][1]),(shear_wall[2][0],shear_wall[2][1])])
        wall_pts=wall_pts+get_intersection_points(beam,other_line)
    wall_pts=sorted(wall_pts,key=lambda x:pow((x[0]-pt_l[0]),2)+pow((x[1]-pt_l[1]),2))
    for i in range(len(wall_pts)-1):
        line=LineString([(wall_pts[i][0],wall_pts[i][1]),(wall_pts[i+1][0],wall_pts[i+1][1])])
        #print(line)
        if line.length>3000:
            '''
            for shear_wall in shear_walls:
                if shear_wall[1][1]==-1133 or shear_wall[2][1]==-1133:
                    print(shear_wall)
            print(f"排序过的点集为：{wall_pts}")
            print(f"过长区域：{line}")
            '''
            return True
    return False

def action_not_depend_well(shear_walls,walls,added_beams,action): #依据shear_walls(加入该action后的shear_walls)和added_beams判断action是否得到了支撑
    is_depend_l,is_depend_r=False,False
    line_l=(action['coords'][0],action['coords'][1])
    line_r=(action['coords'][2],action['coords'][3])
    for other_beam in added_beams:
        other_l=(other_beam[1][0],other_beam[1][1])
        other_r=(other_beam[2][0],other_beam[2][1])
        l_depend,r_depend=point_is_on_line(line_l,other_l,other_r,1),point_is_on_line(line_r,other_l,other_r,1)
        if point_is_on_line(line_l,other_l,other_r,0)==True and point_is_on_line(line_r,other_l,other_r,0)==True:
            continue
        is_depend_l=is_depend_l+l_depend
        is_depend_r=is_depend_r+r_depend
    for shear_wall in shear_walls:
        seg_l=(shear_wall[1][0],shear_wall[1][1])
        seg_r=(shear_wall[2][0],shear_wall[2][1])
        l_depend,r_depend=point_is_on_line(line_l,seg_l,seg_r,0),point_is_on_line(line_r,seg_l,seg_r,0)
        if l_depend==True and r_depend==True:
            continue
        is_depend_l=is_depend_l+l_depend
        is_depend_r=is_depend_r+r_depend
    
    for wall in walls:
        seg_l=(wall[1][0],wall[1][1])
        seg_r=(wall[2][0],wall[2][1])
        l_depend,r_depend=point_is_on_line(line_l,seg_l,seg_r,0),point_is_on_line(line_r,seg_l,seg_r,0)
        if l_depend==True and r_depend==True:
            continue
        is_depend_l=is_depend_l+l_depend
        is_depend_r=is_depend_r+r_depend
    
    if not is_depend_l or not is_depend_r:
        return True
    return False

def action_cover_shear_walls(shear_walls,action): #判断新加的beam是否有和某个shear_wall重合
    beam=LineString([(action['coords'][0],action['coords'][1]),(action['coords'][2],action['coords'][3])])
    for shear_wall in shear_walls:
        other_line=LineString([shear_wall[1],shear_wall[2]])
        if beam.overlaps(other_line):
            return True
    return False

def action_intersect_opening(action,openings): #判断新加的beam是否和某个opening相交（不算相切）
    beam=LineString([(action['coords'][0],action['coords'][1]),(action['coords'][2],action['coords'][3])])
    #print(f"beam:{beam},openings:{openings}")
    for opening in openings:#遍历opening
        points = [(p[0], p[1]) for p in opening[1:]]
        #保证points首位相连
        if points[0] != points[-1]:
            points.append(points[0])
        polygon = Polygon(points)
        if polygon.is_valid==False:
            IS_FIXED,polygon=fix_polygon(polygon)
            if IS_FIXED==False:
                continue
        if beam.intersects(polygon) and not beam.touches(polygon):
            return True
    return False

def action_out_of_boundary(action,house_areas): #判断新加的beam是否越界
    beam=LineString([(action['coords'][0],action['coords'][1]),(action['coords'][2],action['coords'][3])])
    for house_area in house_areas:#遍历house_area
        points = [(p[0], p[1]) for p in house_area[1:]]
        #保证points首位相连
        if points[0] != points[-1]:
            points.append(points[0])
        polygon = Polygon(points)
        if polygon.is_valid==False:
            IS_FIXED,polygon=fix_polygon(polygon)
            if IS_FIXED==False:
                continue
        if beam.covered_by(polygon):#beam可以在polygon内部或者在polygon边界
            return False
    return True #如果beam不被任何一个house_area包含,返回出界

def action_cross_other_segments(action,lines): #判断beam与其他线是否交叉（暂时弃用）
    beam=LineString([(action['coords'][0],action['coords'][1]),(action['coords'][2],action['coords'][3])])
    for line in lines:
        other_line=LineString([(line[1][0],line[1][1]),(line[2][0],line[2][1])])
        if beam.crosses(other_line):
            return True
    return False

def action_overlap_other_segments(action,walls,beams_pre): #判断beam与其他beam或者wall是否重合
    beam=LineString([(action['coords'][0],action['coords'][1]),(action['coords'][2],action['coords'][3])])
    for line in walls:
        other_line=LineString([(line[1][0],line[1][1]),(line[2][0],line[2][1])])
        if beam.overlaps(other_line)==True or beam.equals(other_line)==True:
            return True
    for line in beams_pre:
        other_line=LineString([(line[1][0],line[1][1]),(line[2][0],line[2][1])])
        if beam.overlaps(other_line)==True or beam.equals(other_line)==True:
            return True
    return False

def get_action_direction(action):
    pt_1,pt_2=(action['coords'][0],action['coords'][1]),(action['coords'][2],action['coords'][3])
    if pt_1==pt_2:
        return 'point'
    elif pt_1[0]==pt_2[0]:
        return 'verticle'
    elif pt_1[1]==pt_2[1]:
        return 'horizontal'
    else:
        return 'diagnal'

def judge_beam_is_valid(beam,walls,house_areas,shear_walls,beams,beams_pre,openings):
    #将beam转化为action
    action={"act_type":None,"coords": (int(beam[1][0]), int(beam[1][1]), int(beam[2][0]), int(beam[2][1])),
            "act_char_span": None,"act_token_span": None}
    action_direction=get_action_direction(action=action)
    #判断beam是否合法
    if action_out_of_boundary(action=action,house_areas=house_areas)==True: #判断beam是否在边界内部
       #print(f"{beam} out of boundary!")
       return False,f"{beam} out of boundary!",0
    elif action_not_depend_well(shear_walls=shear_walls,walls=walls,added_beams=beams,action=action)==True: #判断beam是否支撑合理
        #print(f"{beam} not depend well!")
        return False,f"{beam} not depend well!",1
    elif action_overlap_other_segments(action=action,walls=walls,beams_pre=beams_pre)==True:
        #print(f"{beam} overlap other segments!")
        return False,f"{beam} overlap walls or beams!",2
    elif action_direction=='point' or (action_direction!='verticle' and action_direction!='horizontal'):#判断beam方向是否合理
        #print(f"{beam} has wrong direction!")
        return False,f"{beam} has wrong direction!",3
    elif action_cross_too_long(action=action,shear_walls=shear_walls)==True: #判断beam是否跨度过大
        #print(f"{beam} cross too long!")
        return False,f"{beam} cross too long!",4
    elif action_cover_shear_walls(shear_walls=shear_walls,action=action)==True: #判断beam有无覆盖到shear_wall
        #print(f"{beam} cover shear wall!")
        return False,f"{beam} cover shear wall!",5
    elif action_intersect_opening(action=action,openings=openings)==True: #判断beam是否与opening相交
        #print(f"{beam} intersect opening!")
        return False,f"{beam} intersect opening!",6
    #print(f"{beam} is ok!")
    return True,None,None

def judge_shearwall_action_is_valid(sh_action,walls,sh_pre_list): #一个有效的shearwall,必须与某一个wall是covered by的关系
    line1=LineString([(sh_action['coords'][0],sh_action['coords'][1]),(sh_action['coords'][2],sh_action['coords'][3])])
    IS_COVERED=False
    for wall in walls: #判断这个shearwall是一个wall的子线段
        line2=LineString([(wall[1][0],wall[1][1]),(wall[2][0],wall[2][1])])
        if line1.equals(line2):
            IS_COVERED=True
            break
    if IS_COVERED==False:
        #print(f"{line1}不属于任何一个wall的子线段")
        return False,0

    for line2 in sh_pre_list: #判断这个action是否不被之前的action完全包含
        if line1.covered_by(line2):
            #print(f"{line1}被某一个已添加的shearwall完全覆盖")
            return False,1
        if line1.overlaps(line2)==True:
            #print(f"{line1}与{line2}重合")
            line1=line1.difference(line2)
            #print(f"减去重合后的结果: {line1}")
    return True,None

def get_design_score(context,answer):
    #print("here")
    #提取房屋基本元素和设计处理完成的beam
    house_items=get_segments_info_five_plus_two(context,POLYGON_TYPES,LINE_TYPES,POINT_TYPES)
    predict_structures=get_segments_info_five_plus_two(answer,POLYGON_TYPES,LINE_TYPES,POINT_TYPES)
    exterior_walls,walls,beams,openings,house_areas,shear_walls_added=[],[],[],[],[],[]
    for seg in house_items:
        if seg[0]=="exterior_wall":
            exterior_walls.append(seg)
        elif seg[0]=="wall":
            walls.append(seg)
        elif seg[0]=='opening':
            openings.append(seg)
        elif seg[0]=="inoutbox":
            house_areas.append(seg)
    for seg in predict_structures:
        if seg[0]=="beam":
            beams.append(seg)
        if seg[0]=="shearwall":
            shear_walls_added.append(seg)

    #beams.append(["beam",(2344,1737),(2344,2127)]) #测试

    #print(f"walls:{walls}")
    #print(f"openings:{openings}")
    #print(f"边界是：{house_areas}")

    #得到房屋总面积和边界是否有斜线
    total_area=get_total_area(house_areas=house_areas)
    #print(f"total_area:{total_area}")

    #去除非法beam,得到valid_beams
    shear_walls=get_shear_walls(walls=walls,exterior_walls=exterior_walls,shear_walls_added=shear_walls_added,beams=beams)
    shear_walls_valid=get_valid_shear_walls(walls=walls,exterior_walls=exterior_walls,shear_walls_added=shear_walls_added,beams=beams)
    #print(f"valid_shear_walls:{valid_shear_walls}")

    beams_valid_pre,beams_valid=beams,beams
    shear_walls_valid_pre=shear_walls_valid
    invalid_beams=[]

    OPT_ROUND=0
    while 1==1:
        beam_list=[]
        beam_pre_list=[]
        for beam in beams_valid:
            BEAM_IS_VALID,_,_=judge_beam_is_valid(beam=beam,walls=walls,house_areas=house_areas,shear_walls=shear_walls_valid,
                                                  beams=beams_valid,beams_pre=beam_pre_list,openings=openings)
            if BEAM_IS_VALID==True:
                beam_list.append(beam)
            elif BEAM_IS_VALID==False:
                #print(f"beam:{beam} is invalid")
                invalid_beams.append(beam)
                #实验代码--------------（实验后删除）
                #beam_list.append(beam)
                #--------------------------------

            beam_pre_list.append(beam)
        beams_valid=beam_list
        shear_walls_valid=get_valid_shear_walls(walls=walls,exterior_walls=exterior_walls,shear_walls_added=shear_walls_added,beams=beams_valid)
        if beams_valid_pre==beams_valid and shear_walls_valid_pre==shear_walls_valid:
            break
        OPT_ROUND=OPT_ROUND+1
        beams_valid_pre=beams_valid
        shear_walls_valid_pre=shear_walls_valid

    #依据shear_walls_valid和beams_valid提取diaphragms
    diaphragms=extract_diaphragm_from_design(shear_walls=shear_walls_valid,beams=beams_valid)

    #从diaphragms中提取valid_diaphragms
    valid_diaphragm_area=0
    valid_diaphragms=[]
    
    for poly in diaphragms:
        #if poly.area>4e7 or poly.area<4e4: #diaphragm不能过小或过大
        #    print(f"{list(poly.exterior.coords)}{poly.area}面积不符！")
        #   continue
        if judge_is_rectangle(poly)==False: #不允许划分出的diaphragm是非矩形
            #print(f"{list(poly.exterior.coords)}非矩形！")
            continue
        if get_lw_ratio_from_rectangle(poly)>7: #长宽比不能超过5
            #print(f"{list(poly.exterior.coords)}长宽比大于7！")
            continue
        if poly_out_of_boundary(poly=poly,house_areas=house_areas)==True:
            #print(f"{list(poly.exterior.coords)}超出边界！")
            continue
        if poly_overlap_and_not_cover_opening(poly=poly,openings=openings)==True:
            #print(f"{list(poly.exterior.coords)}与opening重叠但不包含！")
            continue
        if poly_contain_invalid_beams(poly=poly,invalid_beams=invalid_beams)==True:
            #print(f"{list(poly.exterior.coords)}包含了非法的beams！")
            continue

        valid_diaphragm_area=valid_diaphragm_area+poly.area
        valid_diaphragms.append(poly)
        #print(f"有效面积：{poly.area},有效面积区域:{list(poly.exterior.coords)}")
    area_ratio=valid_diaphragm_area/total_area
    house_score=2*area_ratio-1
    #print(f"有效区域比列为{area_ratio},房屋分数为{house_score}")
    #print(f"valid_beams:{beams_valid},openings:{openings}")

    #构造result并返回
    result={
        "house_score":house_score,"area_ratio":area_ratio,"beams":beams,"valid_beams":beams_valid,"diaphragms":diaphragms,
        "valid_diaphragms":valid_diaphragms,"shear_walls":shear_walls,"shear_walls_valid":shear_walls_valid,
        "walls":walls,"exterior_walls":exterior_walls,"house_areas":house_areas,"openings":openings,"OPT_ROUND":OPT_ROUND,
    }
    return result

def get_diaphragm_result(context,answer):
    return get_design_score(context,answer)

def get_valid_beams_and_shearwalls(beams_added,shear_walls_added,walls,exterior_walls,house_areas,openings):
    shear_walls_valid=get_valid_shear_walls(walls=walls,exterior_walls=exterior_walls,shear_walls_added=shear_walls_added,beams=beams_added)
    #print(f"valid_shear_walls:{valid_shear_walls}")

    beams_valid_pre,beams_valid=beams_added,beams_added
    shear_walls_valid_pre=shear_walls_valid
    invalid_beams=[]

    while 1==1:
        beam_list=[]
        beam_pre_list=[]
        for beam in beams_valid:
            BEAM_IS_VALID,_,_=judge_beam_is_valid(beam=beam,walls=walls,house_areas=house_areas,shear_walls=shear_walls_valid,
                                                  beams=beams_valid,beams_pre=beam_pre_list,openings=openings)
            if BEAM_IS_VALID==True:
                beam_list.append(beam)
            elif BEAM_IS_VALID==False:
                invalid_beams.append(beam)
            beam_pre_list.append(beam)
        beams_valid=beam_list
        shear_walls_valid=get_valid_shear_walls(walls=walls,exterior_walls=exterior_walls,shear_walls_added=shear_walls_added,beams=beams_valid)
        if beams_valid_pre==beams_valid and shear_walls_valid_pre==shear_walls_valid:
            break
        beams_valid_pre=beams_valid
        shear_walls_valid_pre=shear_walls_valid
    
    return beams_valid,shear_walls_valid

'''
def trunc_response_and_calculate_score(context,completion_predict,response,trunc_result,tokenizer):
    SCORE_RANGE=1
    INVALID_PENALTY,ILLEGAL_PENALTY=0.05,0.2 #var
    MAX_TOKEN_NUM=300 #var
    base_result=get_design_score(context,completion_predict)
    base_house_score=2*SCORE_RANGE*base_result["area_ratio"]-SCORE_RANGE
    invalid_process_num,illegal_num=0,0

    beams_added=get_segments_info_five_plus_two(completion_predict,[],['beam'],[])
    shear_walls_added=get_segments_info_five_plus_two(completion_predict,[],['shearwall'],[])

    actions=get_actions_and_index_info_index(response)
    if len(actions)==0:
        return trunc_result
    for action_index,action in enumerate(actions):
        #print(f"s1:{response[:action['act_char_span'][1]]},s2:{response[action['act_char_span'][1]:]}")
        response_after_action=response[:action['act_char_span'][1]]
        answer_after_action=process_completion_predict_based_on_response(completion_predict,response_after_action)
        result=get_design_score(context,answer_after_action)
        house_score=2*SCORE_RANGE*result["area_ratio"]-SCORE_RANGE
        act_type=action['act_type']
        seg_type=action['seg_type']
        
        #判断要加入的beam是否合法
        if seg_type=='beam':
            beam_seg=['beam',(int(action['coords'][0]),int(action['coords'][1])),(int(action['coords'][2]),int(action['coords'][3]))]
            if act_type=='add' and beam_seg in beams_added: #添加一条已经存在的beam
                invalid_process_num=invalid_process_num+1
                #print(f"添加的beam{beam_seg}已经存在")
            elif act_type=='remove' and beam_seg not in beams_added: #删除一条不存在的beam
                invalid_process_num=invalid_process_num+1
                #print(f"要删除的beam{beam_seg}不存在")
            else:  #依据加减beam的合法性得到score_val
                BEAM_IS_VALID,_,_=judge_beam_is_valid(beam=beam_seg,walls=result["walls"],house_areas=result['house_areas'],
                shear_walls=result["shear_walls_valid"],
                beams=result["valid_beams"],beams_pre=beams_added,openings=result['openings'])

                if act_type=="add":
                    if BEAM_IS_VALID==False: #该beam有效，但是添加非法
                        #print(f"要添加的beam{beam_seg}非法")
                        illegal_num=illegal_num+1
                    beams_added.append(beam_seg)
                elif act_type=="remove":
                    #if BEAM_IS_VALID==False: #该beam有效，移除非法
                        #print("移除了一条非法的beam")
                    #    score_val=score_val+0.5
                    beams_added=del_list_last_element(lst=beams_added,elm=beam_seg)
        
        #判断要加入的shearwall是否合法
        elif seg_type=='shearwall':
            sh_seg=['shearwall',(action['coords'][0],action['coords'][1]),(action['coords'][2],action['coords'][3])]
            sh_pre_list=get_valid_shear_walls(walls=result["walls"],exterior_walls=result["exterior_walls"],shear_walls_added=shear_walls_added,beams=result["valid_beams"])
            SH_IS_VALID=judge_shearwall_action_is_valid(sh_action=action,walls=result["walls"],
            sh_pre_list=[LineString([(x[1][0],x[1][1]),(x[2][0],x[2][1])]) for x in sh_pre_list])
            
            if act_type=="add":
                if sh_seg in shear_walls_added:
                    #print(f"要添加的shearwall{sh_seg}已经存在")
                    invalid_process_num=invalid_process_num+1
                elif SH_IS_VALID==False:
                    #print(f"要添加的shearwall{sh_seg}非法")
                    illegal_num=illegal_num+1
                shear_walls_added.append(sh_seg)
            elif act_type=="remove" and sh_seg not in shear_walls_added:
                #print(f"要移除的shearwall{sh_seg}不存在")
                invalid_process_num=invalid_process_num+1
            else:
                if act_type=="add":
                    shear_walls_added.append(sh_seg)
                elif act_type=="remove":
                    shear_walls_added=del_list_last_element(lst=shear_walls_added,elm=sh_seg)

        #对score_val进行[-1,1]区间的clamp
        delta=house_score-base_house_score
        #if delta!=0:
        #    print(f"delta:{delta}不为0！")
        if abs(delta)>1e-4 or action_index==len(actions)-1:
            #print(f"completion_predict:{completion_predict},response_after_action:{response_after_action}")
            len_ratio=len(response_after_action)/MAX_TOKEN_NUM
            if delta!=0:
                final_score=2*delta/len_ratio
                final_score=final_score-INVALID_PENALTY*invalid_process_num-ILLEGAL_PENALTY*illegal_num
            else:
                final_score=-0.5*len_ratio-INVALID_PENALTY*invalid_process_num-ILLEGAL_PENALTY*illegal_num
            final_score=max(-1*SCORE_RANGE,min(SCORE_RANGE,final_score))
            
            answer_after_action=process_completion_predict_filt(context=context,completion_predict=completion_predict,
                                        response=response_after_action)
            trunc_result["prompt_list"].append(construct_prompt(context,completion_predict))
            trunc_result["response_list"].append(response_after_action)
            trunc_result["response_encoded_list"].append(tokenizer(response_after_action, return_offsets_mapping=False))
            trunc_result["answer_list"].append(answer_after_action)
            trunc_result["score_list"].append(final_score)
            trunc_result["delta_list"].append(delta)
            
            trunc_result=\
                trunc_response_and_calculate_score(context=context,completion_predict=answer_after_action,
                response=response[action['act_char_span'][1]:],
                trunc_result=trunc_result,tokenizer=tokenizer)
            break
    return trunc_result
'''
'''
def get_score_from_cmp_and_response(context,completion_predict,response,tokenizer):
    SCORE_RANGE=1
    INVALID_PENALTY,ILLEGAL_PENALTY=0.05,0.05 #var
    MAX_TOKEN_NUM=200 #var
    base_result=get_design_score(context,completion_predict)
    base_house_score=2*SCORE_RANGE*base_result["area_ratio"]-SCORE_RANGE
    invalid_process_num,illegal_num=0,0

    beams_added=get_segments_info_five_plus_two(completion_predict,[],['beam'],[])
    shear_walls_added=get_segments_info_five_plus_two(completion_predict,[],['shearwall'],[])

    actions=get_actions_and_index_info_index(response)
    for action in actions:
        response_after_action=response[:action['act_char_span'][1]]
        answer_after_action=process_completion_predict_based_on_response(completion_predict,response_after_action)
        result=get_design_score(context,answer_after_action)
        act_type=action['act_type']
        seg_type=action['seg_type']
        
        #判断要加入的beam是否合法
        if seg_type=='beam':
            beam_seg=['beam',(int(action['coords'][0]),int(action['coords'][1])),(int(action['coords'][2]),int(action['coords'][3]))]
            if act_type=='add' and beam_seg in beams_added: #添加一条已经存在的beam
                invalid_process_num=invalid_process_num+1
            elif act_type=='remove' and beam_seg not in beams_added: #删除一条不存在的beam
                invalid_process_num=invalid_process_num+1
            else:  #依据加减beam的合法性得到score_val
                BEAM_IS_VALID,_,_=judge_beam_is_valid(beam=beam_seg,walls=result["walls"],house_areas=result['house_areas'],
                shear_walls=result["shear_walls_valid"],
                beams=result["valid_beams"],beams_pre=beams_added,openings=result['openings'])

                if act_type=="add":
                    if BEAM_IS_VALID==False: #该beam有效，但是添加非法
                        illegal_num=illegal_num+1
                    beams_added.append(beam_seg)
                elif act_type=="remove":
                    beams_added=del_list_last_element(lst=beams_added,elm=beam_seg)
        
        #判断要加入的shearwall是否合法
        elif seg_type=='shearwall':
            sh_seg=['shearwall',(action['coords'][0],action['coords'][1]),(action['coords'][2],action['coords'][3])]
            sh_pre_list=get_valid_shear_walls(walls=result["walls"],exterior_walls=result["exterior_walls"],shear_walls_added=shear_walls_added,beams=result["valid_beams"])
            SH_IS_VALID=judge_shearwall_action_is_valid(sh_action=action,walls=result["walls"],
            sh_pre_list=[LineString([(x[1][0],x[1][1]),(x[2][0],x[2][1])]) for x in sh_pre_list])
            
            if act_type=="add":
                if sh_seg in shear_walls_added:
                    invalid_process_num=invalid_process_num+1
                elif SH_IS_VALID==False:
                    illegal_num=illegal_num+1
                shear_walls_added.append(sh_seg)
            elif act_type=="remove" and sh_seg not in shear_walls_added:
                invalid_process_num=invalid_process_num+1
            else:
                if act_type=="add":
                    shear_walls_added.append(sh_seg)
                elif act_type=="remove":
                    shear_walls_added=del_list_last_element(lst=shear_walls_added,elm=sh_seg)
    
    answer=process_completion_predict_based_on_response(completion_predict,response)
    result=get_design_score(context,answer)
    house_score=2*SCORE_RANGE*result["area_ratio"]-SCORE_RANGE
    delta=house_score-base_house_score
    len_response=len(tokenizer(response).input_ids)
    len_ratio=len_response/MAX_TOKEN_NUM
    if delta!=0:
        final_score=2*delta/len_ratio
        final_score=final_score-INVALID_PENALTY*invalid_process_num-ILLEGAL_PENALTY*illegal_num
    else:
        final_score=-0.5*len_ratio-INVALID_PENALTY*invalid_process_num-ILLEGAL_PENALTY*illegal_num
    final_score=max(-1*SCORE_RANGE,min(SCORE_RANGE,final_score))
    
    return final_score,delta,invalid_process_num,illegal_num

    '''

'''
'''
'''
if __name__=="__main__":#带response版测试
    mode_path="/home/jiuxing_li/five_plus_two_optimization/train_model_new/GRPO/base_model/ernie_base_model_7"
    tokenizer=AutoTokenizer.from_pretrained(mode_path)

    INPUT_JSON = "train_json_data/five_plus_two_train_jsonl_data/design_3.27/base_model_train/train_set_100_actionized_1.jsonl"#var
    #OUTPUT_PATH = "five_plus_two_optimization/five_plus_two_test/five_plus_two_predict_jsonl/data_2.4/data_2.4_random_0.6_qwen_1.5b_diaphragm_check.jsonl"
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        for f_index, line in enumerate(f):
            if f_index!=163:
                continue
            print(f_index)
            record = json.loads(line)
            house,floor,bound,context,completion_predict,response = record["house"],record["floor"],record['bound'],record["context"],\
            record["completion_predict"],record["response"]
            completion_predict=filter_completion_predict(completion_predict,context)
            answer=process_completion_predict_based_on_response(completion_predict,response)
            #print(answer)

            #得到房屋整体设计分数
            result=get_design_score(context,answer)
            result_before=get_design_score(context,completion_predict)
            print(f"house_score_before:{result_before['house_score']},house_score_after:{result['house_score']}")
            if abs(result['house_score']-result_before['house_score'])>1e-4:
                print("house_score有变化！")
            else:
                continue

            #为每个token进行打分
            tokenizer=AutoTokenizer.from_pretrained(mode_path)
            trunc_result=trunc_response_and_calculate_score(context=context,completion_predict=completion_predict,
            response=response,trunc_result={"prompt_list": [], "response_list": [], "response_encoded_list": [], "answer_list": [], "score_list": [],"delta_list": []},tokenizer=tokenizer)

            if len(trunc_result['prompt_list'])<3:
                continue
            
            print(f"prompt_list:{trunc_result['prompt_list'][0]},answer_list:{trunc_result['answer_list'][0]},\
                response_list:{trunc_result['response_list'][0]},score_list:{trunc_result['score_list'][0]},delta_list:{trunc_result['delta_list'][0]}")

            #print(f"shear walls:{shear_walls_return}")
            #print(f"shear walls valid:{shear_walls_valid_return}")
            #print(f"beams valid: {valid_beams_return}")
            #print(f"walls:{result['walls']}")
            output_pic_dir=f'five_plus_two_optimization/five_plus_two_test/test_pic/data_3.27/trunc_reward_design_check/sample{f_index+1}_{house}_{floor}_{result["area_ratio"]}.png'
            visualize_six_axes(context=context,house_areas=result["house_areas"],cp_beams=result_before["beams"],
                cp_shearwalls=result_before["shear_walls"],beams=result["beams"],shearwalls=result["shear_walls"],valid_beams=result["valid_beams"],
                valid_shear_walls=result["shear_walls_valid"],diaphragms=[list(poly.exterior.coords) for poly in result['diaphragms']],
                valid_diaphragms=[list(poly.exterior.coords) for poly in result['valid_diaphragms']],bound=bound,output_path=output_pic_dir,
                valid_diaphragms_before=[list(poly.exterior.coords) for poly in result_before['valid_diaphragms']])
            for i in range(len(trunc_result['prompt_list'])):
                #print(f"Visualizing {i+1},{house},{floor}")
                if i==0:
                    result,result_before=get_design_score(context,trunc_result['answer_list'][i]),get_design_score(context,completion_predict)
                else:
                    result,result_before=get_design_score(context,trunc_result['answer_list'][i]),get_design_score(context,trunc_result['answer_list'][i-1])
                area_ratio=result["area_ratio"]
                delta=trunc_result['delta_list'][i]
                score=trunc_result['score_list'][i]
                output_pic_dir=f"five_plus_two_optimization/five_plus_two_test/test_pic/data_3.27/trunc_reward_design_check/sample{f_index+1}_{house}_{floor}_{area_ratio}_{i+1}_{delta}_{score}.png"
                visualize_six_axes(context=context,house_areas=result["house_areas"],cp_beams=result_before["beams"],
                cp_shearwalls=result_before["shear_walls"],beams=result["beams"],shearwalls=result["shear_walls"],valid_beams=result["valid_beams"],
                valid_shear_walls=result["shear_walls_valid"],diaphragms=[list(poly.exterior.coords) for poly in result['diaphragms']],
                valid_diaphragms=[list(poly.exterior.coords) for poly in result['valid_diaphragms']],bound=bound,output_path=output_pic_dir,
                valid_diaphragms_before=[list(poly.exterior.coords) for poly in result_before['valid_diaphragms']])
            #break
            #if f_index > 100:
            #    break
'''