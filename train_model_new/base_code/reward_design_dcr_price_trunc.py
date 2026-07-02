import json
import sys
sys.path.append("/home/jiuxing_li")
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/five_plus_two_test")
sys.path.append("/home/jiuxing_li/Genia-structural-copilot-saas-server")
sys.path.append("/home/jiuxing_li/Genia-structural-copilot-saas-server/analysis_libs")
sys.path.append("/home/jiuxing_li/Genia-structural-copilot-saas-server/generate_libs")
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/five_plus_two_test")
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/train_model_new/train_base_model")
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/five_plus_two_data_process")
sys.path.append("/home/jiuxing_li/five_plus_two_optimization/train_model_new/base_code")
from standard_function import get_segments_info_five_plus_two
from five_plus_two_visualisation_single_floor import plot_diaphragms,plot_segments
import matplotlib.pyplot as plt
from five_plus_two_visualisation_single_floor import plot_segments_with_force_1
from struct_generate import restore_struct_solutions_from_AI,get_analysis_items_for_ai_designs
from struct_manager import get_house_analysis_items_score
from struct_analysis_v2 import structural_analysis
from generate_analysis_result import save_check_html
from transformers import AutoTokenizer
from shapely.geometry import LineString,Point,box
from typing import TypeAlias,List
from dataclasses import dataclass
import math
from reward_design_trunc import get_diaphragm_result,process_completion_predict_filt,point_is_on_line,\
    del_list_last_element,process_completion_predict_based_on_response,get_actions_and_index_info_index,\
    judge_shearwall_action_is_valid,Polygon,fix_polygon
import random

Group_ASD_Strength_wildcard_list = [
    {"dead": 1.0, "live_roof": 1.0, "live_floor": 1.0, "live":1.0, "snow": 0, "rain": 0, "E": 0, "W": 0},
    {"dead": 1.0, "live_roof": 0, "live_floor": 1.0, "live":1.0, "snow": 1.0, "rain": 0, "E": 0, "W": 0},
    {"dead": 1.0, "live_roof": 0, "live_floor": 1.0, "live":1.0, "snow": 0, "rain": 1.0, "E": 0, "W": 0},
    {"dead": 1.0, "live_roof": 0, "live_floor": 1.0, "live":1.0, "snow": 1.0, "rain": 0, "E": round(1/1.4, 2), "W": 0},
    {"dead": 0.9, "live_roof": 0, "live_floor": 0, "live":1.0, "snow": 0, "rain": 0, "E": round(1/1.4, 2), "W": 0},
    {"dead": 1.0, "live_roof": 0, "live_floor": 1.0, "live":1.0, "snow": 0, "rain": 0, "E": 0, "W": 0.6},
    {"dead": 1.0, "live_roof": 0, "live_floor": 1.0, "live":1.0, "snow": 0.5, "rain": 0, "E": 0, "W": 0.6},
    {"dead": 1.0, "live_roof": 0, "live_floor": 1.0, "live":1.0, "snow": 1.0, "rain": 0, "E": 0, "W": 0.3},
]

import re
POLYGON_TYPES = ["opening","inoutbox"]
LINE_TYPES = ["wall","beam","shearwall","exterior_wall"]
POINT_TYPES = []
mode_path="five_plus_two_optimization/train_model_new/GRPO/base_model/ernie_base_model_7"
HOUSE_ITEM_FOLDER_PATH="/home/jiuxing_li/house_data/noopening_test_detail"

POINT_KEY: TypeAlias = tuple[int, int]
LINE_KEY: TypeAlias = tuple[tuple[int, int], ...]

@dataclass
class POST_ITEM:
    info: str
    force: float

@dataclass
class LINELOAD_ITEM:
    info: str
    force:float

CHECK_CODE=['']

def visualize_six_axes(context,completion_predict,answer,answer_shear_walls,house_areas,answer_shear_walls_valid,
    answer_valid_beams,answer_diaphragms,answer_valid_diaphragms,bound,output_path):
    #print(f"valid_diaphragms:{valid_diaphragms_return}")
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    ax1, ax2, ax3 = axes[0] #初始结构，初始结构+初始方案(含invalid beams),初始结构+设计方案(含invalid beams)
    ax4, ax5, ax6 = axes[1] #初始结构+设计方案(只有valid bemas),初始结构+设计方案(diaphragms,只含valid_beams),初始结构+设计方案(valid_diaphragms,只含valid_beams)
    initial_structures=get_segments_info_five_plus_two(context,POLYGON_TYPES,LINE_TYPES,POINT_TYPES)
    completion_predict_beams=get_segments_info_five_plus_two(completion_predict,[],['beam'],[])
    answer_predict_beams=get_segments_info_five_plus_two(answer,[],['beam'],[])
    #print(answer_predict_beams+shear_walls_answer)
    plot_segments(ax1,initial_structures,POLYGON_TYPES,LINE_TYPES,POINT_TYPES,cut_line_list=None,title="IS",bound=bound)
    plot_segments(ax2,initial_structures,POLYGON_TYPES,["wall","beam","exterior_wall"],POINT_TYPES,cut_line_list=completion_predict_beams,title="IS+CP",bound=bound)
    plot_segments(ax3,initial_structures,POLYGON_TYPES,["wall","beam","exterior_wall","shearwall"],POINT_TYPES,cut_line_list=answer_predict_beams+answer_shear_walls,title="IS+answer_IV",bound=bound)
    plot_segments(ax4,initial_structures,POLYGON_TYPES,["wall","beam","exterior_wall","shearwall"],POINT_TYPES,cut_line_list=answer_valid_beams+answer_shear_walls_valid,title="IS+answer_V",bound=bound)
    plot_diaphragms(ax=ax5,house_areas=house_areas,poly_list=answer_diaphragms,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,title='answer_diaphragms',bound=bound)
    plot_diaphragms(ax=ax6,house_areas=house_areas,poly_list=answer_valid_diaphragms,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,title='answer_valid_diaphragms',code=1,bound=bound)

    plt.tight_layout()
    plt.show()
    plt.savefig(output_path)
    plt.close(fig)

def visualize_force(pre_context,pre_answer,context,answer,bound,output_path,house_areas,pre_post_data=None,pre_lineload_data=None,
            post_data=None,lineload_data=None,unsolved_lineload_list=None,unsolved_post_list=None):
    print(f"visualize_bound:{bound}")
    fig, axes = plt.subplots(2,3,figsize=(24, 16))
    ax1,ax2,ax3=axes[0]
    ax4,ax5,ax6=axes[1]
    
    if pre_post_data!=None:
        pre_post_list=[]
        for pre_post_key in pre_post_data.keys():
            pre_post_item=pre_post_data[pre_post_key]
            pre_post_list.append((pre_post_key[0],pre_post_key[1],pre_post_item.force))
    else:
        pre_post_list=None
    
    if pre_lineload_data!=None:
        pre_lineload_list=[]
        for pre_lineload_key in pre_lineload_data.keys():
            pre_lineload_item=pre_lineload_data[pre_lineload_key]
            pre_lineload_list.append((pre_lineload_key[0],pre_lineload_key[1],pre_lineload_item.force))
    else:
        pre_lineload_list=None
    
    if post_data!=None:
        post_list=[]
        for post_key in post_data.keys():
            post_item=post_data[post_key]
            post_list.append((post_key[0],post_key[1],post_item.force))
    else:
        post_list=None
    
    if lineload_data!=None:
        lineload_list=[]
        for lineload_key in lineload_data.keys():
            lineload_item=lineload_data[lineload_key]
            lineload_list.append((lineload_key[0],lineload_key[1],lineload_item.force))
    else:
        lineload_list=None
    
    answer_diaphragm=get_diaphragm_result(context=context,answer=answer)["valid_diaphragms"]
    answer_diaphragms=[list(poly.exterior.coords) for poly in answer_diaphragm]

    initial_structures=get_segments_info_five_plus_two(context,POLYGON_TYPES,LINE_TYPES,POINT_TYPES)
    initial_structures_pre=get_segments_info_five_plus_two(pre_context,POLYGON_TYPES,LINE_TYPES,POINT_TYPES)
    plot_segments_with_force_1(ax=ax1,segment_list=initial_structures_pre,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,
                POINT_TYPES=POINT_TYPES,cut_line_list=get_segments_info_five_plus_two(pre_answer,[],["beam","shearwall"],[]),
                title="Initial Strcture Pre",bound=bound)
    plot_segments_with_force_1(ax=ax2,segment_list=initial_structures,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,
                cut_line_list=get_segments_info_five_plus_two(answer,[],["beam","shearwall"],[]),
                title="Initial Strcture",bound=bound)
    plot_segments_with_force_1(ax=ax3,segment_list=initial_structures,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,
                cut_line_list=get_segments_info_five_plus_two(answer,[],["beam","shearwall"],[]),
                title="Initial Strcture+Pre Force",bound=bound,pre_post_list=pre_post_list,pre_lineload_list=pre_lineload_list)
    plot_segments_with_force_1(ax=ax4,segment_list=initial_structures,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,
                cut_line_list=get_segments_info_five_plus_two(answer,[],["beam","shearwall"],[]),
                title="Initial Strcture+Force",bound=bound,pre_post_list=post_list,pre_lineload_list=lineload_list,
                unsolved_lineload_list=unsolved_lineload_list,unsolved_post_list=unsolved_post_list)
    plot_diaphragms(ax=ax5,house_areas=house_areas,poly_list=answer_diaphragms,
                    POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,title='answer_diaphragms',bound=bound)

    plt.tight_layout()
    plt.show()
    plt.savefig(output_path)
    plt.close(fig)

def visualize_force_1(context,completion_predict,answer,pre_context,pre_answer,bound,output_path,pre_post_text,pre_lineload_text,base_diaphragm,diaphragm,house_areas):
    fig, axes = plt.subplots(2,3,figsize=(24, 16))
    ax1,ax2,ax3=axes[0]
    ax4,ax5,ax6=axes[1]
    
    pattern_lineload = r"<LINELOAD>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),WEIGHT_\s*(-?\d+)\s*"
    pattern_post = r"<POST>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),WEIGHT_\s*(-?\d+)\s*"
    pre_post_list=[]
    pre_post_data=re.findall(pattern_post, pre_post_text)
    for x1,y1,force in pre_post_data:
        pre_post_list.append((int(x1),int(y1),float(force)))
    
    pre_lineload_list=[]
    pre_lineload_data=re.findall(pattern_lineload, pre_lineload_text)
    for x1,y1,x2,y2,force in pre_lineload_data:
        pre_lineload_list.append(((int(x1),int(y1)),(int(x2),int(y2)),float(force)))
    
    #print(f"pre_post_list:{pre_post_list},pre_lineload_list:{pre_lineload_list}")
    #print(f"visualize_bound:{bound}")
    initial_structures=get_segments_info_five_plus_two(context,POLYGON_TYPES,LINE_TYPES,POINT_TYPES)
    if pre_context!=None and pre_answer!=None:
        initial_structures_pre=get_segments_info_five_plus_two(pre_context,POLYGON_TYPES,LINE_TYPES,POINT_TYPES)
        plot_segments_with_force_1(ax=ax1,segment_list=initial_structures_pre,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,
                    POINT_TYPES=POINT_TYPES,cut_line_list=get_segments_info_five_plus_two(pre_answer,[],["beam","shearwall"],[]),
                    title="Initial Strcture Pre",bound=bound)
    else:
        plot_segments_with_force_1(ax=ax1,segment_list=[],POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,
                    POINT_TYPES=POINT_TYPES,cut_line_list=[],
                    title="Initial Strcture Pre",bound=bound)
    plot_segments_with_force_1(ax=ax2,segment_list=initial_structures,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,
                cut_line_list=get_segments_info_five_plus_two(completion_predict,[],["beam","shearwall"],[]),
                title="Initial Strcture (cmp)",bound=bound,pre_post_list=pre_post_list,pre_lineload_list=pre_lineload_list)
    plot_segments_with_force_1(ax=ax3,segment_list=initial_structures,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,
                cut_line_list=get_segments_info_five_plus_two(answer,[],["beam","shearwall"],[]),
                title="Initial Strcture (answer)",bound=bound)
    plot_segments_with_force_1(ax=ax4,segment_list=initial_structures,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,
                cut_line_list=get_segments_info_five_plus_two(answer,[],["beam","shearwall"],[]),
                title="Initial Strcture (answer)+Force",bound=bound,pre_post_list=pre_post_list,pre_lineload_list=pre_lineload_list)
    
    plot_diaphragms(ax=ax5,house_areas=house_areas,poly_list=base_diaphragm,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,title='base_diaphragms',bound=bound)
    plot_diaphragms(ax=ax6,house_areas=house_areas,poly_list=diaphragm,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,title='diaphragms',code=1,bound=bound)

    plt.tight_layout()
    plt.show()
    plt.savefig(output_path)
    plt.close(fig)

class get_pre_floor_post_and_lineload:
    def __init__(self,resp_struct,bound,house_areas):
        self.resp_struct=resp_struct
        self.bound=bound
        self.house_areas=house_areas
        self.pre_post_dict={}
        self.pre_lineload_dict={}

    @classmethod
    def line_out_of_boundary(cls,line:LineString,house_areas):
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
            if line.covered_by(polygon):
                return False
        return True
    
    @classmethod
    def point_out_of_boundary(cls,pt:Point,house_areas):
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
            if polygon.contains(pt):
                return False
        return True
    
    @classmethod
    def _point_key(cls, point: Point): #为point打上唯一key
        return tuple(int(coord) for coord in (point.x, point.y))

    @classmethod
    def _line_key(cls, line: LineString): #给line打上唯一key
        coords = [tuple(int(c) for c in pt) for pt in line.coords]
        rev_coords = list(reversed(coords))
        return tuple(coords if coords <= rev_coords else rev_coords)
    
    @classmethod
    def get_weight_from_item_force_info(cls,item_force_info):
        react_info=item_force_info["output"]["reactions"]
        max_force=-1e9
        for weight_dict in Group_ASD_Strength_wildcard_list:
            temp_force=0
            for key in react_info.keys():
                force_list=react_info[key]
                for force in force_list:
                    temp_force+=force["magnitude"]*weight_dict[key]
            if temp_force>=max_force:
                result_weight_dict=weight_dict
        return result_weight_dict
    
    def point_in_bound(self,pt:tuple[int,int]):
        x_min,x_max,y_min,y_max=self.bound[0]-30,self.bound[1]+30,self.bound[2]-30,self.bound[3]+30
        if pt[0]>x_max or pt[0]<x_min:
            return False
        elif pt[1]>y_max or pt[1]<y_min:
            return False
        return True

    def line_in_bound(self,line:LineString):
        x_min,x_max,y_min,y_max=self.bound[0]-30,self.bound[1]+30,self.bound[2]-30,self.bound[3]+30
        rect = box(x_min, y_min, x_max, y_max)
        if rect.contains(line):
            return True
        return False
    
    def get_pre_post_from_post(self): #统计底层每个post受到多少力
        for post in self.resp_struct["post"]:
            weight_dict=self.get_weight_from_item_force_info(item_force_info=post["item_force_info"])
            post_info=post["item_force_info"]["output"]
            post_key=self._point_key(Point(post_info["element_points"][0][0],post_info["element_points"][0][1]))
            #if self.point_in_bound(pt=post_key)==False:
            if self.point_out_of_boundary(pt=Point(post_key),house_areas=self.house_areas):
                continue
            if post_key not in self.pre_post_dict.keys():
                self.pre_post_dict[post_key]=POST_ITEM(info="",force=0)
            
            post_force_dict=post_info["reactions"]
            for post_force_dict_key in post_force_dict.keys():
                post_force_list=post_force_dict[post_force_dict_key]
                for post_force in post_force_list:
                    self.pre_post_dict[post_key].force+=post_force["magnitude"]*weight_dict[post_force_dict_key]
    
    @classmethod
    def get_pos_from_relative_pos(cls,points,pos):
        x1,y1,x2,y2=points[0][0],points[0][1],points[1][0],points[1][1]
        L = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
        #print(f"points:{points},pos:{pos},L:{L}")
        def interp(pos):
            t=100*pos/L
            #print(f"t:{t}")
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            #print(f"({x},{y})")
            return (int(x), int(y))
        return interp(pos)
    
    def get_pre_lineload_from_seg(self,seg):
        weight_dict=self.get_weight_from_item_force_info(item_force_info=seg["item_force_info"])
        seg_info=seg["item_force_info"]["output"]
        seg_points=seg_info["element_points"]
        seg_force_dict=seg_info["reactions"]
        for key_force_type in seg_force_dict.keys():
            seg_force_list=seg_force_dict[key_force_type]
            for seg_force in seg_force_list:
                #print(f"key_force_type:{key_force_type},seg_force:{seg_force}")
                if seg_force["start"]!=seg_force["end"]:
                    pt_1,pt_2=self.get_pos_from_relative_pos(points=seg_points,pos=seg_force["start"]),\
                        self.get_pos_from_relative_pos(points=seg_points,pos=seg_force["end"])
                    lineload_key=self._line_key(LineString([(pt_1),(pt_2)]))
                    #pos_1,pos_2=seg_force["start"],seg_force["end"]
                    #print(f"points:{seg_points},pos_1:{pos_1},pos_2:{pos_2},pt_1:{pt_1},pt_2:{pt_2}")
                    if lineload_key not in self.pre_lineload_dict.keys():
                        self.pre_lineload_dict[lineload_key]=LINELOAD_ITEM(info="",force=0)
                    self.pre_lineload_dict[lineload_key].force+=seg_force["magnitude"]*weight_dict[key_force_type]
                    element_id,from_element=seg_info["element_id"],seg_force["from_element"]
                    self.pre_lineload_dict[lineload_key].info+=element_id+f" from {from_element}"+"/"

    def get_pre_post_and_lineload_from_segs(self): #依据底层的beam/shearwall统计传下来的post和lineload    
        for wall in self.resp_struct["wall"]:
            wall_points=wall["item_base_info"]["points"]
            #if self.line_in_bound(LineString([(wall_points[0][0],wall_points[0][1]),(wall_points[1][0],wall_points[1][1])]))==False:
            if self.line_out_of_boundary(LineString([wall_points[0],wall_points[1]]),house_areas=self.house_areas)==True:
                continue
            #print("wall")
            self.get_pre_lineload_from_seg(seg=wall)
    
    def get_pre_post_and_pre_lineload(self):
        self.get_pre_post_from_post()
        self.get_pre_post_and_lineload_from_segs()
        return {"pre_post_dict":self.pre_post_dict,"pre_lineload_dict":self.pre_lineload_dict}

class get_initial_struct_based_on_response:
    def __init__(self,floor,context,answer,floor_design_pre):
        self.floor=floor
        self.context=context
        self.answer=answer
        self.floor_design_pre=floor_design_pre
    BETWEEN_FLOOR_EPS=30
    
    @classmethod
    def get_beam_index_and_json(cls,floor,beams):
        sta_id=0
        beam_result_list=[]
        for beam in beams:
            beam_result_list.append({
                "floor":floor,
                "line": {
                    "start": [beam[1][0],beam[1][1]],
                    "end": [beam[2][0],beam[2][1]]
                },
                "attributes": {
                    "id": f"{floor}-beam-{sta_id}",
                    "src": "shear_system",
                }})
            sta_id=sta_id+1
        return beam_result_list

    @classmethod
    def get_shearwall_index_and_json(cls,floor,shearwalls):
        sta_id=0
        shearwall_result_list=[]
        for shearwall in shearwalls:
            shearwall_result_list.append({
                "floor":floor,
                "line": {
                    "start": [shearwall[1][0],shearwall[1][1]],
                    "end": [shearwall[2][0],shearwall[2][1]]
                },
                "attributes": {
                    "id": f"{floor}-shearwall-{sta_id}"
            }})
            sta_id=sta_id+1
        return shearwall_result_list

    @classmethod
    def get_wall_index_and_json(cls,floor,walls,shearwalls):
        sta_id=0
        wall_result_list=[]
        for wall in walls:
            IS_SHEARWALL=False
            for shearwall in shearwalls:
                if LineString([wall[1],wall[2]]).equals(LineString([shearwall[1],shearwall[2]])):
                    IS_SHEARWALL=True
            wall_result_list.append({
                "floor":floor,
                "line": {
                "start": [wall[1][0],wall[1][1]],
                "end": [wall[2][0],wall[2][1]]
            },
                "attributes": {
                "id": f"{floor}-wall-{sta_id}",
                "is_shear_wall": IS_SHEARWALL,
            }
            })
            sta_id=sta_id+1
        return wall_result_list
    
    @classmethod
    def point_overlap_point(cls,pt_1:Point,pt_2:Point,eps=None):
        if eps==None:
            eps=cls.BETWEEN_FLOOR_EPS
        return pt_1.distance(pt_2) < eps
    
    @classmethod
    def point_in_line(cls,pt:Point,line:LineString,eps=None):
        if eps==None:
            eps=cls.BETWEEN_FLOOR_EPS
        return pt.distance(line) < eps
    
    @classmethod
    def is_parallel(cls,line_1:LineString, line_2:LineString, angle_tol=1e-6):
        x1, y1 = line_1.coords[0]
        x2, y2 = line_1.coords[-1]
        dx1, dy1 = x2 - x1, y2 - y1

        x3, y3 = line_2.coords[0]
        x4, y4 = line_2.coords[-1]
        dx2, dy2 = x4 - x3, y4 - y3

        cross = dx1 * dy2 - dy1 * dx2
        return abs(cross) < angle_tol

    @classmethod
    def line_overlap_line(cls,line_1:LineString,line_2:LineString,eps=None):
        if eps==None:
            eps=cls.BETWEEN_FLOOR_EPS
        if not cls.is_parallel(line_1=line_1, line_2=line_2):
            #print(f"{line_1}与{line_2}不可看作平行")
            return False
        #print(line_1.distance(line_2))
        return line_1.distance(line_2) < eps
    
    @classmethod
    def fuzzy_line_overlap_length(cls,line_1:LineString, line_2:LineString): #已知line_1与line_2可以看作重合,求line_2向line_1所有的线投影长度
        t1 = line_1.project(Point([line_2.coords[0]])) #L.project(P)表示从L的起点到到“离P最近的投影点”时，沿线走过的距离(最长为L)
        t2 = line_1.project(Point([line_2.coords[-1]]))
        t_min, t_max = sorted([t1, t2])
        L = line_1.length
        result_overlap_length = max(0.0, min(L, t_max) - max(0.0, t_min))
        return result_overlap_length

    @classmethod
    def get_post_from_design(cls,walls,shearwalls,beams,floor,post_pre_list):
        UNSOLVED_POST_NUM=0
        post_list=[] #依据构造得到post_list
        for beam in beams:
            beam_pt_1,beam_pt_2=beam[1],beam[2]
            for shearwall in shearwalls:
                shearwall_pt_1,shearwall_pt_2=shearwall[1],shearwall[2]
                #if beam_pt_1 not in post_list and point_is_on_line(beam_pt_1,shearwall_pt_1,shearwall_pt_2,code=0)==True:
                if beam_pt_1 not in post_list and cls.point_in_line(Point(beam_pt_1), LineString([shearwall_pt_1, shearwall_pt_2])) == True:
                    post_list.append(beam_pt_1)
                #if beam_pt_2 not in post_list and point_is_on_line(beam_pt_2,shearwall_pt_1,shearwall_pt_2,code=0)==True:
                if beam_pt_2 not in post_list and cls.point_in_line(Point(beam_pt_2), LineString([shearwall_pt_1, shearwall_pt_2])) == True:
                    post_list.append(beam_pt_2)
                #if shearwall_pt_1 not in post_list and point_is_on_line(shearwall_pt_1,beam_pt_1,beam_pt_2,code=0)==True: #把与beam的交点加入post_list
                if shearwall_pt_1 not in post_list and cls.point_in_line(Point(shearwall_pt_1), LineString([beam_pt_1, beam_pt_2])) == True:
                    post_list.append(shearwall_pt_1)
                #if shearwall_pt_2 not in post_list and point_is_on_line(shearwall_pt_2,beam_pt_1,beam_pt_2,code=0)==True:
                if shearwall_pt_2 not in post_list and cls.point_in_line(Point(shearwall_pt_2), LineString([beam_pt_1, beam_pt_2])) == True:
                    post_list.append(shearwall_pt_2)
        
        if post_pre_list!=[]:
            for post_pre in post_pre_list:
                POST_SOLVED=False
                pt_post_pre=(post_pre["point"][0],post_pre["point"][1])
                if pt_post_pre in post_list:
                    POST_SOLVED=True
                if POST_SOLVED==False:
                    for beam in beams:
                        beam_pt_1,beam_pt_2=beam[1],beam[2]
                        #if point_is_on_line(pt_post_pre,beam_pt_1,beam_pt_2,code=0)==True:
                        if cls.point_in_line(Point(pt_post_pre), LineString([beam_pt_1, beam_pt_2])) == True:
                            POST_SOLVED=True
                            break
                if POST_SOLVED==False:
                    for wall in walls:
                        wall_pt_1,wall_pt_2=wall[1],wall[2]
                        #if point_is_on_line(pt_post_pre,wall_pt_1,wall_pt_2,code=0)==True:
                        if cls.point_in_line(Point(pt_post_pre), LineString([wall_pt_1, wall_pt_2])) == True:
                            post_list.append(pt_post_pre)
                            POST_SOLVED=True
                            break
                if POST_SOLVED==False:
                    #print(f"pt_post_pre:{pt_post_pre}未解决")
                    UNSOLVED_POST_NUM=UNSOLVED_POST_NUM+1
                    #break
        else:
            POST_SOLVED=True
            UNSOLVED_POST_NUM=0

        post_result_list=[] #将post_list的内容构造到post_result_list
        sta_id=0
        for post in post_list:
            post_result_list.append({
                "id": f"{floor}_post_{sta_id}",
                "floor": floor,
                "point": [post[0],post[1]],
                "is_virtual": False,
                "valid":True,
                "post_types": [
                    "shearwall_end",
                    "inter_floor_transfer"
                ],
                })
        return post_result_list,UNSOLVED_POST_NUM

    def get_house_struct(self):
        walls=get_segments_info_five_plus_two(self.context,[],["wall","exterior_wall"],[])
        beams=get_segments_info_five_plus_two(self.answer,[],["beam"],[])
        shearwalls=get_segments_info_five_plus_two(self.answer,[],["shearwall"],[])

        walls_json,shearwalls_json,beams_json=self.get_wall_index_and_json(floor=self.floor,walls=walls,shearwalls=shearwalls),\
        self.get_shearwall_index_and_json(floor=self.floor,shearwalls=shearwalls),self.get_beam_index_and_json(floor=self.floor,beams=beams),\
        
        if self.floor_design_pre==None:
            posts_json,UNSOLVED_POST_NUM=self.get_post_from_design(walls=walls,shearwalls=shearwalls,beams=beams,floor=self.floor,post_pre_list=[])
        else:
            posts_json,UNSOLVED_POST_NUM=self.get_post_from_design(walls=walls,shearwalls=shearwalls,
                    beams=beams,floor=self.floor,post_pre_list=self.floor_design_pre["posts"])
        
        floor_design={
            "walls":walls_json,
            "shearwalls":shearwalls_json,
            "beams":beams_json,
            "posts":posts_json
        }
        return floor_design,UNSOLVED_POST_NUM
      
class calculate_price_reward:
    def __init__(self,resp_struct,extra_data,analysis_score,floor,
                 pre_lineload_dict,pre_post_dict,floor_design):
        self.resp_struct=resp_struct
        self.extra_data=extra_data
        self.analysis_score=analysis_score
        self.floor=floor
        self.pre_lineload_dict=pre_lineload_dict
        self.floor_design=floor_design
        self.pre_post_dict=pre_post_dict
    OVERLAP_THRESHOLD=0.5
    @classmethod
    def get_weight_from_item_force_info(cls,item_force_info):
        gravity_info=item_force_info["output"]["loads_detail"]["gravity"]
        max_force=-1e9
        for weight_dict in Group_ASD_Strength_wildcard_list:
            temp_force=0
            for key in gravity_info.keys():
                force_list=gravity_info[key]
                for force in force_list:
                    temp_force+=force["magnitude"]*weight_dict[key]
            if temp_force>=max_force:
                result_weight_dict=weight_dict
        return result_weight_dict

    @classmethod
    def get_warn_error_from_elm_item(cls,elm_item):
        warn_cnt,error_cnt,warn_list,error_list=0,0,[],[]
        for check_info in elm_item["item_check_info"]:
            if check_info["level"]=="ERROR":
                error_cnt+=1
                error_list.append(check_info)
            elif check_info["level"]=="WARN":
                warn_cnt+=1
                warn_list.append(check_info)
        return {"warn_cnt":warn_cnt,"error_cnt":error_cnt,"warn_list":warn_list,"error_list":error_list}

    def get_warn_error_from_resp(self): #这一部分只负责得到本层楼的建模错误，如果有上层的传力错误通过对pre_force的检查实现
        warn_cnt,error_cnt=0,0
        post_warn_cnt,post_error_cnt,beam_warn_cnt,beam_error_cnt,shearwall_warn_cnt,shearwall_error_cnt=0,0,0,0,0,0
        post_warn_list,beam_warn_list,shearwall_warn_list,post_error_list,beam_error_list,shearwall_error_list=[],[],[],[],[],[]
        post_resp_list,beam_resp_list,shearwall_resp_list=\
            self.resp_struct["post"],self.resp_struct["beam"],self.resp_struct["shearwall"]
        
        for post_resp in post_resp_list:
            result=self.get_warn_error_from_elm_item(elm_item=post_resp)
            post_warn_cnt+=result["warn_cnt"]
            post_error_cnt+=result["error_cnt"]
            post_warn_list+=result["warn_list"]
            post_error_list+=result["error_list"]
        for beam_resp in beam_resp_list:
            result=self.get_warn_error_from_elm_item(elm_item=beam_resp)
            beam_warn_cnt+=result["warn_cnt"]
            beam_error_cnt+=result["error_cnt"]
            beam_warn_list+=result["warn_list"]
            beam_error_list+=result["error_list"]
        for shearwall_resp in shearwall_resp_list:
            result=self.get_warn_error_from_elm_item(elm_item=shearwall_resp)
            shearwall_warn_cnt+=result["warn_cnt"]
            shearwall_error_cnt+=result["error_cnt"]
            shearwall_warn_list+=result["warn_list"]
            shearwall_error_list+=result["error_list"]
        
        warn_cnt=post_warn_cnt+beam_warn_cnt+shearwall_warn_cnt
        error_cnt=post_error_cnt+beam_error_cnt+shearwall_error_cnt
        warning_result={"warn_cnt":warn_cnt,"error_cnt":error_cnt,"post_warn_cnt":post_warn_cnt,
        "post_error_cnt":post_error_cnt,"beam_warn_cnt":beam_warn_cnt,
        "beam_error_cnt":beam_error_cnt,"shearwall_warn_cnt":shearwall_warn_cnt,"shearwall_error_cnt":shearwall_error_cnt,
        "post_warn_list":post_warn_list,"post_error_list":post_error_list,"beam_warn_list":beam_warn_list,"beam_error_list":beam_error_list,
        "shearwall_warn_list":shearwall_warn_list,"shearwall_error_list":shearwall_error_list}

        return warning_result
    
    @classmethod
    def dcr_score_func(cls,dcr):
        def func_low(x):
            return 0.6*math.exp(-5*(0.5-x))
        def func_up(x):
            return 0.6*math.exp(-10*(x-0.8))
        def func_middle(x):
            return 0.6+0.4*(x-0.5)/0.3
        if dcr<0.5:
            return func_low(dcr)
        if dcr>0.8:
            return func_up(dcr)
        return func_middle(dcr)

    @classmethod
    def get_seg_loaded_force(cls,seg_info,weight_dict):
        loaded_force=0
        post_force_dict=seg_info["loads_detail"]["gravity"]
        for post_force_dict_key in post_force_dict.keys():
            post_force_list=post_force_dict[post_force_dict_key]
            for post_force in post_force_list:
                loaded_force+=post_force["magnitude"]*weight_dict[post_force_dict_key]
        return loaded_force
    
    @classmethod
    def calculate_global_dcr_score(cls,loaded_force_list,dcr_list):
        total_loaded_force,global_dcr_score=0,0
        for i in range(len(loaded_force_list)):
            loaded_force,dcr=loaded_force_list[i],dcr_list[i]
            if loaded_force<=0 or dcr<=0:
                continue
            total_loaded_force+=loaded_force
        for i in range(len(loaded_force_list)):
            loaded_force,dcr=loaded_force_list[i],dcr_list[i]
            if loaded_force<=0 or dcr<=0:
                continue
            w=loaded_force/total_loaded_force
            global_dcr_score+=w*cls.dcr_score_func(dcr=dcr)
        return global_dcr_score

    def get_global_dcr_score_from_resp(self):
        loaded_force_list,dcr_list,dcr_score_list=[],[],[]
        #直接读取计算post的受力信息和dcr信息，然后加到总分
        post_loaded_force_list,post_dcr_list,post_dcr_score_list=[],[],[]
        for post in self.resp_struct["post"]:
            if "output" not in post["item_force_info"]:
                continue
            weight_dict=self.get_weight_from_item_force_info(item_force_info=post["item_force_info"])
            post_info=post["item_force_info"]["output"]
            post_loaded_force=self.get_seg_loaded_force(seg_info=post_info,weight_dict=weight_dict)
            post_dcr=post_info["force_detail"]["material"]["DCR"]
            post_dcr=float(post_dcr)
            post_dcr_score=self.dcr_score_func(dcr=post_dcr)
            post_loaded_force_list.append(post_loaded_force)
            post_dcr_list.append(post_dcr)
            dcr_score_list.append(post_dcr_score)
        #处理shearwall
        shearwall_loaded_force_list,shearwall_dcr_list,shearwall_dcr_score_list=[],[],[]
        shearwall_load_force_dict={}
        for wall in self.resp_struct["wall"]: #直接从wall中读取受力信息初始化shearwall_force_dict
            if "output" not in wall["item_force_info"]:
                continue
            weight_dict=self.get_weight_from_item_force_info(item_force_info=wall["item_force_info"])
            wall_info=wall["item_force_info"]["output"]
            wall_loaded_force=self.get_seg_loaded_force(seg_info=wall_info,weight_dict=weight_dict)
            wall_element_id=wall_info["element_id"]
            shearwall_load_force_dict[wall_element_id]=wall_loaded_force
        for shearwall in self.extra_data["Design Result"]["Shear wall"]["Detailed Design"][self.floor]: #整合从extra_data中读取shearwall的dcr信息
            shearwall_info=shearwall["Stud Design"][0]
            shearwall_element_id=shearwall_info["Wall ID"]
            shearwall_dcr=shearwall_info["DCR"]
            shearwall_dcr=float(shearwall_dcr)
            shearwall_dcr_score=self.dcr_score_func(dcr=shearwall_dcr)
            if shearwall_element_id in shearwall_load_force_dict.keys():
                shearwall_loaded_force_list.append(shearwall_load_force_dict[shearwall_element_id])
                shearwall_dcr_list.append(shearwall_dcr)
                shearwall_dcr_score_list.append(shearwall_dcr_score)
            else:
                print(f"{self.floor}中找不到{shearwall_element_id}")
        #处理beam
        beam_loaded_force_list,beam_dcr_list,beam_dcr_score_list=[],[],[]
        beam_load_force_dict={}
        for beam in self.resp_struct["beam"]: #直接从beam中读取受力信息初始化beam_force_dict
            if "output" not in beam["item_force_info"]:
                continue
            weight_dict=self.get_weight_from_item_force_info(item_force_info=beam["item_force_info"])
            beam_info=beam["item_force_info"]["output"]
            beam_loaded_force=self.get_seg_loaded_force(seg_info=beam_info,weight_dict=weight_dict)
            beam_element_id=beam_info["element_id"]
            beam_load_force_dict[beam_element_id]=beam_loaded_force
        #print(f"beam_load_force_dict:{beam_load_force_dict}")
        for detail in self.extra_data["Design Result"]["Beam Design"]["detail"]: #整合从extra_data中读取beam的dcr信息
            if detail["floor"]!=self.floor:
                continue
            beam_list=detail["data"]
            break
        for beam_info in beam_list:
            beam_element_id=beam_info["id"]
            beam_dcr=beam_info["Moment"]
            beam_dcr=float(beam_dcr.strip('%')) / 100
            beam_dcr_score=self.dcr_score_func(dcr=beam_dcr)
            if beam_element_id in beam_load_force_dict.keys():
                beam_loaded_force_list.append(beam_load_force_dict[beam_element_id])
                beam_dcr_list.append(beam_dcr)
                beam_dcr_score_list.append(beam_dcr_score)
            else:
                print(f"{self.floor}中找不到{beam_element_id}")
        
        #计算global_dcr_score
        loaded_force_list=post_loaded_force_list+shearwall_loaded_force_list+beam_loaded_force_list
        dcr_list=post_dcr_list+shearwall_dcr_list+beam_dcr_list
        dcr_score_list=post_dcr_score_list+shearwall_dcr_score_list+beam_dcr_score_list
        global_dcr_score=self.calculate_global_dcr_score(loaded_force_list=loaded_force_list,dcr_list=dcr_list)
        return {"global_dcr_score":global_dcr_score,
                "loaded_force_list":loaded_force_list,"dcr_list":dcr_list,"dcr_score_list":dcr_score_list,
                "post_loaded_force_list":post_loaded_force_list,"post_dcr_list":post_dcr_list,
                "post_dcr_score_list":post_dcr_score_list,"shearwall_loaded_force_list":shearwall_loaded_force_list,
                "shearwall_dcr_list":shearwall_dcr_list,"shearwall_dcr_score_list":shearwall_dcr_score_list,
                "beam_loaded_force_list":beam_loaded_force_list,"beam_dcr_list":beam_dcr_list,
                "beam_dcr_score_list":beam_dcr_score_list}

    def get_house_price(self):
        return self.analysis_score["Total Structural Materials"]
        
    def get_unsolved_lineload(self):
        unsolved_lineload_list,unsolved_lineload_num=[],0
        beams,walls=self.floor_design["beams"],self.floor_design["walls"]
        for lineload_key in self.pre_lineload_dict.keys():
            overlap_list=[]
            lineload_item=self.pre_lineload_dict[lineload_key]
            total_overlap_length=0
            lineload_line=LineString([lineload_key[0], lineload_key[1]])
            if lineload_line.length<get_initial_struct_based_on_response.BETWEEN_FLOOR_EPS:
                continue
            for beam in beams:
                beam_pt_1,beam_pt_2=beam["line"]["start"],beam["line"]["end"]
                beam_line=LineString([beam_pt_1, beam_pt_2])
                if get_initial_struct_based_on_response.line_overlap_line(line_1=lineload_line,line_2=beam_line)==True:
                    overlap_length=get_initial_struct_based_on_response.fuzzy_line_overlap_length(
                                                                    line_1=lineload_line,line_2=beam_line)
                    total_overlap_length+=overlap_length
                    overlap_list.append(beam_line)
            for wall in walls:
                wall_pt_1,wall_pt_2=wall["line"]["start"],wall["line"]["end"]
                wall_line=LineString([wall_pt_1, wall_pt_2])
                if get_initial_struct_based_on_response.line_overlap_line(line_1=lineload_line,line_2=wall_line)==True:
                    overlap_length=get_initial_struct_based_on_response.fuzzy_line_overlap_length(
                                                        line_1=lineload_line,line_2=wall_line)
                    total_overlap_length+=overlap_length
                    overlap_list.append(wall_line)
            if total_overlap_length/lineload_line.length<self.OVERLAP_THRESHOLD:
                #print(f"lineload_line:{lineload_line},overlap_list:{overlap_list}")
                unsolved_lineload_list.append((lineload_key[0],lineload_key[1],lineload_item.force))
                unsolved_lineload_num=unsolved_lineload_num+1
        return unsolved_lineload_list,unsolved_lineload_num
    
    def get_unsolved_post(self):
        unsolved_post_list,unsolved_post_num=[],0
        beams,walls=self.floor_design["beams"],self.floor_design["walls"]
        for post_key in self.pre_post_dict.keys():
            POST_SOLVED=False
            post_item=self.pre_post_dict[post_key]
            for beam in beams:
                beam_pt_1,beam_pt_2=beam["line"]["start"],beam["line"]["end"]
                if get_initial_struct_based_on_response.point_in_line(pt=Point(post_key),line=LineString([beam_pt_1,beam_pt_2])):
                    POST_SOLVED=True
            for wall in walls:
                wall_pt_1,wall_pt_2=wall["line"]["start"],wall["line"]["end"]
                if get_initial_struct_based_on_response.point_in_line(pt=Point(post_key),line=LineString([wall_pt_1,wall_pt_2])):
                    POST_SOLVED=True
            if POST_SOLVED==False:
                unsolved_post_list.append((post_key[0],post_key[1],post_item.force))
                unsolved_post_num+=1
        return unsolved_post_list,unsolved_post_num
    
    def get_unsolved_force_item(self):
        unsolved_lineload_list,unsolved_lineload_num=self.get_unsolved_lineload()
        unsolved_post_list,unsolved_post_num=self.get_unsolved_post()
        return {"unsolved_lineload_list":unsolved_lineload_list,"unsolved_lineload_num":unsolved_lineload_num,
                    "unsolved_post_list":unsolved_post_list,"unsolved_post_num":unsolved_post_num}

def get_analysis_result_from_design(designs,house,house_design_code="XX",floor="XX",f_index="XX",NEED_OUTPUT=False):
    SAVE_HTML=False
    req={"designs":[designs]}
    house_item_path=f"{HOUSE_ITEM_FOLDER_PATH}/{house}/house_items.json" #依据house和floor得到house_items
    with open(house_item_path, "r", encoding="utf-8") as fin:
        house_item=json.load(fin)

    #req_processed=restore_struct_solutions_from_AI(house_json=req,house_items=house_item,generate_params={'postprocess': True})
    #req_processed=req_processed[0]
    analysis_result=get_analysis_items_for_ai_designs(house_items=house_item,house_struct_json=req,project_id=f"{house_design_code}_{floor}")

    #--------测试代码--------------
    #if random.random()<0.2:
    #   analysis_result=[]
    #-----------------------------

    try:
        analysis_score=get_house_analysis_items_score(analysis_result[0][1])
    except:
        print("analysis_result异常！")
        print(f"analysis_result:{analysis_result}")
        OUTPUT_JSON= f"house_data/price_and_valid_check_house_data/req_check_{f_index}_{house_design_code}_{floor}.jsonl"
        with open(OUTPUT_JSON, "w", encoding="utf-8") as fout:
            fout.write("house_item:\n")
            json.dump(house_item,fout,ensure_ascii=False,indent=2)
            fout.write("\n")
            fout.write("req:\n")
            json.dump(req, fout, ensure_ascii=False,indent=2)
        return None
    
    if SAVE_HTML:
        check_html_path="five_plus_two_optimization/five_plus_two_test/test_pic/data_3.27/api_force_check_1/check_html"
        save_check_html(house_analysis_items=analysis_result,output_dir=check_html_path)

    if NEED_OUTPUT==True:
        OUTPUT_JSON= f"house_data/price_and_valid_check_house_data/req_check_{f_index}_{house_design_code}_{floor}.jsonl"
        with open(OUTPUT_JSON, "w", encoding="utf-8") as fout:
            fout.write("house_item:\n")
            json.dump(house_item,fout,ensure_ascii=False,indent=2)
            fout.write("\n")
            fout.write("req:\n")
            json.dump(req, fout, ensure_ascii=False,indent=2)
            #fout.write("\n")
            #fout.write("req_processed:\n")
            #json.dump(req_processed,fout,ensure_ascii=False,indent=2)
            #fout.write("analysis_result:\n")
            #json.dump(analysis_result[0][1], fout, ensure_ascii=False,indent=2)
            fout.write("analysis_result:\n")
            json.dump(analysis_result[0][1], fout, ensure_ascii=False,indent=2)
            fout.write("analysis_score:\n")
            json.dump(analysis_score, fout, ensure_ascii=False,indent=2)
            #fout.write("\n")
            #fout.write("analysis_score:\n")
            #json.dump(analysis_score, fout, ensure_ascii=False,indent=2)
            #fout.write("\n")
            #fout.write("analysis_result_house_check_info:\n")
            #json.dump(analysis_result[0][1]["house_check_info"], fout, ensure_ascii=False,indent=2)
            #fout.write("\n")
            #fout.write("analysis_result_floors:\n")
            #json.dump(analysis_result[0][1]["floors"], fout, ensure_ascii=False,indent=2)
            #fout.write("\n")
            #fout.write("warn_beams:\n")
            #json.dump(warn_beams, fout, ensure_ascii=False,indent=2)
            #fout.write("warn_shearwalls:\n")
            #json.dump(warn_shearwalls, fout, ensure_ascii=False,indent=2)
            #fout.write("warn_floor:\n")
            #json.dump(warn_floor, fout, ensure_ascii=False,indent=2)
    
    return {"resp_floors":analysis_result[0][1]["floors"],"analysis_score":analysis_score,
            "extra_data":analysis_result[0][1]["extra_data"]}

def get_force_items_from_text(pre_lineload_text,pre_post_text):
    #print(pre_lineload_text,pre_post_text)
    pattern_lineload = r"<LINELOAD>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),WEIGHT_\s*(-?\d+)\s*"
    pattern_post = r"<POST>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\),WEIGHT_\s*(-?\d+)\s*"
    pre_lineload_list = [((int(m[0]), int(m[1])), (int(m[2]), int(m[3])), int(m[4]))
                         for m in re.findall(pattern_lineload, pre_lineload_text)]
    pre_post_list = [((int(m[0]), int(m[1])), int(m[2]))
                     for m in re.findall(pattern_post, pre_post_text)]
    return pre_lineload_list,pre_post_list

def transfer_force_text_to_dict(pre_post_text,pre_lineload_text):
    def point_in_line(pt:Point,line:LineString):
        return pt.distance(line) < get_initial_struct_based_on_response.BETWEEN_FLOOR_EPS
    def estimate_force(force):
        result=1
        for _ in range(force):
            result*=10
        return result

    pre_lineload_list,pre_post_list=get_force_items_from_text(pre_lineload_text=pre_lineload_text,pre_post_text=pre_post_text)
    pre_lineload_dict,pre_post_dict={},{}
    lineload_line_list=[]
    for pre_lineload in pre_lineload_list:
        pre_lineload_line=LineString([pre_lineload[0],pre_lineload[1]])
        pre_lineload_dict[get_pre_floor_post_and_lineload._line_key(pre_lineload_line)]=LINELOAD_ITEM(info="",force=estimate_force(pre_lineload[2]))
        lineload_line_list.append(pre_lineload_line)
    for pre_post in pre_post_list:
        pre_post_point=Point(pre_post[0])
        POST_IN_LINELOAD=False
        for lineload_line in lineload_line_list:
            if point_in_line(pt=pre_post_point,line=lineload_line)==True:
                POST_IN_LINELOAD=True
                break
        if POST_IN_LINELOAD==False:
            pre_post_dict[get_pre_floor_post_and_lineload._point_key(pre_post_point)]=POST_ITEM(info="",force=estimate_force(pre_post[1]))
    return pre_lineload_dict,pre_post_dict

def get_api_floor_design_score_item(context,answer,house,floor,designs,floor_design_pre,pre_post_text,pre_lineload_text,design):
    cls_initial_struct=get_initial_struct_based_on_response(floor=floor,context=context, #从api得到error,dcr,price等信息,并计算对应score
    answer=answer,floor_design_pre=floor_design_pre)
    floor_design,_=cls_initial_struct.get_house_struct()
    designs[floor]=floor_design
    resp_result=get_analysis_result_from_design(designs=designs,house=house,house_design_code=str(house)+str(design),floor=floor)
    if resp_result==None:
        print("resp_result为None")
        return None
    
    print("calculate begin")
    pre_lineload_dict,pre_post_dict=transfer_force_text_to_dict(pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text)
    cls_calculate_reward=calculate_price_reward(resp_struct=resp_result["resp_floors"][floor]["structs"]
                ,analysis_score=resp_result["analysis_score"],extra_data=resp_result["extra_data"],floor=floor,
                floor_design=floor_design,pre_lineload_dict=pre_lineload_dict,pre_post_dict=pre_post_dict)
    print("cls_end")
    from five_plus_two_optimization.train_model_new.GRPO.test.test_code.unsupport_force_calculate import get_unsolved_force_item
    unsolved_force_item_result=get_unsolved_force_item(context=context,answer=answer,pre_lineload_text=pre_lineload_text,
                                            pre_post_text=pre_post_text)
    
    error_num=cls_calculate_reward.get_warn_error_from_resp()["error_cnt"]
    error_num+=unsolved_force_item_result["unsolved_lineload_num"]+unsolved_force_item_result["unsolved_post_num"] #得到基础评分结果
    print("error_end")
    dcr=cls_calculate_reward.get_global_dcr_score_from_resp()["global_dcr_score"]
    print("dcr end")
    floor_price=cls_calculate_reward.get_house_price()
    print("calculate end")

    return {"error_num":error_num,"dcr":dcr,"floor_price":floor_price}

def get_floor_pre_designs(target_house,target_design,target_floor,DATA_PATH):
    #input_path="/home/jiuxing_li/train_json_data/five_plus_two_train_jsonl_data/design_3.27/grpo_phase_1_train_data/train_set_inbox_force_prompt_overfitting.jsonl"
    with open(DATA_PATH,"r",encoding="utf-8") as fin:
        floor_design_pre=None
        pre_context,pre_answer=None,None
        designs={}
        for i,line in enumerate(fin):
            #读取输入信息
            record = json.loads(line)
            house,floor,design,context,answer = record["house"],record["floor"],record["design"],\
            record["context"],record["completion_predict"]
            if house!=target_house or design!=target_design:
                continue
            if floor==target_floor:
                break
            cls_initial_struct=get_initial_struct_based_on_response(floor=floor,context=context, #整合设计，得到返回结果
            answer=answer,floor_design_pre=floor_design_pre)
            floor_design,PROCESS_SUCCESS=cls_initial_struct.get_house_struct()
            designs[floor]=floor_design
            floor_design_pre=floor_design
            pre_context,pre_answer=context,answer
    return designs,floor_design_pre,pre_context,pre_answer

def construct_prompt(context,completion_predict,pre_post_text,pre_lineload_text):
    prompt=""
    prompt=prompt+"Making actions based on context and given structures,and support the upper layer for force transmission."
    prompt=prompt+f"context:{context} "
    prompt=prompt+f"upper layer post:{pre_post_text} "
    prompt=prompt+f"upper layer lineload:{pre_lineload_text} "
    prompt=prompt+f"structures:{completion_predict} "
    prompt=f"<s>{prompt}</s>"
    return prompt

def trunc_floor_design_response_and_calculate_score(house,floor,design,context,bound,completion_predict,response
        ,pre_context,pre_answer,designs,floor_design_pre,trunc_result,tokenizer,
        pre_post_text,pre_lineload_text,trunc_index,NEED_VISUALIZE=False):
    
    base_diaphragm_result=get_diaphragm_result(context,completion_predict)
    base_diaphragm_score=base_diaphragm_result["house_score"]

    beams_added=get_segments_info_five_plus_two(completion_predict,[],['beam'],[])
    shear_walls_added=base_diaphragm_result["shear_walls_valid"]
    actions=get_actions_and_index_info_index(response)
    if len(actions)==0:
        return trunc_result
    invalid_process_num=0
    for action_index,action in enumerate(actions): #遍历每个action
        response_after_action=response[:action['act_char_span'][1]]
        answer_after_action=process_completion_predict_filt(context=context,completion_predict=completion_predict,
            response=response_after_action)
        act_type=action['act_type']
        seg_type=action['seg_type']
        
        if seg_type=='beam': #这段代码在这里的作用是统计invalid_process_num
            beam_seg=['beam',(int(action['coords'][0]),int(action['coords'][1])),(int(action['coords'][2]),int(action['coords'][3]))]
            if act_type=='add' and beam_seg in beams_added:
                invalid_process_num=invalid_process_num+1
            elif act_type=='remove' and beam_seg not in beams_added:
                invalid_process_num=invalid_process_num+1
            else:
                if act_type=="add":
                    beams_added.append(beam_seg)
                elif act_type=="remove":
                    beams_added=del_list_last_element(lst=beams_added,elm=beam_seg)
        elif seg_type=='shearwall': #对于shearwall,接口默认是wall的子集,因而如果不是wall的子集也算作invalid_process的一种,因为可以被过滤掉所以不算做error
            sh_seg=['shearwall',(action['coords'][0],action['coords'][1]),(action['coords'][2],action['coords'][3])]
            SH_IS_VALID,SH_IS_VALID_CODE=judge_shearwall_action_is_valid(sh_action=action,walls=base_diaphragm_result["walls"],
            sh_pre_list=[LineString([(x[1][0],x[1][1]),(x[2][0],x[2][1])]) for x in shear_walls_added])
            if act_type=="add" and SH_IS_VALID==False and SH_IS_VALID_CODE==0:
                invalid_process_num=invalid_process_num+1
            elif act_type=="add" and (sh_seg in shear_walls_added or (SH_IS_VALID==False and SH_IS_VALID_CODE==1)):
                invalid_process_num=invalid_process_num+1
            elif act_type=="remove" and sh_seg not in shear_walls_added:
                invalid_process_num=invalid_process_num+1
            else:
                if act_type=="add":
                    shear_walls_added.append(sh_seg)
                elif act_type=="remove":
                    shear_walls_added=del_list_last_element(lst=shear_walls_added,elm=sh_seg)

        diaphragm_result=get_diaphragm_result(context,answer_after_action)
        diaphragm_score=diaphragm_result["house_score"]
        delta_diaphragm_score=diaphragm_score-base_diaphragm_score
        if abs(delta_diaphragm_score) > 1e-6 or action_index==len(actions)-1: #判断是否进入trunc
            base_api_design_result=get_api_floor_design_score_item(context=context,house=house,answer=completion_predict,
                floor=floor,designs=designs,floor_design_pre=floor_design_pre,
                pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text,design=design)
            api_design_result=get_api_floor_design_score_item(context=context,answer=answer_after_action,
                            house=house,floor=floor,designs=designs,floor_design_pre=floor_design_pre,
                            pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text,design=design)
            if base_api_design_result==None or api_design_result==None:
                break
            base_dcr=base_api_design_result["dcr"]
            error_num,dcr,floor_price=api_design_result["error_num"],api_design_result["dcr"],api_design_result["floor_price"]
            delta_dcr=dcr-base_dcr

            trunc_result["prompt_list"].append(construct_prompt(context=context,completion_predict=completion_predict,
            pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text))
            trunc_result["response_list"].append(response_after_action)
            trunc_result["response_encoded_list"].append(tokenizer(response_after_action, return_offsets_mapping=False))
            trunc_result["answer_list"].append(answer_after_action)
            trunc_result["diaphragm_delta_score_list"].append(delta_diaphragm_score)
            trunc_result["diaphragm_area_ratio_list"].append(diaphragm_result["area_ratio"])
            trunc_result["dcr_delta_list"].append(delta_dcr)
            trunc_result["error_num_list"].append(error_num)
            trunc_result["invalid_process_num_list"].append(invalid_process_num)
            trunc_result["floor_price_list"].append(floor_price)

            if NEED_VISUALIZE==True:
                visualize_force_1(context=context,completion_predict=completion_predict,answer=answer_after_action,
                        pre_context=pre_context,pre_answer=pre_answer,            
                        pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text,bound=bound,
                output_path=f"/home/jiuxing_li/five_plus_two_optimization/five_plus_two_test/test_pic/data_3.27/api_trunc_check/{house}_{design}_{floor}_{trunc_index}.png",
                base_diaphragm=[list(poly.exterior.coords) for poly in base_diaphragm_result['valid_diaphragms']],
                diaphragm=[ list(poly.exterior.coords) for poly in diaphragm_result['valid_diaphragms']],
                house_areas=base_diaphragm_result["house_areas"])
            
            #向后递归
            trunc_result=\
                trunc_floor_design_response_and_calculate_score(house=house,floor=floor,design=design,context=context,bound=bound,
                completion_predict=answer_after_action,response=response[action['act_char_span'][1]:],
                pre_context=pre_context,pre_answer=pre_answer,designs=designs,floor_design_pre=floor_design_pre,
                trunc_result=trunc_result,tokenizer=tokenizer,pre_post_text=pre_post_text,
                pre_lineload_text=pre_lineload_text,trunc_index=trunc_index+1,NEED_VISUALIZE=NEED_VISUALIZE)
            break
    return trunc_result

def trunc_floor_design_response_and_calculate_score_1(house,floor,design,context,bound,completion_predict,response_pre,response_after
        ,pre_context,pre_answer,designs,floor_design_pre,trunc_result,tokenizer,
        pre_post_text,pre_lineload_text,trunc_index,NEED_VISUALIZE=False):
    
    completion_predict_combined=process_completion_predict_filt(context=context,completion_predict=completion_predict,response=response_pre)
    base_diaphragm_result=get_diaphragm_result(context,completion_predict_combined)
    base_diaphragm_score=base_diaphragm_result["house_score"]

    beams_added=get_segments_info_five_plus_two(completion_predict_combined,[],['beam'],[])
    shear_walls_added=base_diaphragm_result["shear_walls_valid"]
    actions=get_actions_and_index_info_index(response_after)
    if len(actions)==0:
        return trunc_result
    invalid_process_num=0
    for action_index,action in enumerate(actions): #遍历每个action
        response_after_action=response_pre+response_after[:action['act_char_span'][1]]
        answer_after_action=process_completion_predict_filt(context=context,completion_predict=completion_predict,
            response=response_after_action)
        act_type=action['act_type']
        seg_type=action['seg_type']
        
        if seg_type=='beam': #这段代码在这里的作用是统计invalid_process_num
            beam_seg=['beam',(int(action['coords'][0]),int(action['coords'][1])),(int(action['coords'][2]),int(action['coords'][3]))]
            if act_type=='add' and beam_seg in beams_added:
                invalid_process_num=invalid_process_num+1
            elif act_type=='remove' and beam_seg not in beams_added:
                invalid_process_num=invalid_process_num+1
            else:
                if act_type=="add":
                    beams_added.append(beam_seg)
                elif act_type=="remove":
                    beams_added=del_list_last_element(lst=beams_added,elm=beam_seg)
        elif seg_type=='shearwall': #对于shearwall,接口默认是wall的子集,因而如果不是wall的子集也算作invalid_process的一种,因为可以被过滤掉所以不算做error
            sh_seg=['shearwall',(action['coords'][0],action['coords'][1]),(action['coords'][2],action['coords'][3])]
            SH_IS_VALID,SH_IS_VALID_CODE=judge_shearwall_action_is_valid(sh_action=action,walls=base_diaphragm_result["walls"],
            sh_pre_list=[LineString([(x[1][0],x[1][1]),(x[2][0],x[2][1])]) for x in shear_walls_added])
            if act_type=="add" and SH_IS_VALID==False and SH_IS_VALID_CODE==0:
                invalid_process_num=invalid_process_num+1
            elif act_type=="add" and (sh_seg in shear_walls_added or (SH_IS_VALID==False and SH_IS_VALID_CODE==1)):
                invalid_process_num=invalid_process_num+1
            elif act_type=="remove" and sh_seg not in shear_walls_added:
                invalid_process_num=invalid_process_num+1
            else:
                if act_type=="add":
                    shear_walls_added.append(sh_seg)
                elif act_type=="remove":
                    shear_walls_added=del_list_last_element(lst=shear_walls_added,elm=sh_seg)

        diaphragm_result=get_diaphragm_result(context,answer_after_action)
        diaphragm_score=diaphragm_result["house_score"]
        delta_diaphragm_score=diaphragm_score-base_diaphragm_score
        if abs(delta_diaphragm_score) > 1e-6 or action_index==len(actions)-1: #判断是否进入trunc
            base_api_design_result=get_api_floor_design_score_item(context=context,house=house,answer=completion_predict,
                floor=floor,designs=designs,floor_design_pre=floor_design_pre,
                pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text,design=design)
            api_design_result=get_api_floor_design_score_item(context=context,answer=answer_after_action,
                            house=house,floor=floor,designs=designs,floor_design_pre=floor_design_pre,
                            pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text,design=design)
            if base_api_design_result==None or api_design_result==None:
                break
            base_dcr=base_api_design_result["dcr"]
            error_num,dcr,floor_price=api_design_result["error_num"],api_design_result["dcr"],api_design_result["floor_price"]
            delta_dcr=dcr-base_dcr

            trunc_result["prompt_list"].append(construct_prompt(context=context,completion_predict=completion_predict,
            pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text))
            trunc_result["response_list"].append(response_pre+response_after_action)
            trunc_result["response_encoded_list"].append(tokenizer(response_pre+response_after_action, return_offsets_mapping=False))
            trunc_result["answer_list"].append(answer_after_action)
            trunc_result["diaphragm_delta_score_list"].append(delta_diaphragm_score)
            trunc_result["diaphragm_area_ratio_list"].append(diaphragm_result["area_ratio"])
            trunc_result["dcr_delta_list"].append(delta_dcr)
            trunc_result["error_num_list"].append(error_num)
            trunc_result["invalid_process_num_list"].append(invalid_process_num)
            trunc_result["floor_price_list"].append(floor_price)

            if NEED_VISUALIZE==True:
                visualize_force_1(context=context,completion_predict=completion_predict,answer=answer_after_action,
                        pre_context=pre_context,pre_answer=pre_answer,            
                        pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text,bound=bound,
                output_path=f"/home/jiuxing_li/five_plus_two_optimization/five_plus_two_test/test_pic/data_3.27/api_trunc_check/{house}_{design}_{floor}_{trunc_index}.png",
                base_diaphragm=[list(poly.exterior.coords) for poly in base_diaphragm_result['valid_diaphragms']],
                diaphragm=[ list(poly.exterior.coords) for poly in diaphragm_result['valid_diaphragms']],
                house_areas=base_diaphragm_result["house_areas"])
            
            #向后递归
            trunc_result=\
                trunc_floor_design_response_and_calculate_score(house=house,floor=floor,design=design,context=context,bound=bound,
                completion_predict=completion_predict,response_pre=response_pre+response_after_action,response_after=response[action['act_char_span'][1]:],
                pre_context=pre_context,pre_answer=pre_answer,designs=designs,floor_design_pre=floor_design_pre,
                trunc_result=trunc_result,tokenizer=tokenizer,pre_post_text=pre_post_text,
                pre_lineload_text=pre_lineload_text,trunc_index=trunc_index+1,NEED_VISUALIZE=NEED_VISUALIZE)
            break
    return trunc_result

if __name__=="__main__":#带response版测试
    INPUT_JSON = "train_json_data/five_plus_two_train_jsonl_data/design_3.27/base_model_train/train_set_100_actionized_sort_floor_force_auged(3_times).jsonl"#var
    req=[]
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        house_design_code_pre,floor_design_pre,floor_pre=None,None,None
        designs={}
        pre_context,pre_answer="",""
        for i, line in enumerate(f):
            record = json.loads(line) #依据response生成house_struct
            house,floor,design,bound,context,completion_predict,response = record["house"],record["floor"],record["design"],\
            record['bound'],record["context"],record["completion_predict"],record["response"]
            answer=process_completion_predict_based_on_response(completion_predict=completion_predict,response=response)
            house_design_code=str(house)+str(design)
            house_areas=get_segments_info_five_plus_two(context,["inoutbox"],[],[])
            print(f"{i},{house_design_code},{floor}")

            if house_design_code_pre==None or (house_design_code_pre!=None and house_design_code!=house_design_code_pre): #代表进入一个新的房屋设计
                print("开始新的房屋设计")
                designs={}
                floor_pre=None
            
            cls_initial_struct=get_initial_struct_based_on_response(floor=floor,context=context,
            answer=answer,floor_design_pre=floor_design_pre)
            floor_design,UNSOLVED_POST_NUM=cls_initial_struct.get_house_struct()

            designs[floor]=floor_design
            resp_result=get_analysis_result_from_design(designs=designs,house=house,
                                    house_design_code=house_design_code,floor=floor,f_index=i+1,NEED_OUTPUT=True)
            if i>5:
                break
            
            '''
            if floor_pre!=None:
                print(f"bound:{bound}")
                cls_pre_force_load=get_pre_floor_post_and_lineload(resp_struct=resp_result["resp_floors"][floor_pre]["structs"],
                                                bound=bound,house_areas=house_areas)
                pre_force_load_result=cls_pre_force_load.get_pre_post_and_pre_lineload()
            else:
                pre_force_load_result=None
            
            cls_calculate_reward=calculate_price_reward(resp_struct=resp_result["resp_floors"][floor]["structs"]
                ,analysis_score=resp_result["analysis_score"],extra_data=resp_result["extra_data"],floor=floor)
            warn_error_result=cls_calculate_reward.get_warn_error_from_resp()
            dcr_result=cls_calculate_reward.get_global_dcr_score_from_resp()
            
            #print(f"warn_error_result:{warn_error_result}")
            #pre_post,pre_lineload=force_load_result["pre_post_dict"].keys(),force_load_result["pre_lineload_dict"].keys()
            #print(f"force_load_result:{pre_post,pre_lineload}")
            #print(pre_force_load_result)
            global_dcr_score=dcr_result["global_dcr_score"]
            print(f"dcr_result:{global_dcr_score}")

            if pre_force_load_result==None:
                visualize_force(context=context,answer=answer,pre_context=pre_context,pre_answer=pre_answer,
                                pre_post_data=None,pre_lineload_data=None,bound=bound,
                                output_path=f"five_plus_two_optimization/five_plus_two_test/test_pic/data_3.27/api_force_check/{i+1}_{house_design_code}_{floor}.png")
            else:
                visualize_force(context=context,answer=answer,pre_context=pre_context,pre_answer=pre_answer,
                                pre_post_data=pre_force_load_result["pre_post_dict"]
                                ,pre_lineload_data=pre_force_load_result["pre_lineload_dict"],bound=bound,
                output_path=f"five_plus_two_optimization/five_plus_two_test/test_pic/data_3.27/api_force_check/{i+1}_{house_design_code}_{floor}.png")

            print(f"{house_design_code}_{floor}分析完成")
            house_design_code_pre,floor_design_pre=house_design_code,floor_design
            pre_context,pre_answer,floor_pre=context,answer,floor
            if i>2:
                break
            '''

    '''
    pre_post_text="<POST>(2059,6989),87<POST>(2179,6989),87<POST>(2259,6767),58<POST>(2259,6847),57<POST>(2349,7169),130"
    pre_lineload_text="<LINELOAD>(2364,3016),(2464,3016),4378<LINELOAD>(1724,2731),(1794,2731),2189"
    visualize_force_1(context="test",completion_predict="test",answer="test",pre_context="test",pre_answer="test",
                bound="test",output_path="test",pre_post_text=pre_post_text,pre_lineload_text=pre_lineload_text)
    '''