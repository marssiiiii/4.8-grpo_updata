import re
import json
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle
import sys
sys.path.append("/mnt/efs/jiuxing_li")
from standard_function import get_segments_info_five_plus_two
sys.path.append("five_plus_two_optimization/train_model_new/base_code")
from reward_design import get_design_score

TYPE_COLORS = {
    # ========= LINE TYPES（线状构件） =========
    "wall": "#D32F2F",           # 深红 —— 主体承重墙
    "exterior_wall":"#1976D2",
    "shear_wall": "#2E7D32",
    "door": "#2E7D32",           # 深绿 —— 可通行
    "window": "#1976D2",         # 蓝 —— 采光 / 视线
    "beam": "#7B1FA2",          # 紫 —— 水平承重

    # ========= POINT TYPES（点状构件） =========
    "column": "#212121",         # 深灰黑 —— 垂直承重

    # ========= POLYGON TYPES（面域 / 空间） =========
    "opening": "#F57C00",        # 橙 —— 面开口
    "roof": "#6D4C41",           # 深棕 —— 屋面
    "room": "#81C784",           # 浅绿 —— 室内空间
    "slab": "#BDBDBD",           # 浅灰 —— 楼板
    "joist_area": "#A1887F",     # 灰棕 —— 次结构区
    "truss_area": "#8D6E63",     # 深灰棕 —— 桁架区
    "inoutbox": "#CE93D8",       # 淡紫 —— 过渡空间
    "house_boundary": "#000000", # 黑 —— 建筑边界
    "exterior_area": "#90CAF9",  # 浅蓝 —— 室外区域
    "floorbox": "#FFE082",       # 浅黄 —— 设备 / 功能盒

    # ========= 辅助 / 标注 =========
    "gridline": "#00ACC1"        # 青 —— 轴网
}

POLYGON_TYPES = ["opening","joist_area","truss_area"]
LINE_TYPES = ["exterior_wall","wall","beam","shear_wall"]
POINT_TYPES = []

def plot_diaphragms(ax,poly_list,POLYGON_TYPES,LINE_TYPES,POINT_TYPES,title="",code=0,bound=None):
    if bound==None:
        bound=[-3000,3000,-1000,1000]
    bound[0],bound[1],bound[2],bound[3]=bound[0]-50,bound[1]+50,bound[2]-50,bound[3]+50
    ax.set_xlim(int(bound[0]), int(bound[1]))
    ax.set_ylim(int(bound[2]), int(bound[3]))
    ax.set_aspect("equal", adjustable="box")

    # 绘制网格
    ax.set_xticks(range(int(bound[0]), int(bound[1]), int((bound[1]-bound[0])/10) ))
    ax.set_yticks(range(int(bound[2]), int(bound[3]), int((bound[3]-bound[2])/10) ))
    ax.grid(True, linestyle='--', alpha=0.4)

    # 可视化初始结构
    for poly in poly_list:
        color = (
            random.randint(0, 255)/255.0,
            random.randint(0, 255)/255.0,
            random.randint(0, 255)/255.0,
        )
        x_list,y_list=[],[]
        
        for index in range(len(poly)):
            pt=poly[index]
            x_list.append(pt[0])
            y_list.append(pt[1])
        points = list(zip(x_list, y_list))
        poly = Polygon(
            points,
            closed=True,
            facecolor=color,   # 填充颜色
            alpha=0.4,             # 半透明
            zorder=1,
        )
        ax.add_patch(poly)
        ax.set_aspect('equal')
    
    if code!=0:
        import matplotlib.patches as mpatches
        legend_handles = []
        for type_name, color in TYPE_COLORS.items():
            if type_name not in POLYGON_TYPES and type_name not in LINE_TYPES and type_name not in POINT_TYPES:
                continue
            legend_handles.append(mpatches.Patch(color=color, label=type_name))
        ax.legend(handles=legend_handles, loc='upper right', 
            bbox_to_anchor=(1.5, 1.5))
    
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def plot_segments(ax, segment_list,POLYGON_TYPES,LINE_TYPES,POINT_TYPES,cut_line_list=None,title="",code=0,bound=None):
    if bound==None:
        bound=[-3000,3000,-1000,1000]
    bound[0],bound[1],bound[2],bound[3]=bound[0]-50,bound[1]+50,bound[2]-50,bound[3]+50
    ax.set_xlim(int(bound[0]), int(bound[1]))
    ax.set_ylim(int(bound[2]), int(bound[3]))
    ax.set_aspect("equal", adjustable="box")

    # 绘制网格
    ax.set_xticks(range(int(bound[0]), int(bound[1]), int((bound[1]-bound[0])/10) ))
    ax.set_yticks(range(int(bound[2]), int(bound[3]), int((bound[3]-bound[2])/10) ))
    ax.grid(True, linestyle='--', alpha=0.4)

    # 可视化初始结构
    for seg in segment_list:
        color = TYPE_COLORS.get(seg[0], "gray")
        if color=="gray":
            continue
        x_list,y_list=[],[]
        
        if seg[0] in POLYGON_TYPES: #绘制polygon(半透明)
            for index in range(1,len(seg)):
                pt=seg[index]
                x_list.append(pt[0])
                y_list.append(pt[1])
            points = list(zip(x_list, y_list))
            poly = Polygon(
                points,
                closed=True,
                facecolor=color,   # 填充颜色
                alpha=0.4,             # 半透明
                zorder=1,
            )
            ax.add_patch(poly)
            ax.set_aspect('equal')
            ax.autoscale()

        elif seg[0] in LINE_TYPES: #绘制线
            for index in range(1,len(seg)):
                pt=seg[index]
                x_list.append(pt[0])
                y_list.append(pt[1])
            ax.plot(x_list, y_list, color=color, linewidth=2, zorder=11)
            ax.scatter(x_list, y_list, color=color, s=20, zorder=11)

        elif seg[0] in POINT_TYPES: #绘制点（小矩形）
            for index in range(1,len(seg)):
                pt=seg[index]
                x_list.append(pt[0])
                y_list.append(pt[1])
            w = h = 6
            half = w / 2
            for x, y in zip(x_list, y_list):
                rect = Rectangle(
                    (x - half, y - half),
                    w,
                    h,
                    facecolor=color,
                    edgecolor='black',
                    alpha=1,
                    zorder=13
                )
                ax.add_patch(rect)
    
    # 可视化切割线结构
    if cut_line_list!=None:
        for seg in cut_line_list:
            color = TYPE_COLORS.get(seg[0], "gray")
            if color=="gray":
                continue
            x_list,y_list=[],[]
            if seg[0] in LINE_TYPES: #绘制线
                for index in range(1,len(seg)):
                    pt=seg[index]
                    x_list.append(pt[0])
                    y_list.append(pt[1])
                ax.plot(x_list, y_list, color=color, linewidth=2, zorder=20)
                ax.scatter(x_list, y_list, color=color, s=20, zorder=20)

        #按照TYPE_COLORS添加图例
        if code!=0:
            import matplotlib.patches as mpatches
            legend_handles = []
            for type_name, color in TYPE_COLORS.items():
                if type_name not in POLYGON_TYPES and type_name not in LINE_TYPES and type_name not in POINT_TYPES:
                    continue
                legend_handles.append(mpatches.Patch(color=color, label=type_name))
            ax.legend(handles=legend_handles, loc='upper right', 
                bbox_to_anchor=(1.5, 1.5))
        
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def visualize_segments_four_ax(segment_list,beams,shear_walls,valid_beams,valid_shear_walls,output_dir,poly_list,valid_poly_list,bound):
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    ax1,ax2,ax3=axes[0]
    ax4,ax5,ax6=axes[1]
    plot_segments(ax=ax1,segment_list=segment_list,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,
                  cut_line_list=None, title="Initial Structure",bound=bound)
    #plot_segments(ax2, segment_list, structure_predict, "Initial Structure +Predict Structure")
    #plot_segments(ax3, segment_list, structure_processed, "Initial Structure +Processed Predict Structure",code=1)
    plot_segments(ax=ax2, segment_list=segment_list,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES
                  ,POINT_TYPES=POINT_TYPES, cut_line_list=beams+shear_walls,bound=bound,title="Predict Structure")
    plot_segments(ax=ax3,segment_list=segment_list,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,
                  POINT_TYPES=POINT_TYPES,cut_line_list=valid_beams+valid_shear_walls,bound=bound,title="Valid Predict Structure")
    plot_diaphragms(ax=ax4,poly_list=poly_list,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES,
                    bound=bound,title="diaphragms")
    plot_diaphragms(ax=ax5,poly_list=valid_poly_list,POLYGON_TYPES=POLYGON_TYPES,LINE_TYPES=LINE_TYPES,
                    POINT_TYPES=POINT_TYPES,bound=bound,title="valid diaphragms",code=1)

    plt.tight_layout()
    plt.show()
    plt.savefig(output_dir)
    plt.close(fig)


if __name__=="__main__":
    load_path="train_json_data/five_plus_two_train_jsonl_data/design_3.10/initial_data_train_set_100_shf.jsonl"
    with open(load_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            context = record["context"]
            completion_predict = record['completion_predict']
            house,floor=record['house'],record['floor']
            bound=record["bound"]
            result=get_design_score(context,completion_predict)

            segment_list = get_segments_info_five_plus_two(context,POLYGON_TYPES=POLYGON_TYPES,
                                            LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES)
            diaphragms,valid_diaphragms=result["diaphragms"],result["valid_diaphragms"]

            #print(segment_list)
            
            print(f"Visualizing record {i+1}")
            output_dir=f"five_plus_two_optimization/five_plus_two_test/test_pic/data_3.10/train_100/sample_{i+1}_{house}_{floor}.png"
            visualize_segments_four_ax(segment_list=segment_list,beams=result["beams"],shear_walls=result["shear_walls"],
                                       valid_beams=result["valid_beams"],valid_shear_walls=result["shear_walls_valid"],
                                        output_dir=output_dir,poly_list=[list(poly.exterior.coords) for poly in diaphragms],
                                        valid_poly_list=[list(poly.exterior.coords) for poly in valid_diaphragms],bound=bound
            )
            if i>20:
                break
            #break