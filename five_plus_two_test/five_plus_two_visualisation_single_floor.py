import re
import json
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle
import sys
sys.path.append("/mnt/efs/jiuxing_li")
from standard_function import get_segments_info_five_plus_two

TYPE_COLORS = {
    # ========= LINE TYPES（线状构件） =========
    "wall": "#D32F2F",           # 深红 —— 主体承重墙
    "exterior_wall":"#1976D2",
    "shearwall": "#2E7D32",
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

#POLYGON_TYPES = ["opening","roof","room","slab","joist_area","truss_area","inoutbox","house_boundary","exterior_area","floor_box"]
POLYGON_TYPES = ["opening","joist_area","truss_area"]
LINE_TYPES = ["exterior_wall","wall","beam","shearwall"]
POINT_TYPES = []

def plot_diaphragms(ax,house_areas,poly_list,POLYGON_TYPES,LINE_TYPES,POINT_TYPES,title="",code=0,bound=None):
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
    
    #可视化house_area
    for house_area in house_areas: #绘制polygon(半透明)
        color = TYPE_COLORS.get('joist_area', "gray")
        x_list,y_list=[],[]
        for index in range(1,len(house_area)):
            pt=house_area[index]
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

def plot_segments_with_force(ax, segment_list,POLYGON_TYPES,LINE_TYPES,POINT_TYPES,
                cut_line_list=None,title="",code=0,bound=None,post_list=None,lineload_list=None,
                pre_post_list=None,pre_lineload_list=None,unsolved_post_list=None,unsolved_lineload_list=None):
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
    
    if post_list != None:
        x_list,y_list,z_list=[],[],[]
        for post in post_list:
            x_list.append(post[0])
            y_list.append(post[1])
            z_list.append(post[2])
        
        sc=ax.scatter(x_list,y_list,c="yellow",s=z_list,zorder=25)
        for x, y, z in zip(x_list, y_list, z_list):
            ax.annotate(
                f"{z:.1f}",                 # 显示z值
                xy=(x, y),                  # 点的位置
                xytext=(x + 0.1, y + 0.1),  # 文字位置（稍微偏移）
                arrowprops=dict(
                    arrowstyle="->",
                    color="gray",
                    lw=1
                ),
                fontsize=8
            )

    if pre_post_list != None:
        x_list,y_list,z_list=[],[],[]
        for post in pre_post_list:
            x_list.append(post[0])
            y_list.append(post[1])
            z_list.append(post[2])

        sc=ax.scatter(x_list,y_list,c="black",s=z_list,zorder=26,alpha=0.7)
    
    if lineload_list!=None:
        for item in lineload_list:
            x_list,y_list=[item[0][0],item[1][0]],[item[0][1],item[1][1]]
            ax.plot(x_list, y_list, color="orange", linewidth=item[2]*0.05, zorder=25,alpha=0.5)
            # 👉 计算中点
            x_mid = (x_list[0] + x_list[1]) / 2
            y_mid = (y_list[0] + y_list[1]) / 2

            # 👉 标注数值
            ax.text(
                x_mid,
                y_mid,
                f"{item[2]:.1f}",
                fontsize=8,
                color="red",
                ha='center',
                va='bottom'
            )

    if pre_lineload_list!=None:
        for item in pre_lineload_list:
            x_list,y_list=[item[0][0],item[1][0]],[item[0][1],item[1][1]]
            ax.plot(x_list, y_list, color="black", linewidth=item[2]*0.05, zorder=26,alpha=0.7)
    
    if unsolved_post_list != None:
        x_list,y_list,z_list=[],[],[]
        for post in unsolved_post_list:
            x_list.append(post[0])
            y_list.append(post[1])
            z_list.append(post[2])

        sc=ax.scatter(x_list,y_list,c="#F57C00",s=z_list,zorder=28,alpha=0.8)
    
    if unsolved_lineload_list!=None:
        for item in unsolved_lineload_list:
            x_list,y_list=[item[0][0],item[1][0]],[item[0][1],item[1][1]]
            ax.plot(x_list, y_list, color="#F57C00", linewidth=item[2]*0.05, zorder=28,alpha=0.8)

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

def plot_segments_with_force_1(ax, segment_list,POLYGON_TYPES,LINE_TYPES,POINT_TYPES,cut_line_list=None,
                               title="",code=0,bound=None,pre_post_list=None,pre_lineload_list=None,
                               warn_post_list=None,warn_beam_list=None,warn_wall_list=None,
                               error_post_list=None,error_beam_list=None,error_wall_list=None,
                               unsolved_lineload_list=None,unsolved_post_list=None):
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

    if pre_post_list != None:
        x_list,y_list,z_list=[],[],[]
        for post in pre_post_list:
            x_list.append(post[0])
            y_list.append(post[1])
            z_list.append(post[2])

        sc=ax.scatter(x_list,y_list,c="yellow",s=[z*0.1 for z in z_list],zorder=26,alpha=0.7)

    if pre_lineload_list!=None:
        for item in pre_lineload_list:
            x_list,y_list=[item[0][0],item[1][0]],[item[0][1],item[1][1]]
            ax.plot(x_list, y_list, color="orange", linewidth=item[2]*0.005, zorder=26,alpha=0.7)
    
    if unsolved_lineload_list!=None:
        print("plot unsolved lineload")
        for item in unsolved_lineload_list:
            x_list,y_list=[item[0][0],item[1][0]],[item[0][1],item[1][1]]
            ax.plot(x_list, y_list, color="black", linewidth=item[2]*0.005, zorder=28,alpha=0.6)
    
    if unsolved_post_list != None:
        x_list,y_list,z_list=[],[],[]
        for post in unsolved_post_list:
            x_list.append(post[0])
            y_list.append(post[1])
            z_list.append(post[2])
        sc=ax.scatter(x_list,y_list,c="black",s=[z*0.1 for z in z_list],zorder=28,alpha=0.6)

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

def visualize_segments_three_ax(segment_list,structure_predict,structure_processed,output_dir,poly_list=None):
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(12, 6))
    plot_segments(ax1, segment_list,POLYGON_TYPES,LINE_TYPES,POINT_TYPES, None, "Initial Structure")
    #plot_segments(ax2, segment_list, structure_predict, "Initial Structure +Predict Structure")
    #plot_segments(ax3, segment_list, structure_processed, "Initial Structure +Processed Predict Structure",code=1)
    plot_segments(ax2, segment_list,POLYGON_TYPES,LINE_TYPES,POINT_TYPES, structure_processed, "Initial Structure +Predict Structure")
    plot_diaphragms(ax3,poly_list,POLYGON_TYPES,LINE_TYPES,POINT_TYPES,title="diaphragms",code=1)
    
    plt.tight_layout()
    plt.show()
    plt.savefig(output_dir)
    plt.close(fig)

def visualize_segments_two_ax(segment_list,structure_predict,output_dir):
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plot_segments(ax1, segment_list,POLYGON_TYPES,LINE_TYPES,POINT_TYPES, None, "Initial Structure")
    plot_segments(ax2, segment_list,POLYGON_TYPES,LINE_TYPES,POINT_TYPES, structure_predict, "Initial Structure +Predict Structure",code=1)
    
    plt.tight_layout()
    plt.show()
    plt.savefig(output_dir)
    plt.close(fig)

if __name__=="__main__":
    load_path="five_plus_two_optimization/five_plus_two_test/five_plus_two_predict_jsonl/data_2.4/data_2.4_random_0.6_qwen_1.5b_diaphragm_check.jsonl"
    with open(load_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            prompt = record["context"]
            completion_predict = record['completion_predict']
            processed_context = record['processed_predict']
            diaphragms = record['diaphragms']
            house,floor=record['house'],record['floor']
            shear_walls=record['shear_walls']

            segment_list = get_segments_info_five_plus_two(prompt,POLYGON_TYPES=POLYGON_TYPES,
                                            LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES)
            structure_predict= get_segments_info_five_plus_two(completion_predict,POLYGON_TYPES=POLYGON_TYPES,
                                            LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES)
            structure_predict_processed=get_segments_info_five_plus_two(processed_context,POLYGON_TYPES=POLYGON_TYPES,
                                            LINE_TYPES=LINE_TYPES,POINT_TYPES=POINT_TYPES)
            structure_predict_processed=structure_predict_processed+shear_walls
            #print(segment_list)
            
            print(f"Visualizing record {i+1}")
            output_dir=f"five_plus_two_optimization/five_plus_two_test/test_pic/data_2.4/random_0.6/diaphragm/random_0.6_diaphragm_sample_{i+1}_{floor}.png"
            #visualize_segments_two_ax(segment_list, structure_predict=structure_predict,
            #                                output_dir=output_dir)
            visualize_segments_three_ax(segment_list,structure_predict=structure_predict,structure_processed=structure_predict_processed,
                                        output_dir=output_dir,poly_list=diaphragms)
            #break