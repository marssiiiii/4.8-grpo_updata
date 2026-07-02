#该代码提供只含两点的LineString的交叉，包含，不相交的判断
import math
from shapely.geometry import LineString

def orientation(p, q, r, tol=0): #判断p,q,r三点是否共线,0代表共线
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]) #计算pq向量与qr向量的乘积
    if abs(val) <= tol:
        return 0
    return 1 if val > 0 else 2

def similar(node1,node2,tol):
    if abs(node1[0]-node2[0])<tol and abs(node1[1]-node2[1])<tol:
        return True
    return False

def on_segment(p, q, r, tol): #在已知p,q,r三点共线的情况下判断点q是否在pr线上,不包含端点
    return (min(p[0], r[0]) < q[0] < max(p[0], r[0]) and \
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1])) or \
        (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and 
                min(p[1], r[1]) < q[1] < max(p[1], r[1]))
#    return (min(p[0], r[0]) + tol < q[0] < max(p[0], r[0]) - tol or
#            min(p[1], r[1]) + tol < q[1] < max(p[1], r[1]) - tol)

def is_cross_or_intersect(p1, q1, p2, q2, tol): #交叉返回0，有交点线2在线1内部返回1，线1在线2内部返回2,并返回支撑点
    # 情况 1：判断是否为端点接触（L型、T型、直角）
    if similar(p1,p2,tol) or similar(p1,q2,tol) or similar(q1,p2,tol) or similar(q1,q2,tol):
        return -1,None  # 端点接触不算相交
    
    # 包围盒快速排除（如果两线不相交，则返回-1）
    if (max(p1[0], q1[0]) < min(p2[0], q2[0]) or
        max(p2[0], q2[0]) < min(p1[0], q1[0]) or
        max(p1[1], q1[1]) < min(p2[1], q2[1]) or
        max(p2[1], q2[1]) < min(p1[1], q1[1])):
        return -1,None
    
    o1 = orientation(p1, q1, p2, tol)
    o2 = orientation(p1, q1, q2, tol)
    o3 = orientation(p2, q2, p1, tol)
    o4 = orientation(p2, q2, q1, tol)

    if o1==0 and o2==0 and o3==0 and o4==0: #对共线的情况进行判断，若共线返回-1
        return -1,None

    if o1 != o2 and o3 != o4: #判断交叉（X型交叉）
        if on_segment(p1, p2, q1, tol)==False and on_segment(p1, q2, q1, tol)==False\
        and on_segment(p2, p1, q2, tol)==False and on_segment(p2, q1, q2, tol)==False: #确定无交点
            return 0,None  # 真正相交（内部相交）
    
    '''
    #判断T型交叉
    if (o1 == 0 and on_segment(p1, p2, q1, tol)) or (o2 == 0 and on_segment(p1, q2, q1, tol)): #线2交点在线1内部
        #print(f"{p2},{q2} is interior of {p1} {q1}")
        return 1
    if (o3 == 0 and on_segment(p2, p1, q2, tol)) or (o4 == 0 and on_segment(p2, q1, q2, tol)): #线1交点在线2内部
        return 2
    '''

    #判断T型交叉
    if (o1 == 0 and on_segment(p1, p2, q1, tol)):
        return 1,p2
    elif (o2 == 0 and on_segment(p1, q2, q1, tol)):
        return 1,q2
    elif (o3 == 0 and on_segment(p2, p1, q2, tol)): #线1交点在线2内部
        return 2,p1
    elif (o4 == 0 and on_segment(p2, q1, q2, tol)):
        return 2,q1
    return -1,None

def judge(split_line:LineString,other_line:LineString,tol):
    p1,q1,p2,q2=split_line.coords[0],split_line.coords[1],\
        other_line.coords[0],other_line.coords[1]
    #print(p1,q1,p2,q2)
    return is_cross_or_intersect(p1,q1,p2,q2,tol)

if __name__=="__main__":
    p1,q1,p2,q2=(0.1,0.1),(0.3,0.1),(0.2,0.1),(0.2,0.3)
    print(is_cross_or_intersect(p1,q1,p2,q2,1e-3))