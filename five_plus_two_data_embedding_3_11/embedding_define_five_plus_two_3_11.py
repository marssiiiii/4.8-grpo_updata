from dataclasses import dataclass, field
from shapely.geometry import LineString,Point
from typing import List, Set, Any
import numpy as np
from functools import cached_property
from shapely.geometry import LineString
import math

@dataclass
class StructuralSegment:
    def __init__(self,floor:str,type:str,general_type:str,attribute_type:str=None,build_order:int=-1,point:Point=None,line_string:LineString=None,polygon:list[Point]=None):
        self.floor=floor
        self.type=type
        self.general_type=general_type
        self.attribute_type=attribute_type
        self.line_string=line_string
        self.polygon=polygon
        self.point=point
        self.build_order=build_order
    def __hash__(self):
        return id(self)
    def __eq__(self,other):
        return self is other

class Segment_Embedding:
    def __init__(self,segments:List[StructuralSegment],grid_size:int=256):
        self.segments = segments
        self.grid_size = grid_size
        self.xmin, self.ymin, self.xmax, self.ymax=0,0,1,1
    
    @classmethod
    def tokenized_LOC(cls,val):
        val=int(val)
        return f"<|LOC_{val}|>" 
    
    def _normalize(self, x, y):
        """将坐标映射到 [0, grid_size-1]"""
        xn = (x - self.xmin) / (self.xmax - self.xmin) * (self.grid_size - 1)
        yn = (y - self.ymin) / (self.ymax - self.ymin) * (self.grid_size - 1)
        return int(round(xn)), int(round(yn))
    
    def power_of_two_exponent(self,n: int) -> int:
        if n <= 0:
            raise ValueError("n 必须是正整数")
        if n & (n - 1) != 0:  # 判断是否为2的幂
            raise ValueError(f"{n} 不是2的幂")
        return int(math.log2(n))
    
    def encode_line(self, seg: StructuralSegment):
        x, y = seg.line_string.xy
        x1, y1 = x[0], y[0]
        x2, y2 = x[-1], y[-1]
        if seg.attribute_type=="exterior":
            code=[seg.floor,f"{seg.attribute_type}_{seg.type}",seg.build_order,Point(x1,y1),Point(x2,y2)]
        else:
            code=[seg.floor,seg.type,seg.build_order,Point(x1,y1),Point(x2,y2)]
        return code
    
    def encode_polygon(self, seg: StructuralSegment):
        points = seg.polygon
        code=[seg.floor,seg.type,seg.build_order]
        for p in points:
            code.append(p)
        return code
    
    def encode_point(self, seg: StructuralSegment):
        point=seg.point
        code=[seg.floor,seg.type,seg.build_order]
        code.append(point)
        return code
    
    def encode_all(self) -> List[str]:
        res=[]
        for seg in self.segments:
            if seg.general_type=="line":
                res.append(self.encode_line(seg))
            elif seg.general_type=="poly":
                res.append(self.encode_polygon(seg))
            elif seg.general_type=="point":
                res.append(self.encode_point(seg))
        return res

'''
def main():
    segments = [
    StructuralSegment("1F", LineString([(0, 0), (10, 10)]), "NE", {"wall"}),
    StructuralSegment("1F", LineString([(5, 5), (15, 10)]), "E", {"beam"})
    ]

    encoder = Segment_Embedding(segments,initial_bounds=True)
    bounds=encoder._compute_bounds()
    print(bounds)
    codes = encoder.encode_all()

    for i, code in enumerate(codes):
        print(f"Segment {i}: {code}")

main()
'''