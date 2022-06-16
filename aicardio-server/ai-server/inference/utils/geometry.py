import math
import numpy as np


__all__ = ["line_eq", "scaled_length", "cosine", "dist_to_line",
           "scale_polygon", "intersection"]


def line_eq(p1, p2):
    # Ax + By = C
    A = p2[1]-p1[1]
    B = p1[0]-p2[0]
    C = B*p1[1]+A*p1[0]
    return A,B,C

def scaled_length(p, q, xs=1, ys=1):
    dx = (p[0]-q[0])*xs
    dy = (p[1]-q[1])*ys
    return math.hypot(dx, dy)

def cosine(p1, p2, p3, p4):
    r"""Compute cos between vector (p2 - p1) and vector (p4 - p3)"""
    inner_product = np.abs(np.sum((p2 - p1) * (p4 - p3)))
    length_product = scaled_length(p1, p2) * scaled_length(p3, p4)
    cos =  inner_product / length_product 
    return cos

def dist_to_line(p, line_coeff):
    A, B, C = line_coeff
    return np.abs(A*p[0] + B*p[1] - C) / np.sqrt(A**2 + B**2)

def scale_polygon(polygon, xs=1.0, ys=1.0):
    polygon = polygon.astype(np.float32).copy()
    polygon[:, :, 0] = polygon[:, :, 0]*xs
    polygon[:, :, 1] = polygon[:, :, 1]*ys
    return polygon

def intersection(p1, p2, q1, q2):
    A1,B1,C1 = line_eq(p1, p2)
    A2,B2,C2 = line_eq(q1, q2)
    det = A1*B2-A2*B1
    if abs(det) < 1e-6: 
        return None
    ip = np.array([(B2*C1-B1*C2)/det, (A1*C2-A2*C1)/det], dtype=np.float)
    
    if (ip[0]-q1[0])*(ip[0]-q2[0]) <= 0 and (ip[1]-q1[1])*(ip[1]-q2[1]) <= 0:
        return ip
    else:
        return None

def interesection_of_line_and_hull(p1, p2, hull):
    n = len(hull)
    ret = []
    for i in range(n):
        q1, q2 = hull[i], hull[(i+1)%n]
        ip = intersection(p1,p2,q1,q2)
        if ip is not None: 
            ret.append(ip)
    return np.array(ret)