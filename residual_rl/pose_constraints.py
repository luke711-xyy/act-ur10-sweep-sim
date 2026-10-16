"""Small deterministic convex projections; no simulator or native control IO."""
import numpy as np


def disk(center,radius):
    center=np.asarray(center,dtype=float)
    def project(point):
        delta=point-center; length=np.linalg.norm(delta)
        return center+delta*min(1.,radius/length) if length else point.copy()
    project.boundary=('circle',center,float(radius))
    return project


def halfplane(normal,upper):
    normal=np.asarray(normal,dtype=float); squared=float(normal@normal)
    def project(point):
        return point-normal*max(0.,float(normal@point)-upper)/squared
    project.boundary=('line',normal,float(upper))
    return project


def boundary_intersections(first,second):
    a,p,r=first; b,q,s=second
    if a=='line' and b=='line':
        matrix=np.stack([p,q])
        return [] if abs(np.linalg.det(matrix))<1e-14 else [np.linalg.solve(matrix,[r,s])]
    if a=='line': return boundary_intersections(second,first)
    if b=='line':
        squared=float(q@q); signed=(float(q@p)-s)/np.sqrt(squared)
        height=r*r-signed*signed
        if height < -1e-16: return []
        foot=p-q*(float(q@p)-s)/squared
        tangent=np.array([-q[1],q[0]])/np.sqrt(squared)
        return [foot+sign*np.sqrt(max(0.,height))*tangent for sign in (-1.,1.)]
    delta=q-p; distance=float(np.linalg.norm(delta))
    if distance<1e-14 or distance>r+s+1e-12 or distance<abs(r-s)-1e-12: return []
    along=(r*r-s*s+distance*distance)/(2*distance)
    height=r*r-along*along
    if height < -1e-16: return []
    direction=delta/distance; middle=p+direction*along
    tangent=np.array([-direction[1],direction[0]])
    return [middle+sign*np.sqrt(max(0.,height))*tangent for sign in (-1.,1.)]


def intersection(point,projections):
    """Exact 2-D active-boundary enumeration, including externally tangent disks.

    A nearest point is interior, on one boundary, or at a pair intersection.
    This avoids falsely rejecting feasible nearly tangent disks on an iteration cap.
    """
    point=np.asarray(point,dtype=float)
    def feasible(value):
        return np.isfinite(value).all() and all(np.linalg.norm(value-p(value))<=1e-10 for p in projections)
    if feasible(point): return point.copy()
    single=[p(point) for p in projections]
    possible=[p for p in single if feasible(p)]
    if possible: return min(possible,key=lambda p:np.linalg.norm(p-point)).copy()
    for i,first in enumerate(projections):
        for second in projections[i+1:]:
            possible.extend(p for p in boundary_intersections(first.boundary,second.boundary) if feasible(p))
    if not possible: raise RuntimeError('pose offset constraints infeasible')
    return min(possible,key=lambda p:np.linalg.norm(p-point)).copy()
