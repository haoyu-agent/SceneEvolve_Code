from itertools import product, combinations
import numpy as np
from .kernel import Brick, Orb, Bar, Finish, vector, unit
from .composition import Asset, assemble


def _positive(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0: raise ValueError('Dimension must be finite and positive')
    return number


def _node(body, color, opacity, name):
    return Asset(name, solid=body, finish=Finish.rgb(color, opacity))


def box(size=1., center=(0,0,0), color=(1,1,1), opacity=1.):
    extent = np.repeat(float(size),3) if np.isscalar(size) else vector(size)
    dimensions = tuple(_positive(v) for v in extent)
    return _node(Brick(dimensions, tuple(vector(center))), color, opacity, 'Box')


def ball(radius, center=(0,0,0), color=(1,1,1), opacity=1.):
    return _node(Orb(_positive(radius), tuple(vector(center))), color, opacity, 'Ball')


def rod(radius, *, start=None, end=None, length=None, center=(0,0,0), axis='+z', color=(1,1,1), opacity=1.):
    if start is None and end is None:
        if length is None: raise TypeError('Supply endpoints or length')
        half = unit(axis) * _positive(length) / 2
        a, b = vector(center) - half, vector(center) + half
    elif start is None or end is None or length is not None:
        raise TypeError('Supply both endpoints, without length')
    else:
        a, b = vector(start), vector(end)
    return _node(Bar(_positive(radius),tuple(a),tuple(b)),color,opacity,'Rod')


def fillet_box(size, radius, center=(0,0,0), color=(1,1,1), opacity=1.):
    dimensions = np.repeat(float(size),3) if np.isscalar(size) else vector(size)
    for value in dimensions: _positive(value)
    r = _positive(radius)
    if r > min(dimensions)/2: raise ValueError('Fillet exceeds half the smallest dimension')
    inner = dimensions - 2*r
    signs = np.array(list(product((-1,1), repeat=3)))
    corners = vector(center) + signs * inner/2
    pieces = [ball(r,point,color,opacity) for point in corners]
    for i,j in combinations(range(8),2):
        if np.count_nonzero(signs[i] != signs[j]) == 1 and np.linalg.norm(corners[i]-corners[j]) > 1e-12:
            pieces.append(rod(r,start=corners[i],end=corners[j],color=color,opacity=opacity))
    slabs = np.tile(inner,(3,1))
    np.fill_diagonal(slabs,dimensions)
    pieces.extend(box(s,center,color,opacity) for s in slabs if min(s)>0)
    return assemble(pieces,'FilletBox')


def sweep_tube(points, radius=.03, color=(1,1,1), opacity=1.):
    vertices = [vector(p) for p in points]
    if len(vertices)<2: raise ValueError('A tube needs at least two vertices')
    segments = [rod(radius,start=a,end=b,color=color,opacity=opacity) for a,b in zip(vertices,vertices[1:])]
    caps = [ball(radius,p,color,opacity) for p in vertices]
    return assemble(segments+caps,'Tube')


def _combine(operation, nodes):
    if not nodes: return Asset(operation)
    if len(nodes)==1: return nodes[0]
    result = assemble(nodes,operation)
    result.operation = operation
    return result


def fuse(*nodes):
    return _combine('union',nodes)


def overlap(*nodes):
    return _combine('intersection',nodes)


def subtract(base,*cutters):
    return _combine('difference',(base,*cutters))


def exclusive(left,right):
    return fuse(subtract(left.clone(),right.clone()),subtract(right.clone(),left.clone()))
