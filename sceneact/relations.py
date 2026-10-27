import numpy as np
from .kernel import Envelope, vector, unit, cardinal, direction
from .composition import Asset, assemble


def envelope(node):
    if node.solid is not None:
        result = node.solid.envelope().mapped(node.geometry_frame)
    else:
        result = Envelope.vacant()
    operands = list(node.members.values())
    if node.operation and operands:
        result = envelope(operands[0])
        if node.operation == 'difference': return result
        for operand in operands[1:]:
            result = result.combined(envelope(operand),intersection=node.operation=='intersection')
    else:
        for child in operands: result = result.combined(envelope(child))
    return result


def support_point(node, heading):
    normal = unit(heading)
    candidates = []
    children = list(node.members.values())
    if node.operation == 'intersection':
        bounds = envelope(node)
        if not bounds.empty: candidates.append(bounds.extreme(normal))
    elif node.operation == 'difference' and children:
        candidates.append(support_point(children[0],normal))
    else:
        if node.solid is not None:
            pose = node.geometry_frame
            local = node.solid.farthest(pose[:3,:3].T @ normal)
            candidates.append(pose[:3,:3] @ local + pose[:3,3])
        candidates.extend(support_point(child,normal) for child in children if not envelope(child).empty)
    if not candidates: return np.full(3,np.nan)
    # Stable argmax keeps the first primitive on equal support, including ties.
    return candidates[int(np.argmax(np.asarray(candidates) @ normal))]


def projected_span(node, heading):
    axis = unit(heading)
    return float(support_point(node,-axis) @ axis), float(support_point(node,axis) @ axis)


def projected_width(node, heading):
    low,high = projected_span(node,heading)
    return high-low


def landmark(node, name='center'):
    if not isinstance(name,str): raise TypeError('Landmark must be a name')
    words = name.strip().lower().replace('_',' ').split()
    if not words: raise ValueError('Empty landmark')
    ends = (('left','right'),('front','back'),('bottom','top'))
    fractions = np.full(3,.5)
    for i,pair in enumerate(ends):
        active = set(words).intersection(pair)
        if len(active)>1: raise ValueError('Conflicting landmark directions')
        if active: fractions[i] = float(pair[1] in active)
    valid = {'center','centre','middle',*(word for pair in ends for word in pair)}
    if set(words)-valid: raise ValueError(f'Unknown landmark: {name}')
    bounds = envelope(node)
    return bounds.low + fractions * bounds.dimensions


def relocate(node, matrix):
    return node.clone().apply(matrix)


def shifted(node, offset):
    return node.clone().move(offset)


def resized(node, factors, *, pivot=None):
    return node.clone().resize(factors,pivot=pivot)


def turned(node, axis=None, degrees=None, *, pivot=None, euler=None):
    return node.clone().turn(axis,degrees,pivot=pivot,euler=euler)


def snap(node, target, source_anchor='center', target_anchor=None, offset=(0,0,0)):
    if isinstance(target,Asset):
        point = landmark(target,target_anchor or source_anchor)
    else:
        if target_anchor is not None: raise TypeError('A point target has no named landmark')
        point = vector(target)
    return shifted(node,point+vector(offset)-landmark(node,source_anchor))


def abut(node, target, axis='+z', gap=0.):
    if gap < 0: raise ValueError('Gap must be nonnegative')
    index,sign = cardinal(axis)
    source = envelope(node)
    if isinstance(target,Asset):
        bound = envelope(target)
        coordinate = bound.high[index] if sign>0 else bound.low[index]
    else: coordinate = float(target)
    edge = source.low[index] if sign>0 else source.high[index]
    displacement = np.zeros(3)
    displacement[index] = coordinate + sign*gap - edge
    return shifted(node,displacement)


def center_on(node, target, axes=('x','y','z')):
    displacement = np.zeros(3)
    for axis in axes:
        index,_ = cardinal(axis)
        displacement[index] = target.centroid[index]-node.centroid[index]
    return shifted(node,displacement)


def offset_at(node, reference, offsets):
    values = list(offsets)
    base = reference.centroid if isinstance(reference,Asset) else ((0,0,0) if reference is None else reference)
    if len(values)!=3 or len(base)!=3: raise ValueError('Expected three coordinates')
    target = node.centroid.copy()
    for i,(origin,delta) in enumerate(zip(base,values)):
        if origin is not None and delta is not None: target[i]=origin+delta
    return shifted(node,target-node.centroid)


def _place(nodes, targets, angles=None, normal='+z'):
    outputs=[]
    for i,(node,target) in enumerate(zip(nodes,targets)):
        copy = node.clone()
        if angles is not None: copy.turn(normal,angles[i])
        copy.move(target-copy.centroid)
        outputs.append(copy)
    return assemble(outputs)


def _linear(nodes, axis, distance, surfaces):
    items=list(nodes)
    if not items: return assemble([])
    if distance<0: raise ValueError('Spacing must be nonnegative')
    index,sign=cardinal(axis)
    centers=np.array([node.centroid for node in items])
    increments=np.full(len(items)-1,float(distance))
    if surfaces:
        widths=np.array([node.dimensions[index] for node in items])
        increments+=(widths[:-1]+widths[1:])/2
    centers[1:,index]=centers[0,index]+sign*np.cumsum(increments)
    return _place(items,centers)


def repeat_line(nodes, axis='+x', spacing=1.):
    return _linear(nodes,axis,spacing,False)


def pile(nodes, axis='+z', gap=0.):
    return _linear(nodes,axis,gap,True)


def lattice(nodes, rows=None, columns=None, spacing=(1.,1.), plane='xy', center=(0,0,0), order='row-major'):
    items=list(nodes)
    if not items: return assemble([])
    count=len(items)
    if rows is None and columns is None: rows,columns=1,count
    elif rows is None:
        if columns<=0: raise ValueError('Columns must be positive')
        rows=int(np.ceil(count/int(columns)))
    elif columns is None:
        if rows<=0: raise ValueError('Rows must be positive')
        columns=int(np.ceil(count/int(rows)))
    rows,columns=int(rows),int(columns)
    if min(rows,columns)<=0 or rows*columns<count: raise ValueError('Grid cannot hold all nodes')
    axes=[c for c in plane if c in 'xyz']
    if len(axes)!=2 or len(set(axes))!=2: raise ValueError('Grid plane requires two axes')
    u,v=('xyz'.index(c) for c in axes)
    step=np.repeat(float(spacing),2) if np.isscalar(spacing) else np.array(spacing,dtype=float)
    if step.shape!=(2,): raise ValueError('Grid spacing needs two components')
    if order not in ('row-major','column-major'): raise ValueError('Unknown grid traversal')
    indices=np.arange(count)
    row,col=(indices//columns,indices%columns) if order=='row-major' else (indices%rows,indices//rows)
    targets=np.tile(vector(center),(count,1))
    targets[:,u]+=(col-(columns-1)/2)*step[0]
    targets[:,v]+=((rows-1)/2-row)*step[1]
    return _place(items,targets)


def orbit(nodes, radius, axis='+z', center=(0,0,0), start=0., sweep=360., *, orient=False, rotation_offset=0.):
    items=list(nodes)
    if not items: return assemble([])
    if radius<0: raise ValueError('Radius must be nonnegative')
    index,_=cardinal(axis)
    normal=direction(axis)
    first=np.eye(3)[(index+1)%3]
    second=np.cross(normal,first)
    closed=np.isclose(abs(sweep),360,rtol=1e-9,atol=1e-9)
    angles=np.linspace(start,start+sweep,len(items),endpoint=not closed) if len(items)>1 else np.array([start])
    radians=np.deg2rad(angles)
    targets=vector(center)+radius*(np.cos(radians)[:,None]*first+np.sin(radians)[:,None]*second)
    return _place(items,targets,angles+rotation_offset if orient else None,normal)
