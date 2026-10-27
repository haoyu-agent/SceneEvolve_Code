from copy import deepcopy
from dataclasses import asdict, dataclass
from types import MappingProxyType
import numpy as np
from .kernel import Finish, affine, checked_frame, shift_frame, stretch_frame, spin_frame


class Asset:
    def __init__(self, name='Asset', *, solid=None, finish=None, operation=None):
        self.name = name
        self.solid = solid
        self.finish = finish or Finish()
        self.operation = operation
        self._frame = affine()
        self._body_frame = affine()
        self._owner = None
        self._members = {}

    @property
    def members(self):
        return MappingProxyType(self._members)

    @property
    def world_frame(self):
        return self._frame.copy() if self._owner is None else self._owner.world_frame @ self._frame

    @property
    def geometry_frame(self):
        return self.world_frame @ self._body_frame

    def _release_frame(self):
        """Move a collapsed parent frame to its contents before editing a child."""
        if self._owner is not None:
            self._owner._release_frame()
        self._body_frame = self._frame @ self._body_frame
        for child in self._members.values():
            child._frame = self._frame @ child._frame
        self._frame = affine()

    def traverse(self):
        yield self
        for node in self._members.values():
            yield from node.traverse()

    def _localize(self, world):
        if self._owner is None:
            self._frame = world.copy()
        else:
            try:
                self._frame = np.linalg.solve(self._owner.world_frame, world)
            except np.linalg.LinAlgError:
                self._owner._release_frame()
                self._frame = world.copy()

    def include(self, name, node):
        if not isinstance(node, Asset):
            raise TypeError('Only scene nodes can be attached')
        if self in node.traverse():
            raise ValueError('A hierarchy cannot contain cycles')
        if node._owner is not None and node._owner is not self:
            node = node.clone()
        world = node.world_frame
        try:
            local = np.linalg.solve(self.world_frame, world)
        except np.linalg.LinAlgError:
            self._release_frame()
            local = world.copy()
        for key, member in tuple(self._members.items()):
            if key == name or member is node:
                self.remove(key)
        self._members[name] = node
        node._owner, node._frame = self, local
        return node

    def remove(self, name):
        node = self._members.get(name)
        if node is None: return None
        world = node.world_frame
        removed = set(node.traverse())
        owner = self
        while owner is not None:
            if isinstance(owner, Scene):
                owner._bindings = [rule for rule in owner._bindings if rule.source not in removed and rule.target not in removed]
            owner = owner._owner
        del self._members[name]
        node._owner, node._frame = None, world
        return node

    def clone(self):
        world = self.world_frame
        result = deepcopy(self, {id(self._owner): None} if self._owner is not None else {})
        result._owner, result._frame = None, world
        return result

    def apply(self, matrix):
        self._localize(checked_frame(matrix) @ self.world_frame)
        return self

    def move(self, offset):
        return self.apply(shift_frame(offset))

    def turn(self, axis=None, degrees=None, *, pivot=None, euler=None):
        center = self.centroid if pivot is None else pivot
        if euler is not None:
            if axis is not None or degrees is not None:
                raise TypeError('Use either Euler angles or axis/angle')
            from .kernel import vector
            angles = vector(euler)
            matrix = affine()
            for basis, angle in zip(np.eye(3), angles):
                matrix = spin_frame(basis, angle, center) @ matrix
            return self.apply(matrix)
        if axis is None or degrees is None:
            raise TypeError('Rotation needs an axis and angle')
        return self.apply(spin_frame(axis, degrees, center))

    def resize(self, factors, *, pivot=None):
        return self.apply(stretch_frame(factors, self.centroid if pivot is None else pivot))

    def paint(self, color=(1,1,1), *, opacity=1):
        finish = Finish.rgb(color, opacity)
        for node in self.traverse(): node.finish = finish
        return self

    @property
    def rgba(self):
        return self.finish.rgba

    @property
    def centroid(self):
        from .relations import envelope
        return envelope(self).midpoint

    @property
    def dimensions(self):
        from .relations import envelope
        return envelope(self).dimensions

    @property
    def limits(self):
        from .relations import envelope
        bounds = envelope(self)
        return bounds.low, bounds.high

    def landmark(self, name='center'):
        from .relations import landmark
        return landmark(self, name)

    def extreme(self, heading):
        from .relations import support_point
        return support_point(self, heading)

    def describe(self):
        nodes = list(self.traverse())
        indexes = {id(node): i for i, node in enumerate(nodes)}
        result = []
        for node in nodes:
            body = None if node.solid is None else {'kind': type(node.solid).__name__, **asdict(node.solid)}
            result.append({'name': node.name, 'role': type(node).__name__, 'body': body,
                           'operation': node.operation, 'finish': list(node.rgba),
                           'frame': (node.world_frame if node is self else node._frame).tolist(),
                           'body_frame': node._body_frame.tolist(),
                           'members': {k: indexes[id(v)] for k,v in node._members.items()}})
        constraints = []
        for node in nodes:
            if isinstance(node, Scene):
                for rule in node._bindings:
                    constraints.append({'scene': indexes[id(node)], 'source': indexes[id(rule.source)],
                        'target': indexes[id(rule.target)], 'source_anchor': rule.source_anchor,
                        'target_anchor': rule.target_anchor, 'offset': list(rule.offset)})
        return {'format': 'sceneact.graph.v1', 'nodes': result, 'constraints': constraints}


class Part(Asset):
    def __init__(self, name='Part', *nodes):
        super().__init__(name)
        for i, node in enumerate(nodes): self.include(str(i), node)


@dataclass(frozen=True)
class Binding:
    source: Asset
    target: Asset
    source_anchor: str
    target_anchor: str
    offset: tuple


class Scene(Asset):
    def __init__(self, name='Scene'):
        super().__init__(name)
        self._bindings = []

    def remove(self, name):
        node = self._members.get(name)
        if node is not None:
            members = set(node.traverse())
            self._bindings = [r for r in self._bindings if r.source not in members and r.target not in members]
        return super().remove(name)

    def replicate(self, name, new_name):
        return self.include(new_name, self._members[name].clone())

    def bind(self, source, target, source_anchor='center', target_anchor='center', offset=(0,0,0)):
        from .kernel import vector
        members = set(self.traverse())
        if source not in members or target not in members:
            raise ValueError('Constraint endpoints must belong to this scene')
        if source is target or target in source.traverse() or source in target.traverse():
            raise ValueError('Constraint endpoints cannot be ancestor and descendant')
        rule = Binding(source, target, source_anchor, target_anchor, tuple(vector(offset)))
        previous = self._bindings[:]
        self._bindings.append(rule)
        try: self.settle()
        except Exception:
            self._bindings = previous
            raise
        return source

    def settle(self):
        from graphlib import TopologicalSorter, CycleError
        from .relations import landmark
        dependencies = {i: set() for i in range(len(self._bindings))}
        for i, rule in enumerate(self._bindings):
            for j, other in enumerate(self._bindings):
                if i != j and rule.target in other.source.traverse():
                    dependencies[i].add(j)
        try: order = list(TopologicalSorter(dependencies).static_order())
        except CycleError as exc: raise ValueError('Cyclic spatial constraints') from exc
        root = self
        while root._owner is not None: root = root._owner
        frames = {node: (node._frame.copy(),node._body_frame.copy()) for node in root.traverse()}
        try:
            for i in order:
                rule = self._bindings[i]
                displacement = landmark(rule.target, rule.target_anchor) + rule.offset - landmark(rule.source, rule.source_anchor)
                rule.source.move(displacement)
        except Exception:
            for node, (frame,body_frame) in frames.items():
                node._frame, node._body_frame = frame,body_frame
            raise
        return self


def assemble(nodes, name='Assembly'):
    parent = Asset(name)
    for i, node in enumerate(nodes): parent.include(str(i), node)
    return parent


def restore(document):
    from .kernel import Brick, Orb, Bar
    if document.get('format') != 'sceneact.graph.v1': raise ValueError('Unknown graph format')
    entries = document['nodes']
    if not entries: raise ValueError('Graph has no root')
    classes = {'Asset': Asset, 'Part': Part, 'Scene': Scene}
    solids = {'Brick': Brick, 'Orb': Orb, 'Bar': Bar}
    nodes = []
    for item in entries:
        node = classes[item['role']](item['name'])
        body = item['body']
        node.solid = solids[body['kind']](**{k:tuple(v) if isinstance(v,list) else v for k,v in body.items() if k != 'kind'}) if body else None
        node.finish = Finish(tuple(item['finish']))
        node.operation = item['operation']
        node._frame = checked_frame(item['frame'])
        node._body_frame = checked_frame(item.get('body_frame',affine()))
        nodes.append(node)
    assigned = {0}
    for i, item in enumerate(entries):
        for name, child_index in item['members'].items():
            if child_index <= i or child_index >= len(nodes) or child_index in assigned:
                raise ValueError('Invalid graph edge')
            assigned.add(child_index)
            nodes[i]._members[name] = nodes[child_index]
            nodes[child_index]._owner = nodes[i]
    if len(assigned) != len(nodes): raise ValueError('Unreachable graph nodes')
    for rule in document.get('constraints', []):
        scene = nodes[rule['scene']]
        if not isinstance(scene, Scene): raise ValueError('Constraint owner is not a scene')
        scene._bindings.append(Binding(nodes[rule['source']], nodes[rule['target']], rule['source_anchor'],rule['target_anchor'],tuple(rule['offset'])))
    return nodes[0]
