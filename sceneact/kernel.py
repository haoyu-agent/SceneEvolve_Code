from dataclasses import dataclass
import numpy as np


def vector(value):
    result = np.array(value, dtype=float).reshape(-1)
    if result.size != 3 or not np.isfinite(result).all():
        raise ValueError('A finite three-coordinate value is required')
    return result


def direction(value):
    if not isinstance(value, str):
        return vector(value)
    token = value.strip().lower()
    if token not in ('x','y','z','+x','+y','+z','-x','-y','-z'):
        raise ValueError(f'Invalid cardinal direction: {value}')
    return np.eye(3)['xyz'.index(token[-1])] * (-1 if token.startswith('-') else 1)


def cardinal(value):
    axis = direction(value)
    occupied = np.flatnonzero(axis)
    if len(occupied) != 1:
        raise ValueError('Expected a cardinal direction')
    index = int(occupied[0])
    return index, float(np.sign(axis[index]))


def unit(value):
    result = direction(value)
    length = np.linalg.norm(result)
    if length < 1e-12:
        raise ValueError('Direction must have positive length')
    return result / length


def affine(linear=None, displacement=(0, 0, 0)):
    result = np.eye(4)
    if linear is not None:
        result[:3, :3] = np.asarray(linear, dtype=float).reshape(3, 3)
    result[:3, 3] = vector(displacement)
    return result


def shift_frame(offset):
    return affine(displacement=offset)


def stretch_frame(factors, pivot=(0, 0, 0)):
    scale = np.repeat(float(factors), 3) if np.isscalar(factors) else vector(factors)
    linear = np.diag(scale)
    point = vector(pivot)
    return affine(linear, point - linear @ point)


def spin_frame(axis, degrees, pivot=(0, 0, 0)):
    # Rodrigues expressed through a cross-product operator, rather than entries.
    normal = unit(axis)
    skew = np.column_stack([np.cross(normal, basis) for basis in np.eye(3)])
    angle = np.deg2rad(float(degrees))
    rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
    point = vector(pivot)
    return affine(rotation, point - rotation @ point)


def checked_frame(value):
    matrix = np.array(value, dtype=float).reshape(4, 4)
    if not np.isfinite(matrix).all() or not np.allclose(matrix[3], (0, 0, 0, 1)):
        raise ValueError('Expected a finite affine matrix')
    return matrix


@dataclass(frozen=True)
class Envelope:
    low: np.ndarray
    high: np.ndarray

    @classmethod
    def vacant(cls):
        return cls(np.full(3, np.inf), np.full(3, -np.inf))

    @property
    def empty(self):
        return bool(np.any(self.low > self.high))

    @property
    def midpoint(self):
        return (self.low + self.high) * .5

    @property
    def dimensions(self):
        return self.high - self.low

    def mapped(self, frame):
        if self.empty: return self
        center = frame[:3, :3] @ self.midpoint + frame[:3, 3]
        half = abs(frame[:3, :3]) @ (self.dimensions * .5)
        return Envelope(center - half, center + half)

    def combined(self, other, intersection=False):
        if intersection:
            low, high = np.maximum(self.low, other.low), np.minimum(self.high, other.high)
            return Envelope(low, high) if np.all(low <= high) else Envelope.vacant()
        return Envelope(np.minimum(self.low, other.low), np.maximum(self.high, other.high))

    def extreme(self, heading):
        return np.where(heading > 1e-12, self.high, np.where(heading < -1e-12, self.low, self.midpoint))


@dataclass(frozen=True)
class Finish:
    rgba: tuple = (1., 1., 1., 1.)

    @classmethod
    def rgb(cls, color, opacity=1.):
        channels = tuple(color)
        rgb = tuple(float(np.clip(channels[i] if i < len(channels) else 1, 0, 1)) for i in range(3))
        return cls((*rgb, float(np.clip(1 if opacity is None else opacity, 0, 1))))


@dataclass(frozen=True)
class Brick:
    dimensions: tuple
    origin: tuple

    def envelope(self):
        center, half = vector(self.origin), vector(self.dimensions) * .5
        return Envelope(center - half, center + half)

    def farthest(self, heading):
        return self.envelope().extreme(heading)


@dataclass(frozen=True)
class Orb:
    radius: float
    origin: tuple

    def envelope(self):
        return Envelope(vector(self.origin) - self.radius, vector(self.origin) + self.radius)

    def farthest(self, heading):
        length = np.linalg.norm(heading)
        return vector(self.origin) + (heading * (self.radius / length) if length >= 1e-12 else 0)


@dataclass(frozen=True)
class Bar:
    radius: float
    start: tuple
    end: tuple

    def envelope(self):
        a, b = vector(self.start), vector(self.end)
        length = np.linalg.norm(b - a)
        pad = self.radius * np.sqrt(np.clip(1 - ((b-a) / length)**2, 0, 1)) if length > 1e-9 else self.radius
        return Envelope(np.minimum(a,b) - pad, np.maximum(a,b) + pad)

    def farthest(self, heading):
        a, b = vector(self.start), vector(self.end)
        length = np.linalg.norm(b-a)
        if length < 1e-12:
            return Orb(self.radius, self.start).farthest(heading)
        tangent = (b-a) / length
        axial = float(heading @ tangent)
        radial = heading - axial * tangent
        norm = np.linalg.norm(radial)
        return (b if axial >= 0 else a) + (radial * self.radius / norm if norm >= 1e-12 else 0)
