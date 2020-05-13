import math
from typing import Tuple, Union, Iterable
from game_settings import GameSettings


def is_on_screen(point: Tuple[float, float], margin: Union[float, Tuple[float, float]] = 0.0) -> bool:
	w, h = GameSettings.screen_size
	m = (margin, margin) if not isinstance(margin, tuple) else margin
	return -m[0] <= math.floor(point[0]) < w+m[0] and -m[1] <= math.floor(point[1]) < h+m[1]


def range_check_coords(coords: Tuple[float, float], min: Tuple[float, float], max: Tuple[float, float]) -> bool:
	return min[0] <= coords[0] < max[0] and min[1] <= coords[1] < max[1]


def indicator_location(ship_location: Tuple[float, float], margin: int = 10) -> Tuple[int, int, float]:
	indicator_x = 0
	indicator_y = 0
	ship_direction = math.atan2(ship_location[1] - GameSettings.screen_size[1]/2, ship_location[0]-GameSettings.screen_size[0]/2)
	if GameSettings.screen_side_angles[0] <= ship_direction <= GameSettings.screen_side_angles[1]:
		indicator_x = GameSettings.screen_size[0] - margin
		indicator_y = math.tan(ship_direction) * GameSettings.screen_size[0] / 2 + GameSettings.screen_size[1] / 2 - margin
	elif GameSettings.screen_side_angles[1] <= ship_direction <= GameSettings.screen_side_angles[2]:
		indicator_y = GameSettings.screen_size[1] - margin
		indicator_x = 1/math.tan(ship_direction) * GameSettings.screen_size[1] / 2 + GameSettings.screen_size[0] / 2 - margin
	elif GameSettings.screen_side_angles[2] <= ship_direction <= 2*math.pi or -2*math.pi <= ship_direction <= GameSettings.screen_side_angles[3]:
		indicator_x = margin
		indicator_y = -math.tan(ship_direction) * GameSettings.screen_size[0] / 2 + GameSettings.screen_size[1] / 2 + margin
	elif GameSettings.screen_side_angles[3] <= ship_direction <= GameSettings.screen_side_angles[0]:
		indicator_y = margin
		indicator_x = -1/math.tan(ship_direction) * GameSettings.screen_size[1] / 2 + GameSettings.screen_size[0] / 2 + margin

	return (int(round(indicator_x)), int(round(indicator_y)), ship_direction)


def rotation_matrix(direction: Union[float, Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
	if isinstance(direction, tuple):
		dv_length = math.hypot(direction[0], direction[1])
		dv_unitvec = direction[0] / dv_length, direction[1] / dv_length
		return (
			(dv_unitvec[0], -dv_unitvec[1]),
			(dv_unitvec[1], dv_unitvec[0])
		)
	else:
		return (
			(math.cos(direction), -math.sin(direction)),
			(math.sin(direction), math.cos(direction))
		)


def scale_matrix(scale: Union[float, Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
	if isinstance(scale, tuple):
		return (
			(scale[0], 0),
			(0, scale[1])
		)
	else:
		return (
			(scale, 0),
			(0, scale)
		)


def mul_matrix(vector: Tuple[float, float], matrix: Tuple[Tuple[float, float], Tuple[float, float]] = ((1, 0), (0, 1))) -> Tuple[float, float]:
	return matrix[0][0] * vector[0] + matrix[0][1]*vector[1], matrix[1][0] * vector[0] + matrix[1][1]*vector[1]


def vec_plus(v1: Tuple[float, float], v2: Tuple[float, float]) -> Tuple[float, float]:
	return v1[0]+v2[0], v1[1]+v2[1]


def vec_minus(v1: Tuple[float, float], v2: Tuple[float, float]) -> Tuple[float, float]:
	return v1[0]-v2[0], v1[1]-v2[1]


def vec_mul(v: Tuple[float, float], multiplier: float) -> Tuple[float, float]:
	return v[0]*multiplier, v[1]*multiplier


def vec_len(v: Tuple[float, float]) -> float:
	return math.hypot(v[0], v[1])


def angle_wrap(angle: float) -> float:
	output = angle
	while output > 2*math.pi:
		output -= 2*math.pi
	while output < 0:
		output += 2*math.pi
	return output


def vec_angle(v: Tuple[float, float]) -> float:
	angle = math.atan2(v[1], v[0])
	return angle_wrap(angle)


def vec_unit(v: Tuple[float, float]) -> Tuple[float, float]:
	if v == (0, 0):
		return 1.0, 0.0
	return vec_mul(v, 1/vec_len(v))


def to_polar(cartesian: Tuple[float, float], midpoint: Union[Tuple[float, float], None] = None) -> Tuple[float, float]:
	if midpoint is None:
		return vec_len(cartesian), vec_angle(cartesian)
	else:
		offset_cart = vec_minus(cartesian, midpoint)
		return vec_len(offset_cart), vec_angle(offset_cart)


def to_cartesian(polar: Tuple[float, float], midpoint: Union[Tuple[float, float], None] = None) -> Tuple[float, float]:
	origo_based = math.cos(polar[1]) * polar[0], math.sin(polar[1]) * polar[0]
	if midpoint is None:
		return origo_based
	return vec_plus(origo_based, midpoint)


def vec_decompose(v: Tuple[float, float], direction: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
	dir_angle = vec_angle(direction)
	v_angle = vec_angle(v)
	alpha = math.fabs(min(dir_angle-v_angle, v_angle-dir_angle))
	cosine = math.cos(alpha)
	comp_indir_len = vec_len(v) * cosine
	comp_indir = vec_mul(vec_unit(direction), comp_indir_len)
	comp_tan = vec_minus(v, comp_indir)
	return comp_indir, comp_tan


def dot_product(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
	return v1[0]*v2[0] + v1[1]*v2[1]


def vec_int(v: Tuple[Union[float, int], Union[float, int]]) -> Tuple[int, int]:
	return int(round(v[0])), int(round(v[1]))


class AngleInterval:
	def __init__(self, min_angle: float, min_angle_idx: int, max_angle: Union[float, None] = None, max_angle_idx: Union[int, None] = None):
		self.min_angle = min_angle
		self.min_angle_idx = min_angle_idx
		self.max_angle = min_angle if max_angle is None else max_angle
		self.max_angle_idx = min_angle_idx if max_angle_idx is None else max_angle_idx
		
	def extend(self, added_angle: float, added_angle_idx: int):
		if abs(added_angle - self.max_angle) >= abs(added_angle - self.min_angle) and added_angle > self.max_angle:
			self.max_angle = added_angle
			self.max_angle_idx = added_angle_idx
		elif added_angle < self.min_angle:
			self.min_angle = added_angle
			self.min_angle_idx = added_angle_idx
	
	def ends_with(self, idx: int) -> bool:
		return self.min_angle_idx == idx or self.max_angle_idx == idx
		
	def contains(self, angle: float) -> bool:
		if self.min_angle > self.max_angle:
			# The minimum angle is wrapped around
			return angle >= self.min_angle
		else:
			return self.min_angle <= angle <= self.max_angle


def rotate_point(point: Tuple[float, float], direction: Union[float, Tuple[float, float]]) -> Tuple[float, float]:
	matrix = rotation_matrix(direction)
	return mul_matrix(point, matrix)


def rotate_polygon(points: Iterable[Tuple[float, float]], direction: Union[float, Tuple[float, float]] = (1, 0)) -> Tuple[Tuple[float, float]]:
	matrix = rotation_matrix(direction)
	return tuple(mul_matrix(p, matrix) for p in points)


def scale_polygon(points: Iterable[Tuple[float, float]], scale: Union[float, Tuple[float, float]]=1) -> Tuple[Tuple[float, float]]:
	matrix = scale_matrix(scale)
	return tuple(mul_matrix(p, matrix) for p in points)


def offset_polygon(points: Iterable[Tuple[float, float]], position: Tuple[float, float] = (1, 0)) -> Tuple[Tuple[float, float]]:
	return tuple((x+position[0], y+position[1]) for x, y in points)


def transform_polygon(
		points: Iterable[Tuple[float, float]],
		scale: Union[float, Tuple[float, float]] = 1,
		rotation: Union[float, Tuple[float, float]] = 0,
		translation: Tuple[float, float] = (0, 0)) -> Tuple[Tuple[float, float]]:
	return offset_polygon(rotate_polygon(scale_polygon(points, scale), rotation), translation)