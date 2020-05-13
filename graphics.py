import math
import random
from typing import Tuple, Union, Iterable, Sequence, Dict, List, Callable

import pygame.gfxdraw

import utils
from game_settings import GameSettings

DIRECTION_INDICATOR = "offscreen_direction_indicator"

VELOCITY_INDICATOR = "offscreen_velocity_indicator"

INDICATOR = "offscreen_indicator"

SHIP_RCS_TURN_RIGHT = "ship_rcs_turn_right"

SHIP_RCS_TURN_LEFT = "ship_rcs_turn_left"

SHIP_RCS_MOVE_RIGHT = "ship_rcs_move_right"
SHIP_RCS_MOVE_LEFT = "ship_rcs_move_left"
SHIP_RCS_MOVE_FORWARD = "ship_rcs_move_forward"
SHIP_RCS_MOVE_BACK = "ship_rcs_move_back"

SHIP_FLAME = "ship_flame"

SHIP = "ship"

SHIP_HULL_COLOR = (192, 192, 255)

ASTEROID_INDICATOR = "asteroid_indicator"

BULLET = "bullet"


class Shape:
	
	def __init__(self, offset: Tuple[float, float] = (0.0, 0.0)):
		self.offset = offset
	
	def apply_offset(self):
		pass
	
	def draw(
			self,
			surface: pygame.SurfaceType,
			scale: Union[Tuple[float, float], float] = 1,
			rotation: Union[float, Tuple[float, float]] = 0,
			translation: Tuple[float, float] = (0, 0)):
		pass
	
	def copy(self, apply_offset: bool = False):
		return self


class AnimatedShape(Shape):
	
	animated_shapes = set()
	
	def __init__(self, offset: Tuple[float, float] = (0, 0)):
		super().__init__(offset)
		AnimatedShape.animated_shapes.add(self)
	
	def tick(self, dt=GameSettings.tick):
		pass
	
	def remove(self):
		if self in AnimatedShape.animated_shapes:
			AnimatedShape.animated_shapes.remove(self)
	
	@staticmethod
	def tick_all(dt=GameSettings.tick):
		for shape in AnimatedShape.animated_shapes:
			shape.tick(dt)
	

class SimpleAnimatedShape(AnimatedShape):
	"""
	An animated shape having simple time-dependent rotation, scale, and offset animations
	"""
	
	def __init__(
			self,
			shape: Shape,
			offset: Tuple[float, float] = (0, 0),
			cycle: int = -1,
			scale_animation: Callable[[int], Union[float, Tuple[float, float]]] = None,
			rotation_animation: Callable[[int], Union[float, Tuple[float, float]]] = None,
			offset_animation: Callable[[int], Tuple[float, float]] = None):
		"""
		Initializes a simple animated shape with a static shape and some animations
		:param shape: The static shape
		:param offset: A static constant offset
		:param cycle: The length of the cycle in milliseconds. If -1 is given, the timer will be allowed to overflow
		:param scale_animation: The scale animation function. Input: elapsed time since the object started animating. Output: scale (single float or x-y tuple)
		:param rotation_animation: The rotation animation function. Input: elapsed time. Output: rotation, radians or tuple(x, y) vector
		:param offset_animation: The offset animation function. Input: elapsed time. Output: offset (x, y) relative to the shape's actual center
		"""
		super().__init__(offset)
		self.shape = shape
		self.scale_animation = scale_animation
		self.animated_scale = 1 if scale_animation is None else scale_animation(0)
		self.rotation_animation = rotation_animation
		self.animated_rotation = 0 if rotation_animation is None else rotation_animation(0)
		self.offset_animation = offset_animation
		self.animated_offset = (0, 0) if offset_animation is None else offset_animation(0)
		self.elapsed_time = 0
		self.cycle = cycle
		
	def tick(self, dt=GameSettings.tick):
		self.elapsed_time += dt
		if self.cycle > 0:
			self.elapsed_time = self.elapsed_time % self.cycle
		self.do_animations()
		
	def do_animations(self):
		self.animated_scale = 1 if self.scale_animation is None else self.scale_animation(self.elapsed_time)
		self.animated_rotation = 0 if self.rotation_animation is None else self.rotation_animation(self.elapsed_time)
		self.animated_offset = (0, 0) if self.offset_animation is None else self.offset_animation(self.elapsed_time)
		
	def draw(
			self,
			surface: pygame.SurfaceType,
			scale: Union[Tuple[float, float], float] = 1,
			rotation: Union[float, Tuple[float, float]] = 0,
			translation: Tuple[float, float] = (0, 0)):
		
		if isinstance(scale, tuple):
			if isinstance(self.animated_scale, tuple):
				combined_scale = scale[0]*self.animated_scale[0], scale[1]*self.animated_scale[1]
			else:
				combined_scale = scale[0]*self.animated_scale, scale[1]*self.animated_scale
		else:
			if isinstance(self.animated_scale, tuple):
				combined_scale = scale * self.animated_scale[0], scale * self.animated_scale[1]
			else:
				combined_scale = scale * self.animated_scale
		
		rot_angle = utils.vec_angle(rotation) if isinstance(rotation, tuple) else rotation
		rot_angle += utils.vec_angle(self.animated_rotation) if isinstance(self.animated_rotation, tuple) else self.animated_rotation
		
		combined_translation = utils.vec_plus(utils.vec_plus(translation, self.offset), self.animated_offset)
		
		self.shape.draw(surface, combined_scale, rot_angle, combined_translation)
		

class ComplexShape(Shape):
	def __init__(self, shapes: Iterable[Shape], offset: Tuple[float, float] = (0.0, 0.0)):
		super().__init__(offset)
		self.shapes = list(shapes)
	
	def apply_offset(self):
		for shape in self.shapes:
			shape.offset = utils.vec_plus(shape.offset, self.offset)
			shape.apply_offset()
		self.offset = 0.0, 0.0
	
	def copy(self, apply_offset: bool = False):
		copied_shapes = list([shape.copy() for shape in self.shapes])
		other_shape = ComplexShape(copied_shapes, self.offset)
		if apply_offset:
			other_shape.apply_offset()
		return other_shape
	
	def draw(
			self,
			surface: pygame.SurfaceType,
			scale: Union[Tuple[float, float], float] = 1,
			rotation: Union[float, Tuple[float, float]] = 0,
			translation: Tuple[float, float] = (0, 0)):
		for shape in self.shapes:
			off = utils.rotate_point(self.offset, rotation)
			shape.draw(surface, scale, rotation, utils.vec_plus(translation, off))


class RasterShape(Shape):
	def __init__(self, raster: pygame.SurfaceType, offset: Tuple[float, float] = (0.0, 0.0), rotate_enabled: bool = False, scale_enabled: bool = False):
		super().__init__(offset)
		self.raster = raster
		self.rotate_enabled = rotate_enabled
		self.scale_enabled = scale_enabled
	
	def draw(
			self,
			surface: pygame.SurfaceType,
			scale: Union[Tuple[float, float], float] = 1,
			rotation: Union[float, Tuple[float, float]] = 0,
			translation: Tuple[float, float] = (0, 0)):
		off = utils.rotate_point(self.offset, rotation)
		if self.rotate_enabled or self.scale_enabled:
			raster = pygame.transform.rotozoom(
				self.raster,
				-math.degrees(utils.vec_angle(rotation) if isinstance(rotation, tuple) else rotation),
				scale if not isinstance(scale, tuple) else utils.vec_len(scale)
			)
			additional_offset = (self.raster.get_width() - raster.get_width())/2, (self.raster.get_height() - raster.get_height())/2
			off = utils.vec_plus(off, additional_offset)
		else:
			raster = self.raster
		surface.blit(raster, utils.vec_int(utils.vec_plus(utils.vec_plus(translation, off), utils.vec_mul(self.raster.get_size(), -0.5))))
	
	def apply_offset(self):
		new_raster_size = self.raster.get_width() + abs(self.offset[0]), self.raster.get_height() + abs(self.offset[1])
		new_surface = pygame.Surface(new_raster_size, pygame.SRCALPHA | pygame.HWSURFACE)
		self.draw(new_surface)
		self.raster = new_surface
		self.offset = 0.0, 0.0
	
	def copy(self, apply_offset: bool = False):
		new_surface = pygame.Surface(self.raster.get_size(), pygame.SRCALPHA | pygame.HWSURFACE)
		new_surface.blit(self.raster, (0, 0))
		copied_shape = RasterShape(new_surface, self.offset)
		if apply_offset:
			copied_shape.apply_offset()
		return copied_shape


class Polygon(Shape):
	
	def __init__(
			self,
			points: Sequence[Tuple[float, float]],
			filled: bool = True,
			fill_color: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (0, 0, 0),
			contour: bool = True,
			contour_color: Union[None, Tuple[int, int, int], Tuple[int, int, int, int]] = None,
			offset: Tuple[float, float] = (0.0, 0.0)
	):
		super().__init__(offset)
		self.points = tuple(points)
		self.filled = filled
		self.fill_color = fill_color
		self.contour = contour
		self.contour_color = contour_color if contour and contour_color is not None else fill_color if contour else None
	
	def draw(
			self,
			surface: pygame.SurfaceType,
			scale: Union[Tuple[float, float], float] = 1,
			rotation: Union[float, Tuple[float, float]] = 0,
			translation: Tuple[float, float] = (0, 0)):
		
		off = utils.rotate_point(self.offset, rotation)
		polygon = utils.transform_polygon(self.points, scale, rotation, utils.vec_plus(translation, off))
		if self.filled:
			pygame.gfxdraw.filled_polygon(surface, polygon, self.fill_color)
		if self.contour:
			pygame.gfxdraw.aapolygon(surface, polygon, self.contour_color)
	
	def apply_offset(self):
		new_points = tuple(utils.vec_plus(p, self.offset) for p in self.points)
		self.points = new_points
		self.offset = 0.0, 0.0
	
	def copy(self, apply_offset: bool = False):
		copy_points = tuple(self.points)  # Tuples are immutable so any alterations to individual points will be ignored.
		copy = Polygon(copy_points, self.filled, self.fill_color, self.contour, self.contour_color, self.offset)
		if apply_offset:
			copy.apply_offset()
		return copy


class Circle(Shape):
	def __init__(
			self,
			radius: float,
			filled: bool = True,
			fill_color: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (0, 0, 0),
			contour: bool = True,
			contour_color: Union[None, Tuple[int, int, int], Tuple[int, int, int, int]] = None,
			offset: Tuple[float, float] = (0.0, 0.0)
	):
		super().__init__(offset)
		self.radius = radius
		self.filled = filled
		self.fill_color = fill_color
		self.contour = contour
		self.contour_color = contour_color if contour and contour_color is not None else fill_color if contour else None
	
	def draw(
			self,
			surface: pygame.SurfaceType,
			scale: Union[Tuple[float, float], float] = 1,
			rotation: Union[float, Tuple[float, float]] = 0,
			translation: Tuple[float, float] = (0, 0)):
		off = utils.rotate_point(self.offset, rotation)
		center = int(round(translation[0] + off[0])), int(round(translation[1] + off[1]))
		if not isinstance(scale, tuple):
			radius = int(round(self.radius * scale))
			if self.filled:
				pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, self.fill_color)
			if self.contour:
				pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, self.contour_color)
		else:
			if not rotation == 0:
				print("Rotated ellipses are not supported.")
			rx = int(round(scale[0] * self.radius))
			ry = int(round(scale[1] * self.radius))
			if self.filled:
				pygame.gfxdraw.filled_ellipse(surface, center[0], center[1], rx, ry, self.fill_color)
			if self.contour:
				pygame.gfxdraw.aaellipse(surface, center[0], center[1], rx, ry, self.contour_color)
	
	def apply_offset(self):
		print("Not applicable to a circle.")
	
	def copy(self, apply_offset: bool = False):
		copied = Circle(self.radius, self.filled, self.fill_color, self.contour, self.contour_color, self.offset)
		#  Offset application not applicable to circles.
		return copied
	
	def convert_to_polygon(self, resolution: Union[int, float, None] = None) -> Polygon:
		if resolution is None or isinstance(resolution, float):
			perimeter = 2 * math.pi * self.radius
			res_mul = 1.0 if resolution is None else resolution
			res = int(math.floor(perimeter * res_mul))
		else:
			res = resolution
		angular_diff = 2 * math.pi / res
		polar_vertices = [(self.radius, x * angular_diff) for x in range(res)]
		poly_points = [utils.to_cartesian(pv) for pv in polar_vertices]
		return Polygon(poly_points, self.filled, self.fill_color, self.contour, self.contour_color, self.offset)


rcs_shapes = {
	"FL": Polygon(
		(
			(0, -20),
			(20, -25),
			(20, -15)
		),
		fill_color=(255, 255, 255, 96)
	),
	"FR": Polygon(
		(
			(0, 20),
			(20, 25),
			(20, 15)
		),
		fill_color=(255, 255, 255, 96)
	),
	"RL": Polygon(
		(
			(-20, -20),
			(-40, -15),
			(-40, -25)
		),
		fill_color=(255, 255, 255, 96)
	),
	"RR": Polygon(
		(
			(-20, 20),
			(-40, 15),
			(-40, 25)
		),
		fill_color=(255, 255, 255, 96)
	),
	"LF": Polygon(
		(
			(0, -20),
			(-5, -40),
			(5, -40)
		),
		fill_color=(255, 255, 255, 96)
	),
	"LR": Polygon(
		(
			(-20, -20),
			(-25, -40),
			(-15, -40)
		),
		fill_color=(255, 255, 255, 96)
	),
	"RF": Polygon(
		(
			(0, 20),
			(-5, 40),
			(5, 40)
		),
		fill_color=(255, 255, 255, 96)
	),
	"RR2": Polygon(
		(
			(-20, 20),
			(-25, 40),
			(-15, 40)
		),
		fill_color=(255, 255, 255, 96)
	)
}


class GameShapes:
	shapes: Dict[str, Shape] = {
		SHIP: ComplexShape(
			[
				Polygon(
					(
						(20, 0),
						(-10, -10),
						(0, -20),
						(-20, -20),
						(-15, -10),
						(-15, 10),
						(-20, 20),
						(0, 20),
						(-10, 10)
					),
					fill_color=SHIP_HULL_COLOR,
					contour_color=(255, 255, 255)
				),
				Polygon(
					(
						(7, -2),
						(0, -4),
						(-5, -3),
						(-5, 3),
						(0, 4),
						(7, 2)
					),
					fill_color=(0, 64, 128),
					contour_color=(0, 0, 128)
				)
			]
		),
		SHIP_FLAME: ComplexShape(
			[
				Polygon(
					(
						(-15, -10),
						(-30, -15),
						(-45, 0),
						(-30, 15),
						(-15, 10)
					),
					fill_color=(255, 192, 0, 128),
					contour_color=(255, 192, 0, 192)
				),
				Polygon(
					(
						(-15, -6),
						(-20, -8),
						(-30, 0),
						(-20, 8),
						(-15, 6)
					),
					fill_color=(255, 64, 0, 128),
					contour_color=(255, 0, 0, 64)
				)
			]
		),
		SHIP_RCS_TURN_LEFT: ComplexShape(
			[
				rcs_shapes["FL"],
				rcs_shapes["RR"],
				rcs_shapes["RF"],
				rcs_shapes["LR"]
			]
		),
		SHIP_RCS_TURN_RIGHT: ComplexShape(
			[
				rcs_shapes["FR"],
				rcs_shapes["RL"],
				rcs_shapes["LF"],
				rcs_shapes["RR2"]
			]
		),
		SHIP_RCS_MOVE_LEFT: ComplexShape(
			[
				rcs_shapes["RF"],
				rcs_shapes["RR2"]
			]
		),
		SHIP_RCS_MOVE_RIGHT: ComplexShape(
			[
				rcs_shapes["LF"],
				rcs_shapes["LR"]
			]
		),
		SHIP_RCS_MOVE_FORWARD: ComplexShape(
			[
				rcs_shapes["RL"],
				rcs_shapes["RR"]
			]
		),
		SHIP_RCS_MOVE_BACK: ComplexShape(
			[
				rcs_shapes["FL"],
				rcs_shapes["FR"]
			]
		),
		INDICATOR: Polygon(
			(
				(10, 0),
				(-10, -10),
				(-10, 10)
			),
			fill_color=(192, 192, 255),
			contour_color=(255, 255, 255)
		),
		ASTEROID_INDICATOR: Polygon(
			(
				(6, 0),
				(-6, -6),
				(-6, 6)
			),
			fill_color=GameSettings.asteroid_color
		),
		VELOCITY_INDICATOR: Polygon(
			(
				(10, 0),
				(-10, -5),
				(-10, 5)
			),
			fill_color=(0, 0, 255)
		),
		DIRECTION_INDICATOR: Polygon(
			(
				(8, 0),
				(-8, 4),
				(-10, 0),
				(-8, -4)
			),
			fill_color=(255, 0, 0)
		),
		BULLET: ComplexShape(
			(
				Circle(
					3 * GameSettings.bullet_radius,
					fill_color=(GameSettings.bullet_color[0], GameSettings.bullet_color[1], GameSettings.bullet_color[2], 64),
					contour=True
				),
				Polygon(
					(
						(0, GameSettings.bullet_radius),
						(-2 * GameSettings.bullet_radius, 0),
						(0, -GameSettings.bullet_radius),
						(2 * GameSettings.bullet_radius, 0),
					),
					fill_color=GameSettings.bullet_color,
					contour=True
				),
				Circle(
					0.5 * GameSettings.bullet_radius,
					fill_color=(255, 255, 255),
					contour=False
				)
			)
		)
	}
	
	@staticmethod
	def get_shape(key: str):
		if key in GameShapes.shapes:
			return GameShapes.shapes[key]
		return Circle(1)
	
	def __index__(self, key: str):
		return GameShapes.get_shape(key)


def gauss_blur(bitmap: pygame.SurfaceType, radius: int = 3) -> pygame.SurfaceType:
	
	def color_mul(
			color: Union[Tuple[int, int, int], Tuple[int, int, int, int], Tuple[float, float, float], Tuple[float, float, float, float]],
			multiplier: float) -> Union[Tuple[float, float, float], Tuple[float, float, float, float]]:
		r, g, b = color[0]*multiplier, color[1]*multiplier, color[2]*multiplier
		if len(color) == 3:
			return r, g, b
		a = color[3]*multiplier
		return r, g, b, a
	
	def fcolor_add(
			c1: Union[Tuple[float, float, float], Tuple[float, float, float, float]],
			c2: Union[Tuple[float, float, float], Tuple[float, float, float, float]]
	) -> Union[Tuple[float, float, float], Tuple[float, float, float, float]]:
		r, g, b = c1[0]+c2[0], c1[1]+c2[1], c1[2]+c2[2]
		if len(c1) == 3 and len(c2) == 3:
			return r, g, b
		
		if len(c1) > 3:
			a = c1[3]
		else:
			a = 0
		if len(c2) > 3:
			a += c2[3]
		return r, g, b, a
	
	def fcolor_to_int(c: Union[Tuple[float, float, float], Tuple[float, float, float, float]]) -> Union[Tuple[int, int, int], Tuple[int, int, int, int]]:
		return tuple((int(round(component)) for component in c))
	
	mask_size = 2 * radius + 1
	mask_midpoint = radius + 0.5
	sigma = radius/2.5
	mask = list()
	for u in range(mask_size):
		mask_col = list()
		for v in range(mask_size):
			exponent1 = ((u-mask_midpoint-0.5) / (2*sigma))**2 + ((v-mask_midpoint-0.5) / (2*sigma))**2
			exponent2 = ((u-mask_midpoint+0.5) / (2*sigma))**2 + ((v-mask_midpoint-0.5) / (2*sigma))**2
			exponent3 = ((u-mask_midpoint-0.5) / (2*sigma))**2 + ((v-mask_midpoint+0.5) / (2*sigma))**2
			exponent4 = ((u-mask_midpoint+0.5) / (2*sigma))**2 + ((v-mask_midpoint+0.5) / (2*sigma))**2
			mv = (math.exp(-exponent1) + math.exp(-exponent2) + math.exp(-exponent3) + math.exp(-exponent4))/4
			mask_col.append(mv)
		mask.append(mask_col)
	
	print("Mask is:")
	print(mask)
	
	bitmap_pixels = pygame.PixelArray(bitmap)
	blurred_surface = pygame.Surface(bitmap.get_size(), pygame.SRCALPHA)
	blurred_surface.fill((0, 0, 0, 0))
	blurred_pixels = pygame.PixelArray(blurred_surface)
	w, h = bitmap.get_size()
	for x in range(w):
		for y in range(h):
			u_start = abs(min((x - radius), 0))
			v_start = abs(min((y - radius), 0))
			u_end = radius + min(abs(x - w), radius + 1)
			v_end = radius + min(abs(y - h), radius + 1)
			factor = 0
			value = 0.0, 0.0, 0.0, 0.0
			for u in range(u_start, u_end):
				for v in range(v_start, v_end):
					mask_curr = mask[u][v]
					factor += mask_curr
					masked_coords = x + u - radius, y + v - radius
					try:
						pixel = bitmap.unmap_rgb(bitmap_pixels[masked_coords[0], masked_coords[1]])
						value = fcolor_add(value, color_mul(pixel, mask_curr))
					except IndexError:
						print("Bad index: xy={}, uv={}, masked={}".format((x, y), (u, v), masked_coords))
						breakpoint()
			
			norm_val = fcolor_to_int(color_mul(value, 1/factor))
			blurred_pixels[x, y] = norm_val
	
	blurred_pixels.close()
	
	return blurred_surface


def polygon_arc(radius: float, start_angle: float, end_angle: float, resolution: Union[int, None] = None, fuzzing: float = 0.0) -> Tuple[
	Tuple[float, float], ...]:
	res = resolution if resolution is not None else int(round((end_angle - start_angle) * radius * 0.5)) + 1
	angle_step = (end_angle - start_angle) / res
	points = []
	for x in range(0, res):
		r = radius if fuzzing == 0.0 else radius * (1 + random.random() * fuzzing)
		points.append(utils.to_cartesian((r, start_angle + x * angle_step)))
	points.append(utils.to_cartesian((radius, end_angle)))
	return tuple(points)


def generate_continents(
		planet_radius: float,
		color_ranges: Tuple[Tuple[float, float, Tuple[int, int, int], Tuple[int, int, int]], ...],
		continent_count_range: Tuple[int, int] = (2, 4),
		landmass_ratio_range: Tuple[float, float] = (0.5, 0.66)) -> ComplexShape:
	# If there is a polar color range, add an ice cap
	continent_polys: List[Polygon] = list()
	if len(color_ranges) > 0 and color_ranges[0][0] == 0:
		pole_points = []
		current_radius = planet_radius * color_ranges[0][1] - planet_radius * 0.05
		max_delta = planet_radius * 0.05
		for x in range(0, 360, random.randrange(12, 45)):
			current_radius += (2 * random.random() - 1) * max_delta
			if current_radius < max_delta:
				current_radius = max_delta
			angle = math.radians(x)
			pole_points.append(utils.to_cartesian((current_radius, angle)))
		pole_min_color = color_ranges[0][2]
		pole_max_color = color_ranges[0][3]
		pole_color = (
			random.randint(pole_min_color[0], pole_max_color[0]),
			random.randint(pole_min_color[1], pole_max_color[1]),
			random.randint(pole_min_color[2], pole_max_color[2])
		)
		pole = Polygon(pole_points, fill_color=pole_color)
		continent_polys.append(pole)
	
	# Make continent polygons
	landmass_ratio = landmass_ratio_range[0] + random.random() * (landmass_ratio_range[1] - landmass_ratio_range[0])
	continent_count = random.randint(continent_count_range[0], continent_count_range[1])
	continent_start_angles = []
	continent_end_angles = []
	continent_angle_portions = [math.fabs(random.gauss(3, 1)) for x in range(continent_count)]
	ocean_angle_portions = [cap / landmass_ratio for cap in continent_angle_portions]
	continent_angle_factor = 2 * math.pi / (sum(continent_angle_portions) + sum(ocean_angle_portions))
	print("Angle factor: {}".format(math.degrees(continent_angle_factor)))
	start_angle = 0
	end_angle = 0
	for x in range(len(continent_angle_portions)):
		start_angle = end_angle + ocean_angle_portions[x] * continent_angle_factor
		end_angle = start_angle + continent_angle_portions[x] * continent_angle_factor
		continent_start_angles.append(start_angle)
		continent_end_angles.append(end_angle)
	
	print("Landmass ratio: {}".format(landmass_ratio))
	print("Continent angle portions: {}".format(continent_angle_portions))
	print("Ocean angle portions: {}".format(ocean_angle_portions))
	print("Continent angles: {}".format(
		tuple((math.degrees(continent_start_angles[i]), math.degrees(continent_end_angles[i])) for i in range(0, len(continent_start_angles)))))
	
	for cont_idx in range(continent_count):
		min_angle = continent_start_angles[cont_idx]
		max_angle = continent_end_angles[cont_idx]
		print("Continent {}: {}".format(cont_idx, (math.degrees(min_angle), math.degrees(max_angle))))
		start_angle = min_angle + random.random() * 0.3 * (max_angle - min_angle)
		end_angle = max_angle - random.random() * 0.3 * (max_angle - min_angle)
		angle_step_max = (end_angle - start_angle) / random.randint(4, 8)
		for segment in color_ranges[1:]:
			min_radius = segment[0] * planet_radius
			max_radius = segment[1] * planet_radius
			
			polygon_points = list(polygon_arc(min_radius, start_angle, end_angle))
			
			min_color = segment[2]
			max_color = segment[3]
			min_points_polar = []
			max_points_polar = []
			segment_color = (
				random.randint(min_color[0], max_color[0]),
				random.randint(min_color[1], max_color[1]),
				random.randint(min_color[2], max_color[2])
			)
			segment_step = max((segment[1] - segment[0]) / 6, 0.025) * planet_radius
			print("Segment step: {}".format(segment_step))
			current_radius = min_radius
			while current_radius < max_radius:
				min_points_polar.append((current_radius, start_angle))
				max_points_polar.append((current_radius, end_angle))
				
				next_min_angle = start_angle + (2 * random.random() - 1) * angle_step_max
				next_max_angle = end_angle + (2 * random.random() - 1) * angle_step_max
				if next_min_angle > next_max_angle:
					next_min_angle, next_max_angle = next_max_angle, next_min_angle
				
				start_angle = next_min_angle
				end_angle = next_max_angle
				current_radius += segment_step
			
			polygon_points += list([utils.to_cartesian(p) for p in max_points_polar])
			if segment == color_ranges[-1]:
				end_arc = list(polygon_arc(max_radius, start_angle, end_angle))[::-1]
			else:
				end_arc = list(polygon_arc(max_radius, start_angle, end_angle, fuzzing=0.1))[::-1]
			polygon_points += end_arc
			polygon_points += list([utils.to_cartesian(p) for p in min_points_polar[::-1]])
			segment = Polygon(polygon_points, fill_color=segment_color)
			continent_polys.append(segment)
	
	return ComplexShape(continent_polys)


def generate_craters(
		planet_radius: float,
		depth: int = 3,
		cratered_surface: Tuple[float, float] = (0.3, 0.6),
		crater_size_range: Tuple[float, float] = (0.15, 0.35)) -> ComplexShape:
	def lat_distortion(lat: float) -> float:
		return math.pi * math.sin(lat)
	
	def distorted_lon(lat: float, lon0: float, lon: float) -> float:
		return angle_wrap(lon0 + lat_distortion(lat) * (lon - lon0))
	
	def haversine(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
		lat1, lon1 = c1
		lat2, lon2 = c2
		d_lat = lat2 - lat1
		d_lon = lon2 - lon1
		a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
		c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
		return c  # Radius is assumed to be 1 unit
	
	def angle_wrap(a: float) -> float:
		while a > 2 * math.pi:
			a -= 2 * math.pi
		while a < 0:
			a += 2 * math.pi
		return a
	
	def bearing(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
		lat1, lon1 = c1
		lat2, lon2 = c2
		d_lon = lon2 - lon1
		return angle_wrap(math.atan2(math.sin(d_lon) * math.cos(lat2), math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)))
	
	def angle_plus(a1: float, a2: float) -> float:
		a = a1 + a2
		return angle_wrap(a)
	
	def angle_minus(a1: float, a2: float) -> float:
		return angle_wrap(a1 - a2)
	
	class Crater:
		
		crater_vertex_count = 8
		
		def __init__(self, lat, lon, r, crater_depth):
			self.lat = lat
			self.lon = lon
			self.r = r
			self.depth = crater_depth
			
			vertices = []
			vertex_angle = (2 * math.pi) / Crater.crater_vertex_count
			for v in range(Crater.crater_vertex_count):
				angle = v * vertex_angle
				nat_lat = lat + r * math.sin(angle)
				nat_lon = lon + r * math.cos(angle)
				dist_lon = distorted_lon(nat_lat, lon, nat_lon)
				vertices.append((nat_lat, dist_lon))
			self.vertices = vertices
			self.area = 0
			self.update_area()
		
		def reorder(self, midpoint: Union[None, Tuple[float, float]] = None):
			if midpoint is None:
				avg_lat = sum([x[0] for x in self.vertices]) / len(self.vertices)
				avg_lon = sum([x[1] for x in self.vertices]) / len(self.vertices)
				new_midpoint = (avg_lat, avg_lon)
			else:
				new_midpoint = midpoint
			vertices = list(enumerate(self.vertices))
			angles = [bearing(new_midpoint, v[1]) for v in vertices]
			vertices.sort(key=lambda v: angles[v[0]])
			self.vertices = list([v[1] for v in vertices])
			self.lat, self.lon = new_midpoint
			self.update_area()
		
		def distance_check(self, other) -> Tuple[bool, float]:
			way_too_far = math.fabs(self.lat - other.lat) > (self.r + other.r) or math.fabs(self.lon - other.lon) > (self.r + other.r)
			if way_too_far:
				return False, 2 * math.pi
			midpoint_distance = haversine((self.lat, self.lon), (other.lat, other.lon))
			close_enough = midpoint_distance < (self.r + other.r)
			return close_enough, midpoint_distance
		
		def cutout(self, other) -> bool:
			# breakpoint()
			should_cutout, midpoint_distance = self.distance_check(other)
			if not should_cutout:
				return False  # The other crater is too far away
			vertices_to_del = []
			for vertex in self.vertices:
				if haversine(vertex, (other.lat, other.lon)) < other.r:
					vertices_to_del.append(vertex)
			if len(vertices_to_del) == 0:
				return False  # If the craters were circular they would overlap, but they are polygons...
			vertices_to_add = []
			for vertex in other.vertices:
				if haversine(vertex, (self.lat, self.lon)) < self.r:
					vertices_to_add.append(vertex)
			if len(vertices_to_add) == 0:
				return False  # Again, the world is imperfect.
			
			all_verts = set(self.vertices)
			all_verts.difference_update(vertices_to_del)
			all_verts.update(vertices_to_add)
			self.vertices = list(all_verts)
			
			new_midpoint = None
			if midpoint_distance < other.r:
				# A calculated midpoint is most likely going to be outside the new crater and reordering will be idiotic
				closest_vertex = None
				closest_vertex_distance = math.inf
				furthest_vertex = None
				furthest_vertex_distance = 0
				for vertex in self.vertices:
					v_dist = haversine(vertex, (self.lat, self.lon))
					if v_dist < closest_vertex_distance:
						closest_vertex_distance = v_dist
						closest_vertex = vertex
					if v_dist > furthest_vertex_distance:
						furthest_vertex_distance = v_dist
						furthest_vertex = vertex
				new_midpoint = (closest_vertex[0] + furthest_vertex[0]) / 2, (closest_vertex[1] + furthest_vertex[1]) / 2
			
			self.reorder(new_midpoint)
			return True
		
		def merge(self, other) -> bool:
			# breakpoint()
			should_merge, midpoint_distance = self.distance_check(other)
			if not should_merge:
				return False  # The other crater is too far away
			all_vertices = set(self.vertices).union(other.vertices)
			overlap = set()
			for v in self.vertices:
				if haversine(v, (other.lat, other.lon)) < other.r:
					overlap.add(v)
			if len(overlap) == 0:
				return False  # Imperfect craters
			tmp_len = len(overlap)
			for v in other.vertices:
				if haversine(v, (self.lat, self.lon)) < self.r:
					overlap.add(v)
			if len(overlap) == tmp_len:
				return False  # Again, imperfect
			all_vertices.difference_update(overlap)
			self.vertices = list(all_vertices)
			self.reorder()
		
		def update_area(self):
			mid = self.lat, self.lon
			self.area = 0
			for x in range(len(self.vertices)):
				v1 = self.vertices[x]
				v2 = self.vertices[x - 1]
				theta1 = bearing(mid, v1)
				theta2 = bearing(mid, v2)
				edge1 = haversine(mid, v1)
				edge2 = haversine(mid, v2)
				self.area += math.fabs(math.sin(angle_minus(theta1, theta2))) * edge1 * edge2 * 0.5
		
		def polygonize(self, r: float, alpha_step: int) -> Union[Polygon, None]:
			# breakpoint()
			if len(self.vertices) < 3:
				return None
			polygon_points_polar = []
			for vertex in self.vertices:
				polar_radius = min(1.0, 1 - math.sin(vertex[0])) * r
				polygon_points_polar.append((polar_radius, vertex[1]))
			polygon_points = [utils.to_cartesian(v) for v in polygon_points_polar]
			'''print("Vertices: {}".format(self.vertices))
			print("Polar coords: {}".format(polygon_points_polar))
			print("Points: {}".format(polygon_points))'''
			alpha = self.depth * alpha_step
			return Polygon(polygon_points, fill_color=(0, 0, 0, alpha))
	
	target_cratering = math.pi * (cratered_surface[0] + (cratered_surface[1] - cratered_surface[0]) * random.random())
	current_cratering = 0
	craters = []
	while current_cratering < target_cratering * 0.75:
		while current_cratering < target_cratering:
			crater_depth = random.randrange(depth) + 1
			crater_size = crater_size_range[0] + random.random() * (crater_size_range[1] - crater_size_range[0])
			crater_target_deg = random.gauss(45, 25), random.randrange(360)
			crater = Crater(math.radians(crater_target_deg[0]), math.radians(crater_target_deg[1]), crater_size, crater_depth)
			craters.append(crater)
			current_cratering += crater.area
		idx_1 = 0
		while idx_1 < len(craters):
			crater1 = craters[idx_1]
			idx_2 = idx_1 + 1
			while idx_2 < len(craters):
				crater2 = craters[idx_2]
				should_alter = crater1.distance_check(crater2)[0]
				if should_alter:
					if crater1.depth == crater2.depth:
						if crater1.merge(crater2):
							del craters[idx_2]
						else:
							idx_2 += 1
					elif crater1.depth < crater2.depth:
						crater1.cutout(crater2)
						idx_2 += 1
					elif crater2.depth < crater1.depth:
						crater2.cutout(crater1)
						idx_2 += 1
				else:
					idx_2 += 1
			idx_1 += 1
		current_cratering = sum(crater.area for crater in craters)
	
	alpha_step = 224 // depth
	polygons = filter(lambda x: x is not None, [crater.polygonize(planet_radius, alpha_step) for crater in craters])
	return ComplexShape(polygons)


class NightLights(ComplexShape):
	def __init__(
			self,
			lights: Iterable[Tuple[float, float, int]],
			planet_radius: float, offset: Tuple[float, float] = (0.0, 0.0),
			sun_direction: Tuple[float, float] = (1, 1),
			max_light_size: Union[int, None] = None
	):
		mls = max([light[2] for light in lights]) if max_light_size is None else max_light_size
		alpha_step = 255 // mls
		light_points = [Circle(math.sqrt(p[2]), fill_color=(255, 240, 200, min(255, 2*p[2] * alpha_step)), contour=False, offset=(p[0], p[1])) for p in lights]
		super().__init__(light_points, offset)
		
		terminator_vector = utils.rotate_point(sun_direction, 0.5 * math.pi)
		
		def occluded_equation() -> Callable[[Tuple[float, float]], bool]:
			if terminator_vector[0] == 0:
				if sun_direction[0] > 0:
					return lambda c: c[0] ** 2 + c[1] ** 2 < (planet_radius+5)**2 and c[0] < 0
				else:
					return lambda c: c[0] ** 2 + c[1] ** 2 < (planet_radius+5)**2 and c[0] > 0
			m = terminator_vector[1] / terminator_vector[0]
			if sun_direction[1] > 0:
				return lambda c: c[0] ** 2 + c[1] ** 2 < (planet_radius+5)**2 and c[1] < m * c[0]
			else:
				return lambda c: c[0] ** 2 + c[1] ** 2 < (planet_radius+5)**2 and c[1] > m * c[0]
		
		self.should_draw_point = occluded_equation()
	
	def draw(
			self,
			surface: pygame.SurfaceType,
			scale: Union[Tuple[float, float], float] = 1,
			rotation: Union[float, Tuple[float, float]] = 0,
			translation: Tuple[float, float] = (0, 0)):
		for shape in self.shapes:
			shape_center = utils.rotate_point(shape.offset, rotation)
			if self.should_draw_point(shape_center):
				off = utils.rotate_point(self.offset, rotation)
				shape.draw(surface, scale, rotation, utils.vec_plus(translation, off))


def generate_night_lights(
		planet_radius: float,
		continent_shapes: ComplexShape = None,
		inhabited_area: float = 0.35,
		urbanization_factor: float = 0.4,
		sun_direction: Tuple[float, float] = (1, 1)):
	tmp_surface = pygame.Surface((int(round(2 * planet_radius)), int(round(2 * planet_radius))), pygame.HWSURFACE | pygame.SRCALPHA)
	offset = utils.vec_int(utils.vec_mul(tmp_surface.get_size(), 0.5))
	cs = continent_shapes if continent_shapes is not None else Circle(planet_radius, fill_color=(255, 255, 255, 255))
	cs.draw(tmp_surface, translation=offset)
	land_mask = pygame.mask.from_surface(tmp_surface)
	
	def on_continent(polar_coords: Tuple[float, float], cartesian_offset: Tuple[float, float] = (0.0, 0.0)) -> bool:
		uc_cart = utils.vec_int(utils.vec_plus(utils.vec_plus(utils.to_cartesian(polar_coords), cartesian_offset), offset))
		return utils.range_check_coords(uc_cart, (0, 0), land_mask.get_size()) and land_mask.get_at((uc_cart[0], uc_cart[1]))
	
	light_point_count = int(round((land_mask.count() * inhabited_area) / 5))
	urban_center_count = light_point_count // 15
	urban_centers = []
	
	while len(urban_centers) < urban_center_count:
		for x in range(urban_center_count):
			uc_polar = planet_radius * min(1.0, math.fabs(random.gauss(0.75, 0.25))), math.radians(random.randint(0, 360))
			if on_continent(uc_polar):
				urban_centers.append(uc_polar)
	
	all_lights: List[Union[Tuple[float, float], Tuple[float, float, int]]] = [] + [utils.to_cartesian(uc) for uc in urban_centers]
	urbanization_sigma = planet_radius ** ((1 - urbanization_factor) / 2)
	while len(all_lights) < light_point_count:
		urban_center = random.choice(urban_centers)
		lp_offset = random.gauss(0, urbanization_sigma), random.gauss(0, urbanization_sigma)
		if on_continent(urban_center, lp_offset):
			all_lights.append(utils.vec_plus(utils.to_cartesian(urban_center), lp_offset))
	
	# Cull duplicates
	idx = 0
	removed = 0
	while idx < len(all_lights):
		idx2 = idx + 1
		light_round_coords = utils.vec_int(all_lights[idx])
		while idx2 < len(all_lights):
			other_round_coords = utils.vec_int(all_lights[idx2])
			if light_round_coords == other_round_coords:
				removed += 1
				del all_lights[idx2]
				all_lights[idx] = (all_lights[idx][0], all_lights[idx][1], 2 if len(all_lights[idx]) < 3 else all_lights[idx][2] + 1)
			else:
				idx2 += 1
		# Ensure that every light has a brightness value
		if len(all_lights[idx]) < 3:
			all_lights[idx] = (all_lights[idx][0], all_lights[idx][1], 1)
		idx += 1
	
	print("Removed {} duplicate lights".format(removed))
	
	return NightLights(all_lights, planet_radius, sun_direction=sun_direction)


def generate_atmosphere(planet_radius: float, atmosphere_thickness: float, atmosphere_color: Tuple[int, int, int, int] = (200, 200, 200, 8)) -> RasterShape:
	atmo_radius = planet_radius + atmosphere_thickness
	tmp_surface = pygame.Surface((int(2 * math.ceil(atmo_radius)), int(2 * math.ceil(atmo_radius))), pygame.SRCALPHA)
	
	def local_atmo_thickness(r: float) -> float:
		if r > atmo_radius:
			return 0
		atmo_z = atmo_radius * math.sin(math.acos(r / atmo_radius))
		if r > planet_radius:
			return 2 * atmo_z
		else:
			planet_z = planet_radius * math.sin(math.acos(r / planet_radius))
			return atmo_z - planet_z
	
	def local_atmo_color(r: float) -> Tuple[int, int, int, int]:
		thickness = local_atmo_thickness(r)
		return atmosphere_color[0], atmosphere_color[1], atmosphere_color[2], min(255, int(round(atmosphere_color[3] * thickness / atmosphere_thickness)))
	
	midpoint = int(math.ceil(atmo_radius))
	pixels = pygame.PixelArray(tmp_surface)
	for x in range(tmp_surface.get_width()):
		for y in range(tmp_surface.get_height()):
			r = math.sqrt((x - atmo_radius) ** 2 + (y - atmo_radius) ** 2)
			if r < atmo_radius:
				color = local_atmo_color(r)
				pixels[x, y] = color
	
	return RasterShape(tmp_surface)


def generate_clouds(
		troposphere_radius: float,
		planet_radius: Union[float, None] = None,
		cloud_cover: float = 0.6,
		cyclone_count_range: Tuple[int, int] = (6, 10),
		cloud_color: Tuple[int, int, int, int] = (255, 255, 255, 128),
		independent_rotation: Union[float, None] = None,
		gauss_blurring: Union[int, None] = None) -> Union[ComplexShape, RasterShape, AnimatedShape]:
	texture_size = int(round((troposphere_radius + ((troposphere_radius - planet_radius) if planet_radius is not None else 0)) * math.pi)) \
	               + (0 if gauss_blurring is None else 2*gauss_blurring)
	print("Troposphere radius: {}, planet radius: {}, texture size: {}".format(troposphere_radius, planet_radius, texture_size))
	texture_midpoint = texture_size // 2, texture_size // 2
	big_spiral_radius = int(round((troposphere_radius + ((troposphere_radius - planet_radius) if planet_radius is not None else 0)) * math.pi))/2
	print("Texture midpoint: {}".format(texture_midpoint))
	
	def cyclone(radius: float, cyclone_midpoint: Tuple[float, float], branches: int = 4, turns: float = 1) -> Tuple[Tuple[Tuple[float, float], ...], ...]:
		used_cloud_cover = max(cloud_cover / 2, min(0.75, random.gauss(cloud_cover, cloud_cover / 3)))
		cyclone_turn_angle = turns * 2 * math.pi
		branch_angle = math.pi * 2 / branches
		branch_cloud_angle = branch_angle * used_cloud_cover / 2
		branch_polygons_polar: List[List[Tuple[float, float]]] = []
		for k in range(branches):
			branch_polygons_polar.append([(radius, branch_angle * k)])
		steps = max(int(radius // 3), 8)
		angular_fuzzing_sigma = branch_cloud_angle / 6
		step_turn = cyclone_turn_angle / steps
		for s in range(1, steps):
			step_radius = radius * (steps - s) / steps
			for k in range(branches):
				branch_midpoint = k * branch_angle + s * step_turn
				left = random.gauss(branch_midpoint - branch_cloud_angle - angular_fuzzing_sigma, angular_fuzzing_sigma)
				right = random.gauss(branch_midpoint + branch_cloud_angle + angular_fuzzing_sigma, angular_fuzzing_sigma)
				branch_polygons_polar[k].append((step_radius, left))
				branch_polygons_polar[k].insert(0, (step_radius, right))
		for k in range(branches):
			branch_polygons_polar[k].append((0, 0))
		
		branch_polygon_points = []
		for k in range(branches):
			branch_points = tuple([utils.vec_plus(cyclone_midpoint, utils.to_cartesian(p)) for p in branch_polygons_polar[k]])
			branch_polygon_points.append(branch_points)
		return tuple(branch_polygon_points)
	
	def restrict_to_circle(poly_points: Tuple[Tuple[float, float], ...]) -> Tuple[Tuple[float, float], ...]:
		restricted_points = []
		for p in poly_points:
			r_squared = big_spiral_radius ** 2
			distance_squared = (p[0] - texture_midpoint[0]) ** 2 + (p[1] - texture_midpoint[1]) ** 2
			if distance_squared <= r_squared:
				restricted_points.append(p)
			else:
				polar_point = utils.to_polar(p, texture_midpoint)
				restricted_points.append(utils.to_cartesian((big_spiral_radius, polar_point[1]), texture_midpoint))
		return tuple(restricted_points)
	
	big_spiral = cyclone(big_spiral_radius, texture_midpoint, 8, 2)
	cyclone_midpoints = [
		utils.to_cartesian(
			(random.randrange(int(math.floor(big_spiral_radius))), math.radians(random.randrange(360))),
			texture_midpoint)
		for n
		in range(random.randint(cyclone_count_range[0], cyclone_count_range[1]))
	]
	cyclone_radii = [random.gauss(0.4, 0.15) * utils.vec_len(utils.vec_minus(mp, texture_midpoint)) for mp in cyclone_midpoints]
	
	# Prevent cyclone overlaps
	cyclone_overlaps: List[List[bool]] = [list() for x in range(len(cyclone_midpoints))]
	for idx in range(len(cyclone_midpoints)):
		cyclone1_midpoint = cyclone_midpoints[idx]
		cyclone1_radius = cyclone_radii[idx]
		for idx2 in range(len(cyclone_midpoints)):
			if idx2 == idx:
				cyclone_overlaps[idx].append(False)
			else:
				cyclone2_midpoint = cyclone_midpoints[idx2]
				cyclone2_radius = cyclone_radii[idx2]
				cyclone_overlaps[idx].append(utils.vec_len(utils.vec_minus(cyclone1_midpoint, cyclone2_midpoint)) < cyclone1_radius + cyclone2_radius)
	
	cyclone_overlap_count = [c_o.count(True) for c_o in cyclone_overlaps]
	failed_to_relocate = []
	while any(cyclone_overlap_count):
		most_overlapping_idx = max(range(len(cyclone_overlap_count)), key=lambda x: cyclone_overlap_count[x])
		most_overlapping_radius = cyclone_radii[most_overlapping_idx]
		overlapping = True
		tries = 0
		new_midpoint = cyclone_midpoints[most_overlapping_idx]
		while overlapping and tries < 5:
			overlapping = False
			new_midpoint = utils.to_cartesian((random.randrange(int(math.floor(big_spiral_radius))), math.radians(random.randrange(360))), texture_midpoint)
			for other_idx in range(len(cyclone_midpoints)):
				if other_idx == most_overlapping_idx:
					continue
				else:
					other_midpoint = cyclone_midpoints[other_idx]
					other_radius = cyclone_radii[other_idx]
					overlapping = utils.vec_len(utils.vec_minus(new_midpoint, other_midpoint)) < other_radius + most_overlapping_radius
					if overlapping:
						tries = tries + 1
						break
		if overlapping:
			failed_to_relocate.append(most_overlapping_idx)
			cyclone_midpoints[most_overlapping_idx] = (-5 * texture_size, -5 * texture_size)
			cyclone_radii[most_overlapping_idx] = 0
		else:
			cyclone_midpoints[most_overlapping_idx] = new_midpoint
		cyclone_overlap_count[most_overlapping_idx] = 0
	
	print("Failed to relocate {} cyclones".format(len(failed_to_relocate)))
	failed_to_relocate.sort(reverse=True)
	for failed_to_relocate_idx in failed_to_relocate:
		del cyclone_midpoints[failed_to_relocate_idx]
		del cyclone_radii[failed_to_relocate_idx]
	
	cyclones = [cyclone(cyclone_radii[i], cyclone_midpoints[i]) for i in range(len(cyclone_midpoints))]
	
	def spherify_point(point: Tuple[float, float]) -> Tuple[float, float]:
		polar_point = utils.to_polar(point, texture_midpoint)
		latitude = polar_point[0] / troposphere_radius
		new_radius = troposphere_radius * math.sin(latitude)
		if latitude > (0.5 * math.pi):
			print("Point is on the opposite hemisphere. Cartesian: {}, polar: {}, latitude: {}, new radius: {}".format(point, polar_point, latitude, new_radius))
		return utils.to_cartesian((new_radius, polar_point[1]), texture_midpoint)
	
	def spherify_polygon(polygon: Tuple[Tuple[float, float]]) -> Tuple[Tuple[float, float]]:
		return tuple((spherify_point(p) for p in polygon))
	
	if gauss_blurring is None or gauss_blurring == 0:
		base_layer_color = cloud_color[0], cloud_color[1], cloud_color[2], cloud_color[3] // 2
	else:
		base_layer_color = cloud_color[0], cloud_color[1], cloud_color[2], min(200, int(round(cloud_cover*(cloud_color[3])*gauss_blurring)))
	
	cyclone_polygons = [
		Polygon(spherify_polygon(restrict_to_circle(spiral_arm)), filled=True, fill_color=base_layer_color, contour=False)
		for spiral_arm
		in big_spiral]
	for cy in cyclones:
		cyclone_alpha = random.randrange(int(round(cloud_color[3] * 0.75)), min(200, int(round(cloud_color[3] * 1.5))))
		if gauss_blurring is not None and gauss_blurring > 0:
			cyclone_alpha = min(200, int(round(2 * cloud_cover * cyclone_alpha * gauss_blurring)))
		cyclone_color = cloud_color[0], cloud_color[1], cloud_color[2], cyclone_alpha
		for spiral_arm in cy:
			if len(spiral_arm) >= 3:
				cyclone_polygons.append(Polygon(spherify_polygon(restrict_to_circle(spiral_arm)), filled=True, fill_color=cyclone_color, contour=False))
	
	cyclones_shape = ComplexShape(cyclone_polygons)
	cyclones_shape.offset = utils.vec_mul(texture_midpoint, -1)
	cyclones_shape.apply_offset()
	if gauss_blurring is not None and gauss_blurring > 0:
		blurred_size = (int(math.ceil(troposphere_radius)) + gauss_blurring)*2
		tmp_surface = pygame.Surface((blurred_size, blurred_size), pygame.SRCALPHA)
		tmp_surface.fill((0, 0, 0, 0))
		cyclones_shape.draw(tmp_surface, translation=(blurred_size/2, blurred_size/2))
		blurred = gauss_blur(tmp_surface, gauss_blurring)
		inner_shape = RasterShape(blurred, rotate_enabled=True)
	else:
		inner_shape = cyclones_shape
	if independent_rotation is None or independent_rotation == 0:
		return inner_shape
	else:
		cycle = 1000 * int(round(math.pi * 2 / math.fabs(independent_rotation)))
		return SimpleAnimatedShape(inner_shape, cycle=cycle, rotation_animation=lambda time: (time/1000)*independent_rotation)


def generate_shield(
		shield_radius: float,
		shield_charge: float,
		shield_color: Tuple[int, int, int, int] = (0, 180, 220, 32),
		original_shape: Union[Circle, Polygon] = None) -> Shape:
	fill_color = shield_color[0], shield_color[1], shield_color[2], min(255, int(round(shield_charge * shield_color[3])))
	perimeter_color = shield_color[0], shield_color[1], shield_color[2], min(255, int(round(2 * shield_charge * shield_color[3])))
	if original_shape is None:
		return Circle(shield_radius, filled=True, fill_color=fill_color, contour_color=perimeter_color)
	else:
		original_shape.fill_color = fill_color
		original_shape.contour_color = perimeter_color
		return original_shape


def generate_terminator(planet_radius: float, atmosphere_thickness: float, ambient_light: int = 55, sun_direction: Tuple[float, float] = (1, 1)) -> RasterShape:
	atmo_radius = planet_radius + atmosphere_thickness
	tmp_surface = pygame.Surface((int(2 * math.ceil(atmo_radius)), int(2 * math.ceil(atmo_radius))), pygame.SRCALPHA)
	
	terminator_vector = utils.rotate_point(sun_direction, 0.5 * math.pi)
	full_occlusion_offset = (0.0, 0.0) if atmosphere_thickness == 0 else utils.vec_mul(utils.vec_unit(sun_direction), -atmosphere_thickness)
	
	def occluded_equation() -> Callable[[int, int], bool]:
		if terminator_vector[0] == 0:
			if sun_direction[0] > 0:
				return lambda u, v: math.sqrt((u - atmo_radius) ** 2 + (v - atmo_radius) ** 2) < atmo_radius and u < atmo_radius
			else:
				return lambda u, v: math.sqrt((u - atmo_radius) ** 2 + (v - atmo_radius) ** 2) < atmo_radius and u > atmo_radius
		m = terminator_vector[1] / terminator_vector[0]
		b = (1 - m) * atmo_radius
		if sun_direction[1] > 0:
			return lambda u, v: math.sqrt((u - atmo_radius) ** 2 + (v - atmo_radius) ** 2) < atmo_radius and v < m * u + b
		else:
			return lambda u, v: math.sqrt((u - atmo_radius) ** 2 + (v - atmo_radius) ** 2) < atmo_radius and v > m * u + b
	
	def occlusion_rate_equation(occl_eq: Callable[[int, int], bool]) -> Callable[[int, int], float]:
		if terminator_vector[0] == 0:
			return lambda u, v: 0 if not occl_eq(u, v) else min(1.0, math.fabs(u - atmosphere_thickness) / atmo_radius)
		m = terminator_vector[1] / terminator_vector[0]
		b0 = (1 - m) * atmo_radius
		b = atmo_radius + full_occlusion_offset[1] - m * (atmo_radius + full_occlusion_offset[0])
		return lambda u, v: 0 if not occl_eq(u, v) else min(1.0, math.fabs(v - (m * u + b0)) / math.fabs(b - b0)) if atmosphere_thickness > 0 else 1
	
	occlusion_rate_eq = occlusion_rate_equation(occluded_equation())
	max_alpha = 255 - ambient_light
	
	pixels = pygame.PixelArray(tmp_surface)
	for x in range(tmp_surface.get_width()):
		for y in range(tmp_surface.get_height()):
			'''if x == round(atmo_radius) and 100 < y < atmo_radius:
				breakpoint()'''
			occlusion_rate = occlusion_rate_eq(x, y)
			color = (0, 0, 0, min(max_alpha, int(math.floor(occlusion_rate * max_alpha))))
			pixels[x, y] = color
	
	return RasterShape(tmp_surface)


def generate_asteroid_surface_detail(asteroid_shape: Polygon) -> RasterShape:
	minx = min(asteroid_shape.points, key=lambda p: p[0])
	maxx = max(asteroid_shape.points, key=lambda p: p[0])
	miny = min(asteroid_shape.points, key=lambda p: p[1])
	maxy = max(asteroid_shape.points, key=lambda p: p[1])
	
	size = int(math.ceil(maxx[0]-minx[0]))+1, int(math.ceil(maxy[1]-miny[1]))+1
	
	mask_raster = pygame.Surface(size, pygame.SRCALPHA)
	asteroid_shape.draw(mask_raster, translation=utils.vec_mul(size, 0.5))
	asteroid_mask: pygame.mask.MaskType = pygame.mask.from_surface(mask_raster)
	tmp_raster = pygame.Surface(size, pygame.SRCALPHA)
	tmp_raster.fill((0, 0, 0, 0))
	crater_count = 1 if size[0]*size[1] < 128 else random.randrange((size[0]*size[1])//100, (size[0]*size[1])//64)
	crater_size_range = min(size)//8, min(size)//3
	
	for crater in range(crater_count):
		x = random.randrange(0, size[0])
		y = random.randrange(0, size[1])
		while not asteroid_mask.get_at((x, y)):
			x = random.randrange(0, size[0])
			y = random.randrange(0, size[1])
		radius = random.randint(crater_size_range[0], crater_size_range[1])
		alpha = random.randrange(96, 192)
		pygame.gfxdraw.filled_circle(tmp_raster, x, y, radius, (0, 0, 0, alpha))
		pygame.gfxdraw.aacircle(tmp_raster, x, y, radius, (0, 0, 0, alpha))
		
	blurred = tmp_raster  # gauss_blur(tmp_raster)
	masked = pygame.Surface(size, pygame.SRCALPHA)
	blurred_pixels = pygame.PixelArray(blurred)
	masked_pixels = pygame.PixelArray(masked)
	for x in range(size[0]):
		for y in range(size[1]):
			if asteroid_mask.get_at((x, y)):
				masked_pixels[x, y] = blurred_pixels[x, y]
			else:
				masked_pixels[x, y] = (0, 0, 0, 0)
	blurred_pixels.close()
	masked_pixels.close()
	return RasterShape(masked, rotate_enabled=True)
	


if __name__ == "__main__":
	pygame.init()
	test_display = pygame.display.set_mode((640, 480))
	test_display.fill((0, 255, 255))
	import sys
	
	# continents.draw(test_display, translation=(160, 120))
	
	# pygame.draw.circle(test_display, (0, 0, 255), (160, 120), 100)
	# craters = generate_craters(100)
	# craters.draw(test_display, translation=(160, 120))
	# breakpoint()
	'''term = generate_terminator(200, 20)
	continents = generate_continents(200, (
		(0.0, 0.3, (255, 255, 255), (255, 255, 255)), (0.4, 0.7, (0, 128, 0), (128, 255, 0)), (0.7, 1.0, (128, 255, 0), (255, 255, 64))))
	lights = generate_night_lights(200, continents)'''
	
	rot = 0
	continents_only = False
	terminator_only = False
	
	midpoint = (320, 240)
	
	clouds = generate_clouds(200, independent_rotation=-0.125*math.pi, gauss_blurring=4)
	
	while 1:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.display.quit()
				sys.exit()
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE:
					continents_only = True
				elif event.key == pygame.K_BACKSPACE:
					terminator_only = True
			if event.type == pygame.KEYUP:
				if event.key == pygame.K_SPACE:
					continents_only = False
				elif event.key == pygame.K_BACKSPACE:
					terminator_only = False
		#rot = rot + (20 / 1000) * math.pi / 3
		test_display.fill((128, 128, 128))
		
		# continents.draw(test_display, rotation=rot, translation=midpoint)
		# term.draw(test_display, translation=midpoint)
		# lights.draw(test_display, rotation=rot, translation=midpoint)
		pygame.gfxdraw.aacircle(test_display, 320, 240, 200, (64, 64, 64))
		pygame.gfxdraw.filled_circle(test_display, 320, 240, 200, (0, 255, 255))
		#clouds.draw(test_display, rotation=rot, translation=midpoint)
		clouds.draw(test_display, translation=midpoint)
		clouds.tick(20)
		
		pygame.display.flip()
		
		pygame.time.wait(20)
