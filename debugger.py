from typing import Tuple, Union, Iterable, Set

import pygame.gfxdraw
import pygame
import utils
import math
from game_settings import GameSettings


class Debugger:
	surface: pygame.SurfaceType = None
	
	@staticmethod
	def set_surface(surface: pygame.SurfaceType):
		Debugger.surface = surface
	
	class VecViz:
		def __init__(
				self,
				origin: Tuple[float, float],
				vector: Tuple[float, float],
				color: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (255, 255, 255),
				lifespan: float = GameSettings.tick):
			self.origin = origin
			self.vector = vector
			self.color = color
			self.remaining_lifespan = lifespan
	
	vectors: Set[VecViz] = set()
	
	indicator_points = (
		(0, 0),
		(0.8, -0.2),
		(1, 0),
		(0.8, 0.2)
	)
	
	@staticmethod
	def add_vector(
			origin: Tuple[float, float],
			vector: Tuple[float, float],
			color: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (255, 255, 255),
			lifespan: float = GameSettings.tick):
		if GameSettings.debugger_enabled:
			Debugger.vectors.add(Debugger.VecViz(origin, vector, color, lifespan))
		
	@staticmethod
	def draw_vectors():
		for v in Debugger.vectors:
			length, angle = utils.to_polar(v.vector)
			indicator = utils.transform_polygon(Debugger.indicator_points, (length, math.sqrt(length)), angle, v.origin)
			pygame.gfxdraw.filled_polygon(Debugger.surface, indicator, v.color)
			
	@staticmethod
	def tick(d: float = GameSettings.tick):
		to_remove = set()
		for v in Debugger.vectors:
			v.remaining_lifespan -= d
			if v.remaining_lifespan <= 0:
				to_remove.add(v)
				
		for expired in to_remove:
			Debugger.vectors.remove(expired)