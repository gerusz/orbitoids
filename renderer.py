import sys
from typing import Tuple, Iterable

import pygame
import pygame.gfxdraw as dr

import graphics
import orbitoids_objects as oo
import physics
from game_settings import GameSettings
import math
import random
import utils
from debugger import Debugger
from graphics import GameShapes


def draw_object(surface: pygame.SurfaceType, drawn_object: physics.OnscreenObject, scale: float = 1.0):
	if drawn_object.shape is not None:
		if len(drawn_object.attachments_below) > 0:
			for att_below in drawn_object.attachments_below:
				att_below.shape.draw(surface, scale, drawn_object.direction, drawn_object.position)
		drawn_object.shape.draw(surface, scale, drawn_object.direction, drawn_object.position)
		if len(drawn_object.attachments_above) > 0:
			for att_above in drawn_object.attachments_above:
				att_above.shape.draw(surface, scale, drawn_object.direction, drawn_object.position)


def draw_planet(surface: pygame.SurfaceType, planet: oo.Planet):
	draw_object(surface, planet)


def draw_ship(surface: pygame.SurfaceType, spaceship: oo.Spacheship):
	if utils.is_on_screen(spaceship.position, spaceship.collider_radius):
		draw_object(surface, spaceship)
		# spaceship.shape.draw(surface, rotation=spaceship.direction, translation=spaceship.position)
		
		if spaceship.thrusting:
			GameShapes.get_shape(graphics.SHIP_FLAME).draw(surface, rotation=spaceship.direction, translation=spaceship.position)
		
		if spaceship.turning:
			rcs = GameShapes.get_shape(graphics.SHIP_RCS_TURN_RIGHT) if spaceship.turning_right else GameShapes.get_shape(graphics.SHIP_RCS_TURN_LEFT)
			rcs.draw(surface, rotation=spaceship.direction, translation=spaceship.position)
			
		for (d, rcs_thrust) in spaceship.rcs_moving.items():
			if rcs_thrust:
				rcs = GameShapes.get_shape(d)
				rcs.draw(surface, rotation=spaceship.direction, translation=spaceship.position)
	
	else:
		indicator = GameShapes.get_shape(graphics.INDICATOR)
		speed_indicator = GameShapes.get_shape(graphics.VELOCITY_INDICATOR)
		heading_indicator = GameShapes.get_shape(graphics.DIRECTION_INDICATOR)
		indicator_data = utils.indicator_location(spaceship.position)
		indicator_position = (indicator_data[0], indicator_data[1])
		indicator.draw(surface, rotation=indicator_data[2], translation=indicator_position)
		
		speed_scale = 2 * abs(math.atan(math.hypot(spaceship.velocity[0], spaceship.velocity[1]) / 100)) / (0.5 * math.pi)
		speed_indicator.draw(surface, (1, speed_scale), spaceship.velocity, indicator_position)
		
		heading_indicator.draw(surface, rotation=spaceship.direction, translation=indicator_position)


starfield: Tuple[Tuple[Tuple[float, float], graphics.Shape], ...] = None
starfield_bitmap: pygame.SurfaceType = None


def generate_starfield(star_count=500, min_star_size=1, max_star_size=3, min_saturation=0, max_saturation=32):
	width, height = GameSettings.screen_size
	stars = []
	for n in range(star_count):
		star_x = random.randrange(width)
		star_y = random.randrange(height)
		star_size = random.randint(min_star_size, max_star_size)
		saturation = random.randint(min_saturation, max_saturation)
		hue_vector = (0, 0, 0)
		hue_vector_len = math.hypot(hue_vector[0], hue_vector[1], hue_vector[2])
		while hue_vector_len == 0:
			hue_vector = random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)
			hue_vector_len = math.hypot(hue_vector[0], hue_vector[1], hue_vector[2])
		norm_factor = saturation / (2 * hue_vector_len)
		hue_deltas = hue_vector[0] * norm_factor, hue_vector[1] * norm_factor, hue_vector[2] * norm_factor
		color_offset = max(hue_deltas)
		star_color = 255 - color_offset + hue_deltas[0], 255 - color_offset + hue_deltas[1], 255 - color_offset + hue_deltas[2], 192
		star_points = random.randint(5, 9)
		star_angle = 2 * math.pi / star_points
		star_halfangle = star_angle / 2
		inside_factor = random.randrange(2, 11)/10.0
		star_poly_points = []
		for i in range(star_points):
			star_poly_points.append(utils.to_cartesian((star_size, i*star_angle)))
			star_poly_points.append(utils.to_cartesian((inside_factor*star_size, i*star_angle+star_halfangle)))
		# Pre-transform the polygon
		rotation = math.radians(random.randrange(360))
		star_poly = utils.transform_polygon(star_poly_points, rotation=rotation)
		star_halo = utils.transform_polygon(star_poly, scale=2)
		star_background_radius = star_size*(1-inside_factor)
		star_background_color = (star_color[0], star_color[1], star_color[2], 128)
		star_halo_color = (star_color[0], star_color[1], star_color[2], 128)
		stars.append(
			(
				(star_x, star_y),
				graphics.ComplexShape(
					(
						graphics.Polygon(star_halo, fill_color=star_halo_color, contour=False),
						graphics.Circle(star_background_radius, filled=True, fill_color=star_background_color, contour=False),
						graphics.Polygon(star_poly, fill_color=star_color, contour=False)
					)
				)
			)
		)
	return tuple(stars)


def draw_background(surface: pygame.SurfaceType):
	global starfield
	global starfield_bitmap
	if starfield is None:
		starfield = generate_starfield(max_saturation=96)
		starfield_bitmap = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
		for star in starfield:
			star[1].draw(starfield_bitmap, translation=star[0])
	surface.blit(starfield_bitmap, (0, 0))


def draw_bullet(surface: pygame.SurfaceType, bullet: oo.Bullet):
	if utils.is_on_screen(bullet.position):
		draw_object(surface, bullet)
		# bullet.shape.draw(surface, translation=bullet.position, rotation=bullet.direction)


def draw_asteroid(surface: pygame.SurfaceType, asteroid: oo.Asteroid):
	if utils.is_on_screen(asteroid.position, asteroid.collider_radius):
		draw_object(surface, asteroid)
		# asteroid.shape.draw(surface, rotation=asteroid.direction, translation=asteroid.position)
	else:
		asteroid_indicator = GameShapes.get_shape(graphics.ASTEROID_INDICATOR)
		asteroid_indicator_data = utils.indicator_location(asteroid.position, 6)
		asteroid_indicator_position = asteroid_indicator_data[0], asteroid_indicator_data[1]
		if asteroid.position[0] < 0 or asteroid.position[0] > GameSettings.screen_size[0]:
			indicator_scale_x = math.fabs((GameSettings.screen_size[0]/2) / (asteroid.position[0]-GameSettings.screen_size[0]/2))
		else:
			indicator_scale_x = 1.0
		if asteroid.position[1] < 0 or asteroid.position[1] > GameSettings.screen_size[1]:
			indicator_scale_y = math.fabs((GameSettings.screen_size[1]/2) / (asteroid.position[1]-GameSettings.screen_size[1]/2))
		else:
			indicator_scale_y = 1.0
		asteroid_indicator.draw(surface, scale=(indicator_scale_x, indicator_scale_y), rotation=asteroid_indicator_data[2], translation=asteroid_indicator_position)


font = None


def draw_hud(surface: pygame.SurfaceType, spaceship: oo.Spacheship):
	velocity = "v   = {:.2f} (vx = {:.2f}, vy = {:.2f})".format(math.hypot(spaceship.velocity[0], spaceship.velocity[1]), spaceship.velocity[0],
	                                                            spaceship.velocity[1])
	position = "pos = ({:.2f}, {:.2f})".format(spaceship.position[0], spaceship.position[1])
	direction = "dir = {:.2f}°".format(math.degrees(math.atan2(spaceship.velocity[1], spaceship.velocity[0])))
	heading = "hdg = {:.2f}°".format(math.degrees(math.atan2(spaceship.direction[1], spaceship.direction[0])))
	shield_charge = "shd = {:.2f} / {:.2f}".format(spaceship.shield_charge, spaceship.shield_max_charge)
	global font
	if font is None:
		font = pygame.font.SysFont(pygame.font.get_default_font(), 12, True)
	vel_surf = font.render(velocity, True, (255, 255, 255))
	pos_surf = font.render(position, True, (255, 255, 255))
	dir_surf = font.render(direction, True, (255, 255, 255))
	hdg_surf = font.render(heading, True, (255, 255, 255))
	shd_surf = font.render(shield_charge, True, (255, 255, 255))
	pygame.draw.rect(surface, (0, 0, 0), pygame.Rect(0, 0, vel_surf.get_width(), 72))
	surface.blit(vel_surf, (10, 12))
	surface.blit(pos_surf, (10, 24))
	surface.blit(dir_surf, (10, 36))
	surface.blit(hdg_surf, (10, 48))
	surface.blit(shd_surf, (10, 60))

	
def draw_radar(surface: pygame.SurfaceType, planets: Iterable[oo.Planet], spaceship: oo.Spacheship):
	radar_surface = pygame.Surface((GameSettings.screen_size[0]//4, GameSettings.screen_size[1]//4), pygame.SRCALPHA)
	pygame.gfxdraw.filled_polygon(radar_surface, ((0, 0), (radar_surface.get_width(), 0), radar_surface.get_size(), (0, radar_surface.get_height())), (0, 64, 0, 128))
	
	def radar_coordinates(coordinates):
		off_center = utils.vec_minus(coordinates, (GameSettings.screen_size[0]/2, GameSettings.screen_size[1]/2))
		scaled_coordinates = utils.vec_plus((GameSettings.screen_size[0]//8, GameSettings.screen_size[1]//8), utils.vec_mul(off_center, 0.0625))
		return int(round(scaled_coordinates[0])), int(round(scaled_coordinates[1]))
	
	for planet in planets:
		radar_pos = radar_coordinates(planet.position)
		planet.shape.draw(radar_surface, 0.0625, 0, radar_pos)
	
	physics.Kinematic.kinematic_objects_lock.acquire()
	asteroids = filter(lambda o: isinstance(o, oo.Asteroid), physics.Kinematic.kinematic_objects)
	physics.Kinematic.kinematic_objects_lock.release()
	
	for asteroid in asteroids:
		radar_pos = radar_coordinates(asteroid.position)
		pygame.gfxdraw.filled_circle(radar_surface, radar_pos[0], radar_pos[1], 1, GameSettings.asteroid_color)
	
	ship_position = radar_coordinates(spaceship.position)
	pygame.gfxdraw.filled_circle(radar_surface, ship_position[0], ship_position[1], 2, graphics.SHIP_HULL_COLOR)
	
	for rect_size in range(0, 4):
		left, top = radar_coordinates((-0.5 * rect_size * GameSettings.screen_size[0], -0.5 * rect_size * GameSettings.screen_size[1]))
		right, bottom = radar_coordinates(((rect_size * 0.5 + 1) * GameSettings.screen_size[0], (rect_size * 0.5 + 1) * GameSettings.screen_size[1]))
		if rect_size > 0:
			pygame.gfxdraw.rectangle(radar_surface, pygame.Rect(left, top, right-left, bottom-top), (0, 128, 0, 128))
		else:
			pygame.gfxdraw.rectangle(radar_surface, pygame.Rect(left, top, right-left, bottom-top), (255, 255, 255, 128))
	
	pygame.gfxdraw.line(radar_surface, radar_surface.get_width() // 2, 0, radar_surface.get_width() // 2, radar_surface.get_height(), (0, 128, 0, 128))
	pygame.gfxdraw.line(radar_surface, 0, radar_surface.get_height() // 2, radar_surface.get_width(), radar_surface.get_height() // 2, (0, 128, 0, 128))
	pygame.gfxdraw.rectangle(radar_surface, radar_surface.get_rect(), (0, 255, 0))
	
	surface.blit(radar_surface, (surface.get_width()-radar_surface.get_width(), surface.get_height()-radar_surface.get_height()))


def render(surface: pygame.SurfaceType, planets: Iterable[oo.Planet], spaceship: oo.Spacheship):
	surface.fill((0, 0, 0))
	draw_background(surface)
	
	for planet in planets:
		draw_planet(surface, planet)
	
	physics.Kinematic.kinematic_objects_lock.acquire()
	bullets = filter(lambda o: isinstance(o, oo.Bullet), physics.Kinematic.kinematic_objects)
	asteroids = filter(lambda o: isinstance(o, oo.Asteroid), physics.Kinematic.kinematic_objects)
	physics.Kinematic.kinematic_objects_lock.release()
	
	for bullet in bullets:
		draw_bullet(surface, bullet)
	for asteroid in asteroids:
		draw_asteroid(surface, asteroid)
	draw_ship(surface, spaceship)
	draw_hud(surface, spaceship)
	draw_radar(surface, planets, spaceship)
	Debugger.draw_vectors()
	pygame.display.flip()


if __name__ == "__main__":
	screen = pygame.display.set_mode(GameSettings.screen_size)
	planet = oo.Planet(80, 300)
	ship = oo.Spacheship((30, 30), (0, 10))
	while 1:
		pygame.time.wait(20)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.display.quit()
				sys.exit()
		ship.apply_acceleration(planet.gravity_force(ship.position))
		ship.tick(20)
		render(screen, [planet], ship)
