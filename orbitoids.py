from typing import List

import pygame

import graphics
import orbitoids_objects as oo
import sys

import utils
from debugger import Debugger
from renderer import render
import physics
import random
import math

from game_settings import GameSettings

planets: List[oo.Planet] = list()


def spawn_asteroid():
	planet_pos = planets[0].position
	max_distance = int(round(min(planet_pos[0], planet_pos[1], GameSettings.screen_size[0]-planet_pos[0], GameSettings.screen_size[1]-planet_pos[1])))
	r = random.randrange(int(round(2*planets[0].radius))+10, max_distance)
	theta = math.radians(random.randrange(360))
	position = utils.vec_plus(utils.to_cartesian((r, theta)), planet_pos)
	velocity = random.randrange(-20, 20), random.randrange(-20, 20)
	rotation = math.radians(random.randrange(-20, 20))
	asteroid = oo.Asteroid(position, velocity, initial_rotation=rotation, min_mass=GameSettings.asteroid_min_mass, max_mass=GameSettings.asteroid_max_mass)
	
	eccentric = bool(random.getrandbits(1))
	eccentricity = 0 if not eccentric else random.gauss(0.25, 0.15)
	at_periapsis = bool(random.getrandbits(1))
	max_eccentricity = 0.9 if at_periapsis else (1-((planets[0].radius+10)/r)**2)**0.5
	if eccentricity > max_eccentricity:
		eccentricity = max_eccentricity
	if eccentric and eccentricity < 0.1:
		eccentricity = 0.1
	physics.stable_orbit(asteroid, planets[0], eccentricity, at_periapsis)


def game():
	pygame.init()
	if GameSettings.fullscreen:
		screen = pygame.display.set_mode(GameSettings.screen_size, pygame.HWSURFACE | pygame.FULLSCREEN)
	else:
		screen = pygame.display.set_mode(GameSettings.screen_size, pygame.HWSURFACE)
	Debugger.set_surface(screen)
	global planets
	planets = [oo.Moon(80, 3000), oo.Moon(15, 100, (GameSettings.screen_size[0] // 2, GameSettings.screen_size[1] // 2 - 200), (192, 192, 192))]
	
	continents = graphics.generate_continents(
		80,
		(
			(0.0, 0.2, (255, 255, 255), (255, 255, 255)),
			(0.4, 0.6, (0, 128, 0), (128, 255, 0)),
			(0.6, 0.8, (255, 192, 0), (255, 255, 64)),
			(0.8, 1.0, (0, 128, 0), (0, 192, 64))
		))
	planets[0].attach_object(continents)
	
	lights = graphics.generate_night_lights(80, continents)
	planets[0].attach_object(lights)
	
	cyclones = graphics.generate_clouds(85, 80, independent_rotation=-math.pi/20, gauss_blurring=2)
	planets[0].attach_object(cyclones)

	terminator = graphics.generate_terminator(90, 10)
	planets[0].attach_object(terminator)

	atmosphere = graphics.generate_atmosphere(80, 10)
	planets[0].attach_object(atmosphere)
	
	planets[0].rotation = 0.05
	
	craters = graphics.generate_craters(
		15,
		cratered_surface=(0.6, 0.9)
	)
	planets[1].attach_object(craters)
	
	moon_terminator = graphics.generate_terminator(17, 0)
	planets[1].attach_object(moon_terminator)
	
	moon_lights = graphics.generate_night_lights(15, inhabited_area=0.35, urbanization_factor=0.75)
	planets[1].attach_object(moon_lights)
	
	physics.stable_orbit(planets[1], planets[0])
	physics.tidal_lock(planets[0], planets[1])
	
	ship = oo.Spacheship((GameSettings.screen_size[0]//2, 30), (0, 0))
	physics.stable_orbit(ship, physics.barycenter(planets))
	pygame.font.init()
	ship.thrusting = False
	clock = pygame.time.Clock()
	
	quitting = False
	ap_last_calculated = 0
	
	while not quitting:
		if not pygame.display.get_init():
			break
		
		clock.tick(1000/GameSettings.tick)
		
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				quitting = True
			if event.type == pygame.KEYDOWN:
				if event.key in GameSettings.ship_manual_control_keys:
					ship.autopilot_off()
					ap_last_calculated = 0
				if event.key == pygame.K_ESCAPE:
					pygame.quit()
				if event.key == pygame.K_UP:
					ship.thrusting = True
				if event.key == pygame.K_LEFT:
					ship.turn(False)
				if event.key == pygame.K_RIGHT:
					ship.turn(True)
				if event.key == pygame.K_SPACE:
					ship.shooting = True
				if event.key == pygame.K_BACKSPACE:
					spawn_asteroid()
				if event.key == pygame.K_w:
					ship.rcs_moving[graphics.SHIP_RCS_MOVE_FORWARD] = True
				if event.key == pygame.K_a:
					ship.rcs_moving[graphics.SHIP_RCS_MOVE_LEFT] = True
				if event.key == pygame.K_s:
					ship.rcs_moving[graphics.SHIP_RCS_MOVE_BACK] = True
				if event.key == pygame.K_d:
					ship.rcs_moving[graphics.SHIP_RCS_MOVE_RIGHT] = True
				if event.key == pygame.K_q:
					ship.autopilot_enabled = True
					ap_last_calculated = 0
					
			elif event.type == pygame.KEYUP:
				if event.key == pygame.K_UP:
					ship.thrusting = False
				if event.key == pygame.K_RIGHT:
					ship.stop_turning()
				if event.key == pygame.K_LEFT:
					ship.stop_turning()
				if event.key == pygame.K_SPACE:
					ship.shooting = False
				if event.key == pygame.K_w:
					ship.rcs_moving[graphics.SHIP_RCS_MOVE_FORWARD] = False
				if event.key == pygame.K_a:
					ship.rcs_moving[graphics.SHIP_RCS_MOVE_LEFT] = False
				if event.key == pygame.K_s:
					ship.rcs_moving[graphics.SHIP_RCS_MOVE_BACK] = False
				if event.key == pygame.K_d:
					ship.rcs_moving[graphics.SHIP_RCS_MOVE_RIGHT] = False
		
		physics.tick_physics()
		if ship.autopilot_enabled:
			if ap_last_calculated <= 0:
				orbit_vector = physics.stable_orbit_vector(False, 0, physics.barycenter(planets), ship)
				ship.autopilot(orbit_vector)
				ap_last_calculated = 1000
			else:
				ap_last_calculated -= GameSettings.tick
		physics.Kinematic.tick_all()
		physics.center_planets(planets)
		
		physics.OnscreenObject.objects_lock.acquire()
		collision_enabled = list(filter(lambda o: o.collision_enabled, physics.OnscreenObject.objects))
		physics.OnscreenObject.objects_lock.release()
		
		for idx, object1 in enumerate(collision_enabled):
			if len(collision_enabled) <= idx+1:
				break
			for object2 in collision_enabled[idx+1:]:
				if oo.detect_collision(object1, object2):
					object1.collided(object2)
					object2.collided(object1)
		
		graphics.AnimatedShape.tick_all()
		
		if pygame.display.get_init():
			render(screen, planets, ship)
		Debugger.tick()
	
	pygame.display.quit()
	print("Bye!")
	sys.exit()


if __name__ == "__main__":
	game()
