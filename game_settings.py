import math
import pygame


class GameSettings:
	screen_size = 1280, 720
	fullscreen = False
	g_constant = 5
	ship_thrust = 10  # pixel/tick^2
	ship_max_rotation = math.radians(180)  # rad/s
	ship_rcs_rotation_rate = math.radians(10)  # rad/tick
	ship_rcs_thrust = 3  # pixel/tick^2
	ship_mass = 200
	ship_fire_delay = 200  # ms
	ship_shield_max_charge = 500
	ship_shield_recharge_cooldown = 3000  # ms
	ship_shield_recharge_rate = 100  # /s
	ship_autopilot_angle_tolerance = math.radians(5)
	ship_autopilot_speed_tolerance = 10
	ship_manual_control_keys = {
		pygame.K_UP,
		pygame.K_DOWN,
		pygame.K_LEFT,
		pygame.K_RIGHT,
		pygame.K_w,
		pygame.K_a,
		pygame.K_s,
		pygame.K_d
	}
	tick = 20  # ms
	screen_side_angles = (
		math.atan2(-screen_size[1]/2, screen_size[0]/2),
		math.atan2(screen_size[1]/2, screen_size[0]/2),
		math.atan2(screen_size[1]/2, -screen_size[0]/2),
		math.atan2(-screen_size[1]/2, -screen_size[0]/2),
	)
	bullet_radius = 2
	bullet_speed = 500
	bullet_color = (255, 0, 255)
	bullet_consumed_mass = 150
	bullet_explosion_strength = 250
	asteroid_min_mass = 250
	asteroid_max_mass = 1000
	asteroid_density = 0.01
	asteroid_color = (96, 64, 0)
	debugger_enabled = True
	kessler_rv_threshold = 200
	kessler_min_mass = 50
	