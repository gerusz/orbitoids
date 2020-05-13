from typing import Tuple, List, Union

import pygame

import utils
from game_settings import GameSettings
import math
import random
import graphics

from physics import OnscreenObject, Massive, Kinematic
import physics
from debugger import Debugger


def detect_collision(one: OnscreenObject, two: OnscreenObject) -> bool:
	return math.hypot(one.position[0] - two.position[0], one.position[1] - two.position[1]) <= one.collider_radius + two.collider_radius


class Planet(Massive):
	
	def __init__(
			self,
			radius: float,
			mass: float,
			position: Tuple[int, int] = (GameSettings.screen_size[0] // 2, GameSettings.screen_size[1] // 2),
			color: Tuple[int, int, int] = (0, 128, 255)):
		Massive.__init__(self, mass, position, 0, graphics.Circle(radius, fill_color=color))
		self.radius = radius
		self.collider_radius = radius
		self.collision_enabled = True
		self.color = color


class Spacheship(Kinematic):
	
	def __init__(
			self,
			position: Tuple[float, float],
			initial_velocity: Tuple[float, float] = (0, 0),
			initial_direction: Tuple[float, float] = (0, 1)):
		
		shape = graphics.GameShapes.get_shape(graphics.SHIP)
		super().__init__(position, initial_velocity, initial_direction, 0, GameSettings.ship_mass, 400, shape=shape)
		self.thrusting = False
		self.turning = False
		self.turning_right = False
		self.collider_radius = math.sqrt(2) * 20 + 5
		self.rcs_moving = {
			graphics.SHIP_RCS_MOVE_FORWARD: False,
			graphics.SHIP_RCS_MOVE_BACK: False,
			graphics.SHIP_RCS_MOVE_LEFT: False,
			graphics.SHIP_RCS_MOVE_RIGHT: False
		}
		self.target_rotation = 0.0
		self.firing_delay: int = 0
		self.shooting = False
		
		self.shield_max_charge = GameSettings.ship_shield_max_charge
		self.shield_charge = GameSettings.ship_shield_max_charge
		self.shield_recharge_rate = GameSettings.ship_shield_recharge_rate
		self.shield_recharge_cooldown = GameSettings.ship_shield_recharge_cooldown
		
		shield_below_shape = graphics.generate_shield(self.collider_radius + 5, 1.0)
		shield_above_shape = graphics.generate_shield(self.collider_radius + 5, 1.0)
		
		shield_below_shape.offset = (-4, 0)
		shield_above_shape.offset = (-4, 0)
		
		self.shield_below = OnscreenObject((0, 0), 0, shield_below_shape)
		self.shield_above = OnscreenObject((0, 0), 0, shield_above_shape)
		self.attach_object(self.shield_below, True)
		self.attach_object(self.shield_above)
		self.shield_last_redrawn_at = 1.0
		
		self.autopilot_enabled = False
		self.autopilot_target = (0, 0)
	
	def thrust(self, ratio: float = 1.0):
		thrust_force = utils.rotate_point((ratio * GameSettings.ship_thrust, 0), self.direction)
		self.apply_acceleration(thrust_force)
		self.thrusting = True
	
	def turn(self, right: bool = False):
		self.target_rotation = GameSettings.ship_max_rotation if right else -GameSettings.ship_max_rotation
		
	def stop_turning(self):
		self.target_rotation = 0
		self.stabilize()
			
	def stabilize(self, target_rotation: Union[None, float] = None):
		tr = self.target_rotation if target_rotation is None else target_rotation
		if 0 < math.fabs(self.rotation - tr) < GameSettings.ship_rcs_rotation_rate:
			self.rotation = tr
			self.turning = False
		elif not self.rotation == tr:
			self.rotation += GameSettings.ship_rcs_rotation_rate if self.turning_right else -GameSettings.ship_rcs_rotation_rate
			self.turning = True
			self.turning_right = (self.rotation < tr)
			
	def tick(self, dt=GameSettings.tick):
		if self.thrusting:
			self.thrust()
		self.rcs_thrust()
		if self.firing_delay > 0:
			self.firing_delay -= dt
		elif self.shooting:
			self.shoot()
		if self.shield_charge < self.shield_max_charge:
			if self.shield_recharge_cooldown == 0:
				self.recharge_shield(dt)
			else:
				self.shield_recharge_cooldown -= dt
		if self.autopilot_enabled:
			self.autopilot(self.autopilot_target)
		self.stabilize()
		super().tick(dt)
		
	def recharge_shield(self, dt=GameSettings.tick):
		self.shield_charge += self.shield_recharge_rate * (dt / 1000)
		if self.shield_charge >= self.shield_max_charge:
			self.shield_charge = self.shield_max_charge
			self.shield_recharge_cooldown = GameSettings.ship_shield_recharge_cooldown
		if math.fabs(self.shield_charge/self.shield_max_charge - self.shield_last_redrawn_at) > 0.025 or self.shield_charge == self.shield_max_charge:
			#  Redraws are a bit spread out because regenerating the shape takes some CPU
			self.shield_last_redrawn_at = self.shield_charge/self.shield_max_charge
			self.regen_shield_shape()
		
	def regen_shield_shape(self):
		shield_charge_rate = self.shield_charge/self.shield_max_charge
		self.shield_below.shape = graphics.generate_shield(self.collider_radius + 5, shield_charge_rate, original_shape=self.shield_below.shape)
		self.shield_above.shape = graphics.generate_shield(self.collider_radius + 5, shield_charge_rate, original_shape=self.shield_above.shape)
		
	def rcs_thrust(self):
		rel_thrust = [0.0, 0.0]
		for d, thrusting in self.rcs_moving.items():
			if not thrusting:
				continue
			if d == graphics.SHIP_RCS_MOVE_RIGHT:
				rel_thrust[1] += GameSettings.ship_rcs_thrust
			if d == graphics.SHIP_RCS_MOVE_LEFT:
				rel_thrust[1] -= GameSettings.ship_rcs_thrust
			if d == graphics.SHIP_RCS_MOVE_FORWARD:
				rel_thrust[0] += GameSettings.ship_rcs_thrust
			if d == graphics.SHIP_RCS_MOVE_BACK:
				rel_thrust[0] -= GameSettings.ship_rcs_thrust
				
		thrust_vector = utils.rotate_point(tuple(rel_thrust), self.direction)
		self.apply_acceleration(thrust_vector)
		
	def reset_rcs(self):
		for d in self.rcs_moving.keys():
			self.rcs_moving[d] = False
	
	def shoot(self):
		bullet_location = (max(25.0, self.collider_radius+5), 0)
		bullet_velocity = (GameSettings.bullet_speed, 0)
		rotation_matrix = utils.rotation_matrix(self.direction)
		bullet_location = utils.vec_plus(utils.mul_matrix(bullet_location, rotation_matrix), self.position)
		bullet_velocity = utils.vec_plus(self.velocity, utils.mul_matrix(bullet_velocity, rotation_matrix))
		
		rotation = self.rotation
		if self.turning:
			if self.turning_right:
				rotation += GameSettings.ship_max_rotation
			else:
				rotation -= GameSettings.ship_max_rotation
		angular_velocity = 0, (self.collider_radius + 1) * rotation
		angular_velocity = utils.rotate_point(angular_velocity, self.direction)
		bullet_velocity = utils.vec_plus(bullet_velocity, angular_velocity)
		Bullet(bullet_location, bullet_velocity, initial_direction=self.direction)
		self.firing_delay = GameSettings.ship_fire_delay
	
	def damage(self, damage: float):
		if self.shield_charge > 0:
			self.shield_charge -= damage
			if self.shield_charge <= 0:
				self.shield_charge = 0
			self.shield_recharge_cooldown = GameSettings.ship_shield_recharge_cooldown
			self.regen_shield_shape()
		# Hull damage TBD
	
	def autopilot(self, target_vector: Tuple[float, float]):
		Debugger.add_vector(self.position, target_vector, (255, 0, 255))
		print("AP*** Autopilot tick")
		self.reset_rcs()
		self.autopilot_target = target_vector
		target_direction = utils.vec_minus(target_vector, self.velocity)
		Debugger.add_vector(self.position, target_direction, (255, 0, 0))
		
		if utils.vec_len(target_direction) < GameSettings.ship_autopilot_speed_tolerance:
			self.target_rotation = 0
			self.autopilot_off()
			self.reset_rcs()
			return
		
		tgt_angle = utils.vec_angle(target_direction)
		dir_angle = utils.vec_angle(self.direction)
		print("AP*** Target angle: {}, current direction: {}, tolerance: {}".format(tgt_angle, dir_angle, GameSettings.ship_autopilot_angle_tolerance))
		if math.fabs(dir_angle-tgt_angle) < GameSettings.ship_autopilot_angle_tolerance:
			print("AP*** Close enough to target direction - adjusting thrust")
			# Direction matches - kill rotation
			self.target_rotation = 0
			# Reset RCS thrusters
			self.reset_rcs()
			
			# See what needs to be done with the velocities
			indir_comp, tan_comp = utils.vec_decompose(utils.vec_mul(self.velocity, -1), target_direction)
			print("AP*** Velocity: {}, target difference: {}".format(self.velocity, target_direction))
			print("AP*** Components: {}, {}".format(indir_comp, tan_comp))
			tan_comp_polar = utils.to_polar(tan_comp)
			print("AP*** Tangent component polar: {}".format(tan_comp_polar))
			if tan_comp_polar[0] > GameSettings.ship_rcs_thrust:
				print("AP*** Lateral thrust enabled")
				# RCS right-left thrust
				rcs_right = utils.angle_wrap(tan_comp_polar[1]-dir_angle) < math.pi
				if rcs_right:
					self.rcs_moving[graphics.SHIP_RCS_MOVE_RIGHT] = True
					print("AP*** lateral thrust right")
				else:
					self.rcs_moving[graphics.SHIP_RCS_MOVE_LEFT] = True
					print("AP*** lateral thrust left")
			
			indir_comp_polar = utils.to_polar(indir_comp)
			target_speed = utils.vec_len(target_vector)
			angle_difference = utils.angle_wrap(tgt_angle - indir_comp_polar[1])
			print("AP*** angle difference = {} ({}°)".format(angle_difference, math.degrees(angle_difference)))
			if math.fabs(utils.vec_len(self.velocity) - target_speed) < GameSettings.ship_autopilot_speed_tolerance:
				# Current vector within the tolerances
				self.autopilot_off()
			elif math.pi * 1.5 < angle_difference or angle_difference < math.pi*0.5:
				print("AP*** Good direction - speed difference is {}".format(target_speed - indir_comp_polar[0]))
				# This means the ship is slower than the target vector
				if target_speed - indir_comp_polar[0] > GameSettings.ship_thrust:
					self.thrust()
				else:
					self.rcs_moving[graphics.SHIP_RCS_MOVE_FORWARD] = True
			elif math.pi*1.5 > angle_difference > math.pi*0.5:
				print("AP*** Too fast, thrusting back.")
				# This means the ship is faster than the target vector
				self.rcs_moving[graphics.SHIP_RCS_MOVE_BACK] = True
			self.rcs_thrust()
		# Else the angle is very wrong
		else:
			self.thrusting = False
			self.reset_rcs()
			print("AP*** Far away from target direction - rotating")
			turn_right = utils.angle_wrap(dir_angle - tgt_angle) > math.pi
			if\
				math.fabs(self.rotation) > 0 and \
				math.fabs(self.rotation / GameSettings.ship_rcs_rotation_rate) > math.fabs((dir_angle-tgt_angle)/((GameSettings.tick/1000) * self.rotation)):
				print("AP*** Rotation dangerously high - slowing")
				target_rotation_scale = 0
			else:
				target_rotation_scale = GameSettings.ship_max_rotation * math.sqrt(math.fabs(dir_angle-tgt_angle)/math.pi)
			if target_rotation_scale >= GameSettings.ship_rcs_rotation_rate:
				self.target_rotation = (1 if turn_right else -1) * target_rotation_scale
				print("AP*** Target rotation = {}".format(self.target_rotation))
			else:
				self.target_rotation = 0
			self.turning_right = (self.rotation < self.target_rotation)
		
	def autopilot_off(self):
		self.autopilot_enabled = False
		self.reset_rcs()
		self.target_rotation = 0
		self.thrusting = False
		
	def collided(self, other):
		if isinstance(other, Planet):
			forces = physics.bounce(self, other)
			self.damage(utils.vec_len(forces[0][0]))


class Moon(Planet, Kinematic):
	def __init__(
			self,
			radius: float,
			mass: float,
			position: Tuple[int, int] = (GameSettings.screen_size[0] // 2, GameSettings.screen_size[1] // 2),
			color: Tuple[int, int, int] = (0, 128, 255),
			initial_velocity: Tuple[float, float] = (0, 0),
			initial_direction: Tuple[float, float] = (0, 1)):
		Planet.__init__(self, radius, mass, position, color)
		Kinematic.__init__(self, position, initial_velocity, initial_direction, mass=mass, shape=graphics.Circle(radius, fill_color=color), reactive=False)
		print("Created moon @ {}, r={}, m={}".format(self.position, self.radius, self.mass))
		self.collider_radius = radius


class Bullet(Kinematic):
	
	def __init__(self, position: Tuple[float, float], initial_velocity: Tuple[float, float] = (0, 0), initial_direction: Tuple[float, float] = (0, 1)):
		super().__init__(position, initial_velocity, initial_direction, shape=graphics.GameShapes.get_shape(graphics.BULLET), initial_rotation=math.pi)
		self.collider_radius = GameSettings.bullet_radius
		self.collision_enabled = True
	
	def collided(self, other):
		self.remove()
	#  For now
	
	def tick(self, dt=1000):
		super(Bullet, self).tick(dt)
		if not utils.is_on_screen(self.position, utils.vec_mul(GameSettings.screen_size, 2.0)):
			self.remove()


class Asteroid(Kinematic):
	def __init__(
			self,
			position: Tuple[float, float],
			initial_velocity: Tuple[float, float] = (0, 0),
			initial_direction: Tuple[float, float] = (0, 1),
			initial_rotation: float = 0.0,
			min_mass: float = -math.inf,
			max_mass: float = math.inf
	):
		
		moment_multiplier, outer_points, total_mass = self.generate_asteroid(max_mass, min_mass)
		
		self.points = outer_points
		super().__init__(
			position,
			initial_velocity,
			initial_direction,
			initial_rotation,
			total_mass,
			moment_multiplier,
			graphics.Polygon(
				outer_points,
				fill_color=GameSettings.asteroid_color
			)
		)
		self.collider_radius = max(utils.vec_len(p) for p in outer_points)
		self.ev_checked = False
		# Test
		self.attach_object(graphics.generate_asteroid_surface_detail(self.shape))
	
	def collided(self, other):
		self.ev_checked = False
		if isinstance(other, Planet):
			print("Asteroid collided with a planet")
			self.remove()
		elif isinstance(other, Kinematic):
			if isinstance(other, Bullet):
				# BOOM
				hit_angle = utils.vec_angle(utils.vec_minus(self.position, other.position))
				self.explode(hit_angle)
			elif other not in self.collided_this_tick:
				forces = physics.bounce(self, other)
				self.collided_this_tick.add(other)
				other.collided_this_tick.add(self)
				if isinstance(other, Spacheship):
					other.damage(utils.vec_len(forces[1][0]))
				
	def apply_acceleration(self, acceleration: Tuple[float, float], should_explode_if_large: bool = True):
		super().apply_acceleration(acceleration)
		if should_explode_if_large and utils.vec_len(acceleration) > GameSettings.kessler_rv_threshold:
			print("Kessler syndrome - acceleration {} larger than threshold {}".format(utils.vec_len(acceleration), GameSettings.kessler_rv_threshold))
			self.explode(utils.vec_angle(utils.vec_mul(acceleration, -1)), consumed_mass=GameSettings.kessler_min_mass, explosion_strength=math.sqrt(utils.vec_len(acceleration)))
	
	def explode(
			self,
			hit_angle: float,
			fragments: Union[int, Tuple[int, int]] = (2, 4),
			consumed_mass: float = GameSettings.bullet_consumed_mass,
			explosion_strength: float = GameSettings.bullet_explosion_strength):
		spawn_total_mass = self.mass - consumed_mass
		if spawn_total_mass <= 0:
			# Asteroid completely destroyed
			self.remove()
			print("Asteroid destroyed - total mass {} < consumed mass {}".format(self.mass, consumed_mass))
			return
		fragment_count = random.randint(fragments[0], fragments[1]) if isinstance(fragments, tuple) else random.randint(2, fragments)
		fragment_mass_distrib = list([random.gauss(1, 0.25) for x in range(fragment_count)])
		md_offset = 0.5 - min(fragment_mass_distrib)
		for i in range(len(fragment_mass_distrib)):
			fragment_mass_distrib[i] += md_offset
		total_mass_distrib = sum(fragment_mass_distrib)
		fragment_masses = list([
			(math.floor(frag_mass * spawn_total_mass / total_mass_distrib), math.ceil(frag_mass * spawn_total_mass / total_mass_distrib))
			for frag_mass
			in fragment_mass_distrib
		])
		
		asteroid_fragments = list([
			Asteroid(self.position, self.velocity, self.direction, self.rotation, fm[0], fm[1]) for fm in fragment_masses
		])
		distribution_length = sum([a.collider_radius for a in asteroid_fragments]) + 10 * len(asteroid_fragments)
		distribution_dir = utils.rotate_point((1, 0), hit_angle + 0.5 * math.pi)
		distribution_offset = utils.vec_mul(distribution_dir, -0.5 * distribution_length)
		current_offset = distribution_offset
		for af in asteroid_fragments:
			asteroid_offset_length = af.collider_radius + 5
			current_offset = utils.vec_plus(current_offset, utils.vec_mul(distribution_dir, asteroid_offset_length))
			af.position = utils.vec_plus(af.position, current_offset)
			current_offset = utils.vec_plus(current_offset, utils.vec_mul(distribution_dir, asteroid_offset_length))
		
		explosion_location = utils.to_cartesian((self.collider_radius, hit_angle))
		for af in asteroid_fragments:
			dv_portion = explosion_strength * af.mass / spawn_total_mass
			dv = utils.vec_mul(utils.vec_unit(utils.vec_minus(af.position, explosion_location)), dv_portion)
			af.apply_acceleration(dv, False)
		
		self.remove()
		print("Asteroid exploded to {} pieces. Hit angle was {}°, central explosion vector: {}".format(fragment_count, math.degrees(hit_angle), explosion_location))
	
	@staticmethod
	def generate_asteroid(max_mass, min_mass):
		blobs = random.randint(1, 3)
		points = []
		for b in range(blobs):
			radius = random.randint(400, 1000)
			offset = 0 if blobs == 1 else random.randint(radius // 4, radius // 2 + radius // 4)
			angle = random.randint(0, 360)
			radius = radius / 100.0
			offset = offset / 100.0
			angle = math.radians(angle)
			offset_vector = utils.mul_matrix((offset, 0), utils.rotation_matrix(angle))
			for i in range(8):
				point = utils.mul_matrix((radius, 0), utils.rotation_matrix(i * 0.25 * math.pi))
				points.append(utils.vec_plus(point, offset_vector))
		print("Generated {} points for {} blobs".format(len(points), blobs))
		points_polar = list([(utils.vec_len(point), utils.vec_angle(point)) for point in points])
		distance_ordering = list(range(len(points)))
		distance_ordering.sort(key=lambda x: points_polar[x][0], reverse=True)
		angle_intervals: List[utils.AngleInterval] = []
		outer_points_polar = []
		for idx in distance_ordering:
			point = points_polar[idx]
			if not any(interval.contains(point[1]) for interval in angle_intervals):
				prev_neighbor = (idx - 1) % len(distance_ordering)
				next_neighbor = (idx + 1) % len(distance_ordering)
				extends = False
				for interval in angle_intervals:
					if not extends and interval.ends_with(prev_neighbor) or interval.ends_with(next_neighbor):
						interval.extend(point[1], idx)
						outer_points_polar.append(point)
						extends = True
				if not extends:
					angle_intervals.append(utils.AngleInterval(point[1], idx))
					outer_points_polar.append(point)
			else:
				containing_interval = list(filter(lambda interval: interval.contains(point[1]), angle_intervals))[0]
				if math.fabs(point[0] - points_polar[containing_interval.min_angle_idx][0]) < 0.05:
					outer_points_polar.append(point)
		outer_points_polar.sort(key=lambda p: p[1])
		print("Culled inner points, {} remaining".format(len(outer_points_polar)))
		outer_points_polar_fuzzed = []
		for polar_point in outer_points_polar:
			distance_fuzz = random.randint(80, 120) / 100.0
			outer_points_polar_fuzzed.append((polar_point[0] * distance_fuzz, polar_point[1]))
		outer_points_polar_fuzzed.sort(key=lambda p: p[1])
		rescaled = True
		point_masses = list()
		while rescaled:
			# Calculate mass and CoM
			point_masses = list([0 for p in outer_points_polar_fuzzed])
			for i in range(len(outer_points_polar_fuzzed)):
				curr = outer_points_polar_fuzzed[i]
				prev = outer_points_polar_fuzzed[i - 1]
				angle = curr[1] - prev[1]
				if angle < 0:
					angle = angle + 2 * math.pi
				triangle_area = math.fabs(math.sin(angle)) * curr[0] * prev[0]
				# print("Triangle area: {}".format(triangle_area))
				triangle_mass = GameSettings.asteroid_density * triangle_area ** 1.5
				point_masses[i] += triangle_mass
				point_masses[i - 1] += triangle_mass
			
			total_mass = sum(point_masses)
			if total_mass < min_mass and math.fabs(total_mass - min_mass) > 0.01:
				scaling = (min_mass / total_mass) ** (1 / 3)
				print("Asteroid mass {} smaller than min mass {}. Rescaling by {}.".format(total_mass, min_mass, scaling))
				rescaled = True
			elif total_mass > max_mass and math.fabs(total_mass - max_mass) > 0.01:
				scaling = (max_mass / total_mass) ** (1 / 3)
				print("Asteroid mass {} greater than max mass {}. Rescaling by {}.".format(total_mass, max_mass, scaling))
				rescaled = True
			else:
				scaling = 1
				rescaled = False
			
			if rescaled:
				outer_points_polar_fuzzed = list([(p[0] * scaling, p[1]) for p in outer_points_polar_fuzzed])
		outer_points_cartesian = list([utils.mul_matrix((p[0], 0), utils.rotation_matrix(p[1])) for p in outer_points_polar_fuzzed])
		com = (0, 0)
		for i in range(len(outer_points_cartesian)):
			com = (com[0] + outer_points_cartesian[i][0] * point_masses[i] / total_mass, com[1] + outer_points_cartesian[i][1] * point_masses[i] / total_mass)
		com_offset = utils.vec_mul(com, -1)
		print("CoM: {}, CoM offset: {}".format(com, com_offset))
		outer_points = list([utils.vec_plus(p, com_offset) for p in outer_points_cartesian])
		moment_multiplier = 0.0
		for i in range(len(outer_points)):
			r = 0.5 * utils.vec_len(outer_points[i])
			m = point_masses[i]
			moment_multiplier += m * r ** 2
		return moment_multiplier, outer_points, total_mass
	
	def tick(self, dt=GameSettings.tick):
		super().tick(dt)
		if not utils.is_on_screen(self.position) and physics.current_barycenter is not None and not self.ev_checked:
			r = utils.vec_len(utils.vec_minus(self.position, physics.current_barycenter.position))
			gm = GameSettings.g_constant * physics.current_barycenter.mass
			escape_velocity = math.sqrt(2 * gm / r)
			vel = utils.vec_len(self.velocity) * GameSettings.tick / 1000
			self.ev_checked = True
			if vel >= escape_velocity:
				# Buh-bye
				print("Asteroid at {} reached escape velocity ({} > {}) - r = {}, gm = {}".format(self.position, vel, escape_velocity, r, gm))
				self.remove()
			else:
				print("Asteroid at {} will return ({} < {})".format(self.position, vel, escape_velocity))
		elif utils.is_on_screen(self.position) and self.ev_checked:
			self.ev_checked = False
