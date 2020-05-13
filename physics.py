from typing import Tuple, Iterable, Union, List
from debugger import Debugger
import math
import graphics
from threading import Lock

import utils
from game_settings import GameSettings


def normalize_vector(vector: Tuple[float, float]) -> Tuple[float, float]:
	length = math.hypot(vector[0], vector[1])
	return vector[0]/length, vector[1]/length


def tick_physics():
	Massive.massive_objects_lock.acquire()
	for massive in Massive.massive_objects:
		Kinematic.kinematic_objects_lock.acquire()
		for kinematic in Kinematic.kinematic_objects:
			if not kinematic == massive:  # Moons don't exert gravity on themselves
				kinematic.apply_acceleration(massive.gravity_force(kinematic.position))
		Kinematic.kinematic_objects_lock.release()
	Massive.massive_objects_lock.release()


class OnscreenObject:
	objects_lock = Lock()
	objects = set()
	
	def __init__(self, position: Tuple[float, float], direction: Union[float, Tuple[float, float]], shape: graphics.Shape = None):
		self.position = position
		self.collider_radius = 0
		self.collision_enabled = False
		self.direction = direction if isinstance(direction, tuple) else utils.rotate_point((1, 0), direction)
		self.shape = shape
		OnscreenObject.objects_lock.acquire()
		OnscreenObject.objects.add(self)
		OnscreenObject.objects_lock.release()
		self.attachments_below: List[OnscreenObject] = list()
		self.attachments_above: List[OnscreenObject] = list()
		
	def remove(self):
		OnscreenObject.objects_lock.acquire()
		if self in OnscreenObject.objects:
			OnscreenObject.objects.remove(self)
		OnscreenObject.objects_lock.release()
	
	def collided(self, other):
		pass
	
	def attach_object(self, other_object, below: bool = False):
		obj = OnscreenObject((0, 0), 0, other_object) if isinstance(other_object, graphics.Shape) else other_object
		if below:
			self.attachments_below.append(obj)
		else:
			self.attachments_above.append(obj)
			

class Massive(OnscreenObject):
	
	massive_objects_lock = Lock()
	massive_objects = set()
	
	def __init__(self, mass: float, position: Tuple[float, float], direction: Union[float, Tuple[float, float]], shape: graphics.Shape = None, virtual: bool = False):
		OnscreenObject.__init__(self, position, direction, shape)
		self.mass = mass
		self.virtual = virtual
		if not virtual:
			
			Massive.massive_objects_lock.acquire()
			Massive.massive_objects.add(self)
			Massive.massive_objects_lock.release()
			self.collision_enabled = True
			
	def remove(self):
		Massive.massive_objects_lock.acquire()
		if not self.virtual and self in Massive.massive_objects:
			Massive.massive_objects.remove(self)
		Massive.massive_objects_lock.release()
		super().remove()
	
	def gravity_force(self, other_position: Tuple[float, float]) -> Tuple[float, float]:
		r_vector = utils.vec_minus(self.position, other_position)  # self.position[0] - other_position[0], self.position[1] - other_position[1]
		distance = utils.vec_len(r_vector)
		force = (GameSettings.tick / 1000.0) * GameSettings.g_constant * self.mass / (distance ** 2)
		force_vector = utils.vec_mul(r_vector, force)
		return force_vector


def barycenter(planets: Iterable[Massive]) -> Massive:
	sum_mass = 0
	sum_xy = (0, 0)
	for planet in planets:
		sum_mass += planet.mass
		sum_xy = sum_xy[0] + planet.position[0] * planet.mass, sum_xy[1] + planet.position[1] * planet.mass
	barycenter_pos = (sum_xy[0]/sum_mass, sum_xy[1]/sum_mass)
	return Massive(sum_mass, barycenter_pos, 0, virtual=True)


class Kinematic(OnscreenObject):
	kinematic_objects = set()
	kinematic_objects_lock = Lock()
	
	def __init__(
			self,
			position: Tuple[float, float],
			initial_velocity: Tuple[float, float] = (0, 0),
			initial_direction: Tuple[float, float] = (0, 1),
			initial_rotation: float = 0.0,
			mass: float = 1.0,
			inertial_moment_multiplier: float = 1.0,
			shape: graphics.Shape = None,
			reactive: bool = True
	):
		"""
		Initializes an object with 2D physics enabled
		:param position: The position of the object
		:param initial_velocity: The initial velocity of the object
		:param initial_direction: The initial heading of the object
		:param initial_rotation: The initial angular velocity of the object
		:param mass: The mass of the object
		:param inertial_moment_multiplier: The inertial moment multiplier factor of the object
		:param shape: The shape attached to this object
		:param reactive: Whether the object reacts to collisions
		"""
		super().__init__(position, direction=initial_direction, shape=shape)
		self.velocity = initial_velocity
		self.rotation = initial_rotation
		self.mass = mass
		self.moment_multiplier = inertial_moment_multiplier
		
		Kinematic.kinematic_objects_lock.acquire()
		Kinematic.kinematic_objects.add(self)
		Kinematic.kinematic_objects_lock.release()
		
		self.collision_enabled = True
		self.collided_this_tick = set()
		self.reactive = reactive
		self.lock_target: Union[OnscreenObject, None] = None
	
	def apply_acceleration(self, acceleration: Tuple[float, float]):
		"""
		Applies a direct acceleration on the object in the center of mass
		:param acceleration: The acceleration
		:return: None
		"""
		Debugger.add_vector(self.position, utils.vec_mul(acceleration, 1000/GameSettings.tick), (255, 128, 0))
		self.velocity = utils.vec_plus(self.velocity, acceleration)
	
	def apply_torque(self, force_scalar: float, leverage: float):
		"""
		Applies a direct torque on the object
		:param force_scalar: The scalar size of the force that is exerting the torque
		:param leverage: The length of the lever
		:return: None
		"""
		inertial_moment = self.mass * self.moment_multiplier
		torque = force_scalar * leverage
		d_omega = torque / inertial_moment
		self.rotation += d_omega
		Debugger.add_vector((self.position[0], self.position[1]+leverage), (force_scalar, 0), (0, 255, 255))
	
	def apply_force(self, force_vector: Tuple[float, float], action_point: Tuple[float, float], absolute: bool = False):
		"""
		Applies a force on the object
		:param force_vector: The force to apply
		:param action_point: The offset of the force from the center of mass
		:param absolute: Whether the action point is relative to the object or absolute to the screen
		:return: None
		"""
		ap = action_point if not absolute else utils.vec_minus(self.position, action_point)
		accel_comp_dir = utils.vec_mul(ap, -1)
		f_acc, f_tan = utils.vec_decompose(force_vector, accel_comp_dir)
		acceleration = utils.vec_mul(f_acc, 1 / self.mass)
		tan_force = utils.vec_len(f_tan)
		tan_direction = 1 if ((f_tan[1] < 0 and f_acc[1] < 0) or (f_tan[1] >= 0 and f_acc[1] >= 0)) and (
				(f_tan[0] < 0 and f_acc[0] < 0) or (f_tan[0] >= 0 and f_acc[0] >= 0)) else -1
		leverage = utils.vec_len(accel_comp_dir)
		Debugger.add_vector(self.position, utils.vec_mul(force_vector, 1000/GameSettings.tick), (255, 0, 0))
		self.apply_acceleration(acceleration)
		self.apply_torque(tan_force * tan_direction, leverage)
	
	def tick(self, dt=GameSettings.tick):
		self.position = self.position[0] + (dt / 1000) * self.velocity[0], self.position[1] + (dt / 1000) * self.velocity[1]
		if self.lock_target is None:
			self.direction = utils.mul_matrix(self.direction, utils.rotation_matrix(self.rotation * (dt / 1000)))
		else:
			self.direction = utils.vec_minus(self.lock_target.position, self.position)
		self.collided_this_tick.clear()
		Debugger.add_vector(self.position, self.velocity, (0, 255, 0), dt)
		Debugger.add_vector(self.position, utils.vec_mul(self.direction, 2*self.collider_radius), (0, 0, 128), dt)
		
	def tidal_lock(self, other: OnscreenObject):
		self.lock_target = other
		
	def remove(self):
		Kinematic.kinematic_objects_lock.acquire()
		if self in Kinematic.kinematic_objects:
			Kinematic.kinematic_objects.remove(self)
		Kinematic.kinematic_objects_lock.release()
		super().remove()
		
	@staticmethod
	def tick_all(dt=GameSettings.tick):
		Kinematic.kinematic_objects_lock.acquire()
		current_objects = list(Kinematic.kinematic_objects)
		Kinematic.kinematic_objects_lock.release()
		for kinematic_object in current_objects:
			kinematic_object.tick(dt)


def stable_orbit(satellite: Kinematic, primary: Massive, eccentricity: float = 0.0, at_periapsis: bool = False):
	"""
	Puts the given object onto a stable orbit around a primary at its current coordinates with the minimal dv
	:param at_periapsis: Whether the current point is a periapsis. If not, it's treated as an apoapsis.
	:param eccentricity: The orbit's eccentricity, [0, 1). 0 = circular.
	:param primary: The primary object, Massive
	:param satellite: The satellite, Kinematic
	:return: None
	"""
	new_velocity = stable_orbit_vector(at_periapsis, eccentricity, primary, satellite)
	satellite.velocity = new_velocity


def stable_orbit_vector(at_periapsis, eccentricity, primary, satellite):
	"""
	Gets the velocity vector required for the given object to be at a stable orbit around a primary at its current coordinates with the minimal dv
	:param at_periapsis: Whether the current point is a periapsis. If not, it's treated as an apoapsis.
	:param eccentricity: The orbit's eccentricity, [0, 1). 0 = circular.
	:param primary: The primary object, Massive
	:param satellite: The satellite, Kinematic
	:return: The vector that should put this object on the designated stable orbit
	"""
	print("Setting stable orbit around {}/{} with eccentricity {}, at periapsis: {}".format(primary.position, primary.mass, eccentricity, at_periapsis))
	acp = primary.gravity_force(satellite.position)
	acp_strength = math.hypot(acp[0], acp[1]) * (1000 / GameSettings.tick)
	distance = math.hypot(primary.position[0] - satellite.position[0], primary.position[1] - satellite.position[1])
	orbital_speed = math.sqrt(acp_strength * distance)
	if 0.01 < eccentricity < 1:
		ratio2 = 1 - eccentricity ** 2
		ratio = math.sqrt(ratio2)
		if at_periapsis:
			orbital_speed = orbital_speed / ratio
		else:
			orbital_speed = orbital_speed * ratio
		print("Eccentric orbit - ratio = {}, speed = {}".format(ratio, orbital_speed))
	else:
		print("Circular orbit - speed = {}".format(orbital_speed))
	current_velocity = satellite.velocity
	if current_velocity[0] == current_velocity[1]:
		current_velocity = (current_velocity[0], current_velocity[1] - 1)
	print("Current velocity: {}".format(current_velocity))
	acp_unit = utils.vec_unit(acp)
	print("Acp unit vector: {}".format(acp_unit))
	comp_rad, comp_tan = utils.vec_decompose(current_velocity, acp_unit)
	print("Current velocity components: rad={}, tan={}".format(comp_rad, comp_tan))
	new_dir = utils.vec_angle(comp_tan)
	print("Direction: {}".format(new_dir))
	new_velocity = utils.mul_matrix((orbital_speed, 0), utils.rotation_matrix(new_dir))  # math.cos(new_dir) * orbital_speed, math.sin(new_dir) * orbital_speed
	print("New initial velocity: {}".format(new_velocity))
	if isinstance(primary, Kinematic):
		new_velocity = utils.vec_plus(new_velocity, primary.velocity)
	return new_velocity


def bounce(object1: Kinematic, object2: Union[Massive, Kinematic]) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float]], Tuple[Tuple[float, float], Tuple[float, float]]]:
	"""
	Bounces two objects off each other
	:param object1: The first object. Must be a Kinematic.
	:param object2: The second object. Can be Kinematic or stationary Massive
	:return: A tuple containing the velocity changes (x, y) and torques (force, lever) of both objects
	"""
	
	sum_mass = (object1.mass if isinstance(object2, Kinematic) and object2.reactive else object2.mass) + object2.mass
	time_scale = 1  # GameSettings.tick / 1000
	
	if object1.reactive:
		print("Object 1: velocity before collision: {}".format(object1.velocity))
		rel_1 = utils.vec_minus(object1.position, object2.position)
		rvel_1 = utils.vec_minus(object1.velocity, object2.velocity)
		dir_scale_1 = utils.dot_product(rvel_1, rel_1) / utils.vec_len(rel_1) ** 2
		pre_mass_1 = utils.vec_mul(rel_1, dir_scale_1)
		dv_1 = utils.vec_mul(pre_mass_1, -2 * time_scale * object2.mass / sum_mass)
		object1.apply_acceleration(dv_1)
		print("Object 1: dv: {}".format(dv_1))
		print("Object 1: velocity after collision: {}".format(object1.velocity))
		Debugger.add_vector(object1.position, dv_1, (255, 0, 255), 1000)
		
		lever1 = utils.vec_mul(utils.vec_unit(utils.vec_minus(object1.position, object2.position)), object1.collider_radius)
		torque1 = utils.dot_product(dv_1, lever1)
		object1.apply_torque(torque1, utils.vec_len(lever1))
	else:
		dv_1 = (0, 0)
		lever1 = 0
		torque1 = 0
		
	if isinstance(object2, Kinematic) and object2.reactive:
		print("Object 2: velocity before collision: {}".format(object2.velocity))
		rel_2 = utils.vec_minus(object2.position, object1.position)
		rvel_2 = utils.vec_minus(object2.velocity, object1.velocity)
		dir_scale_2 = utils.dot_product(rvel_2, rel_2) / utils.vec_len(rel_2)**2
		pre_mass_2 = utils.vec_mul(rel_2, dir_scale_2)
		dv_2 = utils.vec_mul(pre_mass_2, -2 * time_scale * object1.mass / sum_mass)
		object2.apply_acceleration(dv_2)
		print("Object 2: dv: {}".format(dv_2))
		print("Object 2: velocity after collision: {}".format(object2.velocity))
		Debugger.add_vector(object2.position, dv_2, (255, 0, 255), 500)
		lever2 = utils.vec_mul(utils.vec_unit(utils.vec_minus(object2.position, object1.position)), object2.collider_radius)
		torque2 = utils.dot_product(dv_2, lever2)
		object2.apply_torque(torque2, utils.vec_len(lever2))
	else:
		dv_2 = (0, 0)
		lever2 = 0
		torque2 = 0
		
	dv_1_mag = utils.vec_len(dv_1)
	dv_2_mag = utils.vec_len(dv_2)
	
	if object1.reactive:
		jerk_1 = dv_1 if dv_1_mag >= 1000/GameSettings.tick else utils.vec_mul(utils.vec_unit(dv_1), 1000/GameSettings.tick)
	else:
		jerk_1 = (0, 0)
	if object2.reactive:
		jerk_2 = dv_2 if dv_2_mag >= 1000/GameSettings.tick else utils.vec_mul(utils.vec_unit(dv_2), 1000/GameSettings.tick)
	else:
		jerk_2 = (0, 0)
		
	# Make sure that the dvs are not the same.
	if utils.vec_len(utils.vec_minus(jerk_1, jerk_2)) < 1000/GameSettings.tick:
		jerk_1 = jerk_1[0], jerk_1[1]-1000/GameSettings.tick
		jerk_2 = jerk_2[0], jerk_2[1]+1000/GameSettings.tick
	
	while utils.vec_len(utils.vec_minus(object1.position, object2.position)) < object1.collider_radius + object2.collider_radius:
		if object1.reactive:
			object1.position = utils.vec_plus(object1.position, utils.vec_mul(jerk_1, GameSettings.tick / 1000))
		if object2.reactive:
			object2.position = utils.vec_plus(object2.position, utils.vec_mul(jerk_2, GameSettings.tick / 1000))
	
	return (dv_1, (lever1, torque1)), (dv_2, (lever2, torque2))


current_barycenter: Massive = None


def center_planets(planets: Iterable[Massive], screen_size: Tuple[int, int] = GameSettings.screen_size):
	"""
	Keeps the barycenter of the given Massive objects in the center of the screen
	:param planets: The Massive objects (planets, moons, etc...)
	:param screen_size: The center of the screen
	:return: None
	"""
	global current_barycenter
	current_barycenter = barycenter(planets)
	center = current_barycenter.position  # Update the barycenter to make it accessible for other calculations outside
	screen_center = utils.vec_mul(screen_size, 0.5)
	delta = utils.vec_minus(screen_center, center)
	for planet in planets:
		planet.position = utils.vec_plus(planet.position, delta)
		
	
def tidal_lock(primary: Massive, satellite: Kinematic):
	"""
	Gives the satellite a rotation that will make it always show the same face to the primary. Assumes the satellite is on a stable orbit.
	Currently very cheaty.
	:param primary: A Massive object acting as a primary
	:param satellite: A Kinematic object acting as a satellite
	:return: None
	"""
	satellite.tidal_lock(primary)