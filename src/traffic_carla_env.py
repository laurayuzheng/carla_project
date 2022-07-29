import collections
import queue
import time

import numpy as np
import carla

from PIL import Image, ImageDraw

from sumo_integration.bridge_helper import BridgeHelper  # pylint: disable=wrong-import-position
from sumo_integration.carla_simulation import CarlaSimulation  # pylint: disable=wrong-import-position
from sumo_integration.constants import INVALID_ACTOR_ID  # pylint: disable=wrong-import-position
from sumo_integration.sumo_simulation import SumoSimulation  # pylint: disable=wrong-import-position


PRESET_WEATHERS = {
    1: carla.WeatherParameters.ClearNoon,
    2: carla.WeatherParameters.CloudyNoon,
    3: carla.WeatherParameters.WetNoon,
    5: carla.WeatherParameters.MidRainyNoon,
    # 4: carla.WeatherParameters.WetCloudyNoon,
    # 6: carla.WeatherParameters.HardRainNoon,
    # 7: carla.WeatherParameters.SoftRainNoon,

    8: carla.WeatherParameters.ClearSunset,
    9: carla.WeatherParameters.CloudySunset,
    10: carla.WeatherParameters.WetSunset,
    12: carla.WeatherParameters.MidRainSunset,
    # 11: carla.WeatherParameters.WetCloudySunset,
    # 13: carla.WeatherParameters.HardRainSunset,
    # 14: carla.WeatherParameters.SoftRainSunset,
}

WEATHERS = list(PRESET_WEATHERS.values())
VEHICLE_NAME = 'vehicle.lincoln.mkz2017'
COLLISION_THRESHOLD = 20000


def set_sync_mode(client, sync):
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 1.0 / 10.0

    world.apply_settings(settings)


class Camera(object):
    def __init__(self, world, player, w, h, fov, x, y, z, pitch, yaw, type='rgb'):
        bp = world.get_blueprint_library().find('sensor.camera.%s' % type)
        bp.set_attribute('image_size_x', str(w))
        bp.set_attribute('image_size_y', str(h))
        bp.set_attribute('fov', str(fov))

        loc = carla.Location(x=x, y=y, z=z)
        rot = carla.Rotation(pitch=pitch, yaw=yaw)
        transform = carla.Transform(loc, rot)

        self.type = type
        self.queue = queue.Queue()

        self.camera = world.spawn_actor(bp, transform, attach_to=player)
        self.camera.listen(self.queue.put)

    def get(self):
        image = None

        while image is None or self.queue.qsize() > 0:
            image = self.queue.get()

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.type == 'semantic_segmentation':
            return array[:, :, 0]

        return array

    def __del__(self):
        self.camera.destroy()

        with self.queue.mutex:
            self.queue.queue.clear()


def get_nearby_lights(vehicle, lights, pixels_per_meter=5.5, size=512, radius=5):
    result = list()

    transform = vehicle.get_transform()
    pos = transform.location
    theta = np.radians(90 + transform.rotation.yaw)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
        ])

    for light in lights:
        delta = light.get_transform().location - pos

        target = R.T.dot([delta.x, delta.y])
        target *= pixels_per_meter
        target += size // 2

        if min(target) < 0 or max(target) >= size:
            continue

        trigger = light.trigger_volume
        light.get_transform().transform(trigger.location)
        dist = trigger.location.distance(vehicle.get_location())
        a = np.sqrt(
                trigger.extent.x ** 2 +
                trigger.extent.y ** 2 +
                trigger.extent.z ** 2)
        b = np.sqrt(
                vehicle.bounding_box.extent.x ** 2 +
                vehicle.bounding_box.extent.y ** 2 +
                vehicle.bounding_box.extent.z ** 2)

        if dist > a + b:
            continue

        result.append(light)

    return result


def draw_traffic_lights(image, vehicle, lights, pixels_per_meter=5.5, size=512, radius=5):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    transform = vehicle.get_transform()
    pos = transform.location
    theta = np.radians(90 + transform.rotation.yaw)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
        ])

    for light in lights:
        delta = light.get_transform().location - pos

        target = R.T.dot([delta.x, delta.y])
        target *= pixels_per_meter
        target += size // 2

        if min(target) < 0 or max(target) >= size:
            continue

        trigger = light.trigger_volume
        light.get_transform().transform(trigger.location)
        dist = trigger.location.distance(vehicle.get_location())
        a = np.sqrt(
                trigger.extent.x ** 2 +
                trigger.extent.y ** 2 +
                trigger.extent.z ** 2)
        b = np.sqrt(
                vehicle.bounding_box.extent.x ** 2 +
                vehicle.bounding_box.extent.y ** 2 +
                vehicle.bounding_box.extent.z ** 2)

        if dist > a + b:
            continue

        x, y = target
        draw.ellipse(
                (x-radius, y-radius, x+radius, y+radius),
                23 + light.state.real)

    return np.array(image)


class MapCamera(Camera):
    def __init__(self, world, player, size, fov, z, pixels_per_meter, radius):
        super().__init__(
                world, player,
                size, size, fov,
                0, 0, z, -90, 0,
                'semantic_segmentation')

        self.world = world
        self.player = player
        self.pixels_per_meter = pixels_per_meter
        self.size = size
        self.radius = radius

    def get(self):
        image = Image.fromarray(super().get())
        draw = ImageDraw.Draw(image)

        transform = self.player.get_transform()
        pos = transform.location
        theta = np.radians(90 + transform.rotation.yaw)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        for light in self.world.get_actors().filter('*traffic_light*'):
            delta = light.get_transform().location - pos

            target = R.T.dot([delta.x, delta.y])
            target *= self.pixels_per_meter
            target += self.size // 2

            if min(target) < 0 or max(target) >= self.size:
                continue

            x, y = target
            draw.ellipse(
                    (x-self.radius, y-self.radius, x+self.radius, y+self.radius),
                    13 + light.state.real)

        return np.array(image)


class VehiclePool(object):
    def __init__(self, client, n_vehicles):
        self.client = client
        self.world = client.get_world()

        veh_bp = self.world.get_blueprint_library().filter('vehicle.*')
        spawn_points = np.random.choice(self.world.get_map().get_spawn_points(), n_vehicles)
        batch = list()

        for i, transform in enumerate(spawn_points):
            bp = np.random.choice(veh_bp)
            bp.set_attribute('role_name', 'autopilot')

            batch.append(
                    carla.command.SpawnActor(bp, transform).then(
                        carla.command.SetAutopilot(carla.command.FutureActor, True)))

        self.vehicles = list()
        errors = set()

        for msg in self.client.apply_batch_sync(batch):
            if msg.error:
                errors.add(msg.error)
            else:
                self.vehicles.append(msg.actor_id)

        if errors:
            print('\n'.join(errors))

        print('%d / %d vehicles spawned.' % (len(self.vehicles), n_vehicles))

    def __del__(self):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles])


class PedestrianPool(object):
    def __init__(self, client, n_pedestrians):
        self.client = client
        self.world = client.get_world()

        ped_bp = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        con_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        spawn_points = [self._get_spawn_point() for _ in range(n_pedestrians)]
        batch = [carla.command.SpawnActor(np.random.choice(ped_bp), spawn) for spawn in spawn_points]
        walkers = list()
        errors = set()

        for msg in client.apply_batch_sync(batch, True):
            if msg.error:
                errors.add(msg.error)
            else:
                walkers.append(msg.actor_id)

        if errors:
            print('\n'.join(errors))

        batch = [carla.command.SpawnActor(con_bp, carla.Transform(), walker_id) for walker_id in walkers]
        controllers = list()
        errors = set()

        for msg in client.apply_batch_sync(batch, True):
            if msg.error:
                errors.add(msg.error)
            else:
                controllers.append(msg.actor_id)

        if errors:
            print('\n'.join(errors))

        self.walkers = self.world.get_actors(walkers)
        self.controllers = self.world.get_actors(controllers)

        for controller in self.controllers:
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(1.4 + np.random.randn())

        self.timers = [np.random.randint(60, 600) * 20 for _ in self.controllers]

        print('%d / %d pedestrians spawned.' % (len(self.walkers), n_pedestrians))

    def _get_spawn_point(self, n_retry=10):
        for _ in range(n_retry):
            spawn = carla.Transform()
            spawn.location = self.world.get_random_location_from_navigation()

            if spawn.location is not None:
                return spawn

        raise ValueError('No valid spawns.')

    def tick(self):
        for i, controller in enumerate(self.controllers):
            self.timers[i] -= 1

            if self.timers[i] <= 0:
                self.timers[i] = np.random.randint(60, 600) * 20
                controller.go_to_location(self.world.get_random_location_from_navigation())

    def __del__(self):
        for controller in self.controllers:
            controller.stop()

        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walkers])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.controllers])


class TrafficCarlaEnv(object):
    def __init__(self, args, 
                    town='Town01', 
                    tls_manager='none',
                    **kwargs):

        self.sumo = SumoSimulation(args.sumo_cfg_file, args.step_length, args.sumo_host,
                                     args.sumo_port, args.sumo_gui, args.client_order)
        self.carla = CarlaSimulation(args.carla_host, args.carla_port, args.step_length)
        self._client = self.carla.client

        set_sync_mode(self._client, False)

        self._town_name = town
        self._world = self._client.load_world(town)
        self._map = self._world.get_map()

        self._blueprints = self._world.get_blueprint_library()

        self._tick = 0
        self._player = None
        self._player_sumo_id = None # Need this to get state

        # vehicle, sensor
        self._actor_dict = collections.defaultdict(list)
        self._cameras = dict()

        self.tls_manager = tls_manager 
        if tls_manager == 'carla':
            self.sumo.switch_off_traffic_lights()
        elif tls_manager == 'sumo':
            self.carla.switch_off_traffic_lights()

        # Mapped actor ids.
        self.sumo2carla_ids = {}  # Contains only actors controlled by sumo.
        self.carla2sumo_ids = {}  # Contains only actors controlled by carla.

        BridgeHelper.blueprint_library = self.carla.world.get_blueprint_library()
        BridgeHelper.offset = self.sumo.get_net_offset()

        # Configuring carla simulation in sync mode.
        settings = self.carla.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.carla.step_length
        self.carla.world.apply_settings(settings)


    def _set_weather(self, weather_string):
        if weather_string == 'random':
            weather = np.random.choice(WEATHERS)
        else:
            weather = weather_string

        self.weather = weather
        self._world.set_weather(weather)

    def reset(self, weather='random', n_vehicles=10, n_pedestrians=10, seed=0):
        is_ready = False

        while not is_ready:
            np.random.seed(seed)

            self._clean_up()
            self._spawn_player(np.random.choice(self._map.get_spawn_points()))
            self._setup_sensors()

            self._set_weather(weather)
            # self._pedestrian_pool = PedestrianPool(self._client, n_pedestrians)
            # self._vehicle_pool = VehiclePool(self._client, n_pedestrians)

            is_ready = self.ready()

    def _spawn_player(self, start_pose):
        '''TODO: Send hero info to SUMO, then record the sumo id of the hero
            self._player_sumo_id 
        '''
        vehicle_bp = np.random.choice(self._blueprints.filter(VEHICLE_NAME))
        vehicle_bp.set_attribute('role_name', 'hero')

        self._player = self._world.spawn_actor(vehicle_bp, start_pose)

        self._actor_dict['player'].append(self._player)

    def ready(self, ticks=10):
        for _ in range(ticks):
            self.step()

        for x in self._actor_dict['camera']:
            x.get()

        self._time_start = time.time()
        self._tick = 0

        return True

    def get_state(self):
        '''Get the state of the driver's lane'''
        return self.sumo.get_state(self._player_sumo_id)

    def step(self, control=None):
        if control is not None:
            self._player.apply_control(control)

        # self._world.tick()

        # -----------------
        # sumo-->carla sync
        # -----------------
        self.sumo.tick()

        # Spawning new sumo actors in carla (i.e, not controlled by carla).
        sumo_spawned_actors = self.sumo.spawned_actors - set(self.carla2sumo_ids.values())
        for sumo_actor_id in sumo_spawned_actors:
            self.sumo.subscribe(sumo_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)

            carla_blueprint = BridgeHelper.get_carla_blueprint(sumo_actor, self.sync_vehicle_color)
            if carla_blueprint is not None:
                carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform,
                                                                   sumo_actor.extent)

                carla_actor_id = self.carla.spawn_actor(carla_blueprint, carla_transform)
                if carla_actor_id != INVALID_ACTOR_ID:
                    self.sumo2carla_ids[sumo_actor_id] = carla_actor_id
            else:
                self.sumo.unsubscribe(sumo_actor_id)

        # Destroying sumo arrived actors in carla.
        for sumo_actor_id in self.sumo.destroyed_actors:
            if sumo_actor_id in self.sumo2carla_ids:
                self.carla.destroy_actor(self.sumo2carla_ids.pop(sumo_actor_id))

        # Updating sumo actors in carla.
        for sumo_actor_id in self.sumo2carla_ids:
            carla_actor_id = self.sumo2carla_ids[sumo_actor_id]

            sumo_actor = self.sumo.get_actor(sumo_actor_id)
            carla_actor = self.carla.get_actor(carla_actor_id)

            carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform,
                                                               sumo_actor.extent)
            if self.sync_vehicle_lights:
                carla_lights = BridgeHelper.get_carla_lights_state(carla_actor.get_light_state(),
                                                                   sumo_actor.signals)
            else:
                carla_lights = None

            self.carla.synchronize_vehicle(carla_actor_id, carla_transform, carla_lights)

        # Updates traffic lights in carla based on sumo information.
        if self.tls_manager == 'sumo':
            common_landmarks = self.sumo.traffic_light_ids & self.carla.traffic_light_ids
            for landmark_id in common_landmarks:
                sumo_tl_state = self.sumo.get_traffic_light_state(landmark_id)
                carla_tl_state = BridgeHelper.get_carla_traffic_light_state(sumo_tl_state)

                self.carla.synchronize_traffic_light(landmark_id, carla_tl_state)

        # -----------------
        # carla-->sumo sync
        # -----------------
        self.carla.tick()

        # Spawning new carla actors (not controlled by sumo)
        carla_spawned_actors = self.carla.spawned_actors - set(self.sumo2carla_ids.values())
        for carla_actor_id in carla_spawned_actors:
            carla_actor = self.carla.get_actor(carla_actor_id)

            type_id = BridgeHelper.get_sumo_vtype(carla_actor)
            color = carla_actor.attributes.get('color', None) if self.sync_vehicle_color else None
            if type_id is not None:
                sumo_actor_id = self.sumo.spawn_actor(type_id, color)
                if sumo_actor_id != INVALID_ACTOR_ID:
                    self.carla2sumo_ids[carla_actor_id] = sumo_actor_id
                    self.sumo.subscribe(sumo_actor_id)

        # Destroying required carla actors in sumo.
        for carla_actor_id in self.carla.destroyed_actors:
            if carla_actor_id in self.carla2sumo_ids:
                self.sumo.destroy_actor(self.carla2sumo_ids.pop(carla_actor_id))

        # Updating carla actors in sumo.
        for carla_actor_id in self.carla2sumo_ids:
            sumo_actor_id = self.carla2sumo_ids[carla_actor_id]

            carla_actor = self.carla.get_actor(carla_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)

            sumo_transform = BridgeHelper.get_sumo_transform(carla_actor.get_transform(),
                                                             carla_actor.bounding_box.extent)
            if self.sync_vehicle_lights:
                carla_lights = self.carla.get_actor_light_state(carla_actor_id)
                if carla_lights is not None:
                    sumo_lights = BridgeHelper.get_sumo_lights_state(sumo_actor.signals,
                                                                     carla_lights)
                else:
                    sumo_lights = None
            else:
                sumo_lights = None

            self.sumo.synchronize_vehicle(sumo_actor_id, sumo_transform, sumo_lights)

        # Updates traffic lights in sumo based on carla information.
        if self.tls_manager == 'carla':
            common_landmarks = self.sumo.traffic_light_ids & self.carla.traffic_light_ids
            for landmark_id in common_landmarks:
                carla_tl_state = self.carla.get_traffic_light_state(landmark_id)
                sumo_tl_state = BridgeHelper.get_sumo_traffic_light_state(carla_tl_state)

                # Updates all the sumo links related to this landmark.
                self.sumo.synchronize_traffic_light(landmark_id, sumo_tl_state)

        self._tick += 1
        # self._pedestrian_pool.tick()

        transform = self._player.get_transform()
        velocity = self._player.get_velocity()

        # Put here for speed (get() busy polls queue).
        result = {key: val.get() for key, val in self._cameras.items()}
        result.update({
            'wall': time.time() - self._time_start,
            'tick': self._tick,
            'x': transform.location.x,
            'y': transform.location.y,
            'theta': transform.rotation.yaw,
            'speed': np.linalg.norm([velocity.x, velocity.y, velocity.z]),
            })

        return result

    def _setup_sensors(self):
        """
        Add sensors to _actor_dict to be cleaned up.
        """
        self._cameras['rgb'] = Camera(self._world, self._player, 256, 144, 90, 1.2, 0.0, 1.3, 0.0, 0.0)
        self._cameras['rgb_left'] = Camera(self._world, self._player, 256, 144, 90, 1.2, -0.25, 1.3, 0.0, -45.0)
        self._cameras['rgb_right'] = Camera(self._world, self._player, 256, 144, 90, 1.2, 0.25, 1.3, 0.0, 45.0)

        # self._cameras['topdown'] = MapCamera(self._world, self._player, 512, 5, 500.0, 11.75, 8)
        # self._cameras['topdown'] = MapCamera(self._world, self._player, 512, 25, 100.0, 11.75, 8)

        # self._cameras['topdown'] = MapCamera(self._world, self._player, 512, 7.5, 500.0, 8.0, 6)
        # self._cameras['topdown'] = MapCamera(self._world, self._player, 512, 5 * 7.5, 100.0, 8.0, 6)

        # self._cameras['topdown'] = MapCamera(self._world, self._player, 512, 10.0, 500.0, 6.0, 5)
        self._cameras['topdown'] = MapCamera(self._world, self._player, 512, 5 * 10.0, 100.0, 5.5, 5)

    def __enter__(self):
        set_sync_mode(self._client, True)

        return self

    def __exit__(self, *args):
        """
        Make sure to set the world back to async,
        otherwise future clients might have trouble connecting.
        """
        self._clean_up()

        set_sync_mode(self._client, False)

    def _clean_up(self):
        # self._pedestrian_pool = None
        # self._vehicle_pool = None
        self._cameras.clear()

        for actor_type in list(self._actor_dict.keys()):
            self._client.apply_batch([carla.command.DestroyActor(x) for x in self._actor_dict[actor_type]])
            self._actor_dict[actor_type].clear()

        self._actor_dict.clear()

        self._tick = 0
        self._time_start = time.time()

        self._player = None

        ## CARLA-SUMO synchronization clean up 
        settings = self.carla.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.carla.world.apply_settings(settings)

        # Destroying synchronized actors.
        for carla_actor_id in self.sumo2carla_ids.values():
            self.carla.destroy_actor(carla_actor_id)

        for sumo_actor_id in self.carla2sumo_ids.values():
            self.sumo.destroy_actor(sumo_actor_id)

        # Closing sumo and carla client.
        self.carla.close()
        self.sumo.close()
