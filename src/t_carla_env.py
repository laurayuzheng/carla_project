import collections
from multiprocessing import synchronize
import os 
import queue
import time

import numpy as np
import carla
import tempfile
import lxml.etree as ET  # pylint: disable=wrong-import-position
from sumo_integration.util.netconvert_carla import netconvert_carla
import re
import json 
import sumolib
import logging 
import traci 
import random 

from PIL import Image, ImageDraw
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent

from sumo_integration.bridge_helper import BridgeHelper  # pylint: disable=wrong-import-position
from sumo_integration.carla_simulation import CarlaSimulation  # pylint: disable=wrong-import-position
from sumo_integration.constants import INVALID_ACTOR_ID  # pylint: disable=wrong-import-position
from sumo_integration.sumo_simulation import SumoSimulation  # pylint: disable=wrong-import-position

# from leaderboard.team_code.auto_pilot import AutoPilot

from .run_synchronization import SimulationSynchronization

PROJECT_ROOT = "/scratch/2020_CARLA_challenge"
CARLA_HOME = "/home/laura/DrivingSimulators/CARLA_0.9.10"

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

PRESET_WEATHERS_STRING = {
    1: "ClearNoon",
    2: "CloudyNoon",
    3: "WetNoon",
    5: "MidRainyNoon",
    # 4: WetCloudyNoon,
    # 6: HardRainNoon,
    # 7: SoftRainNoon,

    8: "ClearSunset",
    9: "CloudySunset",
    10: "WetSunset",
    12: "MidRainSunset",
    # 11: WetCloudySunset,
    # 13: HardRainSunset,
    # 14: SoftRainSunset,
}

WEATHERS = list(PRESET_WEATHERS.values())
VEHICLE_NAME = 'vehicle.lincoln.mkz2017'
COLLISION_THRESHOLD = 20000

def write_sumocfg_xml(cfg_file, net_file, vtypes_file, viewsettings_file, additional_traci_clients=0):
    """
    Writes sumo configuration xml file.
    """
    root = ET.Element('configuration')

    input_tag = ET.SubElement(root, 'input')
    ET.SubElement(input_tag, 'net-file', {'value': net_file})
    ET.SubElement(input_tag, 'route-files', {'value': vtypes_file})

    gui_tag = ET.SubElement(root, 'gui_only')
    ET.SubElement(gui_tag, 'gui-settings-file', {'value': viewsettings_file})

    ET.SubElement(root, 'num-clients', {'value': str(additional_traci_clients+1)})

    tree = ET.ElementTree(root)

    # with open(cfg_file, 'w+') as f:
    tree.write(cfg_file, pretty_print=True, encoding='UTF-8', xml_declaration=True)


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
    def __init__(self, client, n_vehicles, tm_port):
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

        # tm_port = self.traffic_manager.get_port()
        vehicles_list = self.world.get_actors().filter('vehicle.*')
        for v in vehicles_list:
            v.set_autopilot(True,tm_port)

        print('%d / %d vehicles spawned.' % (len(self.vehicles), n_vehicles))

    def __del__(self):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles])


# class PedestrianPool(object):
#     def __init__(self, client, n_pedestrians):
#         self.client = client
#         self.world = client.get_world()

#         ped_bp = self.world.get_blueprint_library().filter('walker.pedestrian.*')
#         con_bp = self.world.get_blueprint_library().find('controller.ai.walker')

#         spawn_points = [self._get_spawn_point() for _ in range(n_pedestrians)]
#         batch = [carla.command.SpawnActor(np.random.choice(ped_bp), spawn) for spawn in spawn_points]
#         walkers = list()
#         errors = set()

#         for msg in client.apply_batch_sync(batch, True):
#             if msg.error:
#                 errors.add(msg.error)
#             else:
#                 walkers.append(msg.actor_id)

#         if errors:
#             print('\n'.join(errors))

#         batch = [carla.command.SpawnActor(con_bp, carla.Transform(), walker_id) for walker_id in walkers]
#         controllers = list()
#         errors = set()

#         for msg in client.apply_batch_sync(batch, True):
#             if msg.error:
#                 errors.add(msg.error)
#             else:
#                 controllers.append(msg.actor_id)

#         if errors:
#             print('\n'.join(errors))

#         self.walkers = self.world.get_actors(walkers)
#         self.controllers = self.world.get_actors(controllers)

#         for controller in self.controllers:
#             controller.start()
#             controller.go_to_location(self.world.get_random_location_from_navigation())
#             controller.set_max_speed(1.4 + np.random.randn())

#         self.timers = [np.random.randint(60, 600) * 20 for _ in self.controllers]

#         print('%d / %d pedestrians spawned.' % (len(self.walkers), n_pedestrians))

#     def _get_spawn_point(self, n_retry=10):
#         for _ in range(n_retry):
#             spawn = carla.Transform()
#             spawn.location = self.world.get_random_location_from_navigation()

#             if spawn.location is not None:
#                 return spawn

#         raise ValueError('No valid spawns.')

#     def tick(self):
#         for i, controller in enumerate(self.controllers):
#             self.timers[i] -= 1

#             if self.timers[i] <= 0:
#                 self.timers[i] = np.random.randint(60, 600) * 20
#                 controller.go_to_location(self.world.get_random_location_from_navigation())

#     def __del__(self):
#         for controller in self.controllers:
#             controller.stop()

#         self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walkers])
#         self.client.apply_batch([carla.command.DestroyActor(x) for x in self.controllers])


class TrafficCarlaEnv(object):
    def __init__(self, args,
                town='Town01', 
                port=2000, 
                npc_manager='sumo',
                **kwargs):

        # tmpdir = os.path.join(PROJECT_ROOT, "maps")
        # sumo_cfg_file = os.path.join("sumo_integration", "examples", town+".sumocfg")
        
        carla_simulation = CarlaSimulation(args.carla_host, args.carla_port, args.step_length)
        self._client = carla_simulation.client
        self._client.set_timeout(30.0)

        # set_sync_mode(self._client, False)

        self._town_name = town
        self._world = self._client.load_world(town)
        self._map = self._world.get_map()

        self._blueprints = self._world.get_blueprint_library()

        self._tick = 0
        self._player = None
        self.agent = None 
        self.use_agent = args.use_agent
        self._player_sumo_id = None # Need this to get state
        self.npc_manager = npc_manager

        # vehicle, sensor
        self._actor_dict = collections.defaultdict(list)
        self._cameras = dict()

        # For spawning npcs (from spawn_npc_sumo.py)
        current_map = self._map
        # xodr_file = os.path.join(tmpdir, current_map.name + '.xodr')
        xodr_file = os.path.join(CARLA_HOME, "CarlaUE4/Content/Carla/Maps/OpenDrive", current_map.name+'.xodr')
        # current_map.save_to_disk(xodr_file)
        net_file = os.path.join(PROJECT_ROOT, "sumo_integration", "examples", "net", current_map.name + '.net.xml')

        if not os.path.isfile(net_file):
            netconvert_carla(xodr_file, net_file, guess_tls=True)
        basedir = os.path.join(PROJECT_ROOT, "sumo_integration")
        cfg_file = os.path.join(basedir,"examples", current_map.name + '.sumocfg')

        if not os.path.isfile(cfg_file): 
            vtypes_file = os.path.join(basedir, 'examples', 'carlavtypes.rou.xml')
            viewsettings_file = os.path.join(basedir, 'examples', 'viewsettings.xml')
            write_sumocfg_xml(cfg_file, net_file, vtypes_file, viewsettings_file, 0)

        self.sumo_net = sumolib.net.readNet(net_file)
        sumo_simulation = SumoSimulation(cfg_file, args.step_length, args.sumo_host,
                                     args.sumo_port, args.sumo_gui, 1)
        
        
        self.synchronization = SimulationSynchronization(sumo_simulation, carla_simulation, args.tls_manager,
                                                args.sync_vehicle_color, args.sync_vehicle_lights)

        self.safe = args.safe 
        self.number_of_vehicles = args.number_of_vehicles

        self.traffic_manager = self._client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)

        # self._initialize_npcs(n_vehicles=args.number_of_vehicles)
        
    def _set_weather(self, weather_string):
        if weather_string == 'random':
            weather = np.random.choice(WEATHERS)
        else:
            weather = weather_string

        self.weather = weather
        self._world.set_weather(weather)

    def _initialize_npcs(self, n_vehicles=10):
        print("Initializing npcs... ")
        self.number_of_vehicles = n_vehicles

        if self.npc_manager == "sumo":
            # ----------
            # Blueprints
            # ----------
            with open('sumo_integration/vtypes.json') as f:
                vtypes = json.load(f)['carla_blueprints']
            blueprints = vtypes.keys()

            filterv = re.compile('vehicle.*')
            blueprints = list(filter(filterv.search, blueprints))

            if self.safe:
                blueprints = [
                    x for x in blueprints if vtypes[x]['vClass'] not in ('motorcycle', 'bicycle')
                ]
                blueprints = [x for x in blueprints if not x.endswith('isetta')]
                blueprints = [x for x in blueprints if not x.endswith('carlacola')]
                blueprints = [x for x in blueprints if not x.endswith('cybertruck')]
                blueprints = [x for x in blueprints if not x.endswith('t2')]

            if not blueprints:
                raise RuntimeError('No blueprints available due to user restrictions.')

            # --------------
            # Spawn vehicles
            # --------------
            # Spawns sumo NPC vehicles.
            sumo_edges = self.sumo_net.getEdges()

            for i in range(n_vehicles):
                type_id = random.choice(blueprints)
                vclass = vtypes[type_id]['vClass']

                allowed_edges = [e for e in sumo_edges if e.allows(vclass)]
                edge = random.choice(allowed_edges)

                traci.route.add('route_{}'.format(i), [edge.getID()])
                traci.vehicle.add('sumo_{}'.format(i), 'route_{}'.format(i), typeID=type_id)
        else: 
            self._vehicle_pool = VehiclePool(self._client, n_vehicles, self.traffic_manager.get_port())
            self._add_vehiclepool_vehs_to_synch()


    def reset(self, weather='random', n_vehicles=10, n_pedestrians=10, seed=0, ticks=10):
        is_ready = False

        while not is_ready:
            np.random.seed(seed)

            self.synchronization.reset()
            self._clean_up()

            # if init == False:
            self._spawn_player()
            self._initialize_npcs(n_vehicles=n_vehicles)
            
            # for _ in range(ticks):
            #     self.step(warmup=True)

            for _ in range(ticks):
                self.step(warmup=True)

            # self._player_sumo_id = self.synchronization.carla2sumo_ids[self._player.id]
            # self.synchronization.sumo.player_id = self._player_sumo_id

            self._setup_sensors()

            self._set_weather(weather)

            is_ready = self.ready()

    def _spawn_player(self):
        print("Now spawning Player vehicle")
        vehicle_bp = np.random.choice(self._blueprints.filter(VEHICLE_NAME))
        vehicle_bp.set_attribute('role_name', 'hero')

        acceptable_spawn = False 

        while not acceptable_spawn:
            spawn_point = np.random.choice(self._map.get_spawn_points())
            print(spawn_point)
            candidate = self._world.spawn_actor(vehicle_bp, spawn_point)
            
            for i in range(5):
                self.step(warmup=True)

            transform = candidate.get_transform() 
            if transform.location.z > 0.5 or transform.rotation.roll > 0.:
                candidate.destroy() 
            else: 
                acceptable_spawn = True 
                self._player = candidate 

            # self._player = self._world.spawn_actor(vehicle_bp, spawn_point)

        # sumo_id, carla_id = random.choice(list(self.synchronization.sumo2carla_ids.items()))
        # self._player = self._world.get_actor(carla_id) 
        self.agent = BasicAgent(self._player)
        # self._player.set_simulate_physics(False)
        self._actor_dict['player'].append(self._player)
        # self.traffic_manager.ignore_vehicles_percentage(self._player, 100)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self.traffic_manager.auto_lane_change(self._player,True)
        self.traffic_manager.ignore_lights_percentage(self._player, 0)
        self.traffic_manager.ignore_signs_percentage(self._player, 0)
        # self.traffic_manager.vehicle_percentage_speed_difference(self._player,0)
        # self.traffic_manager.auto_lane_change(self._player, True)
        
        
        # Manually add the player to SUMO simulation so we can save the id
        carla_id = self._player.id 
        type_id = BridgeHelper.get_sumo_vtype(self._player)
        color = self._player.attributes.get('color', None) 
        if type_id is not None:
            sumo_actor_id = self.synchronization.sumo.spawn_actor(type_id, color)
            if sumo_actor_id != INVALID_ACTOR_ID:
                self.synchronization.carla2sumo_ids[carla_id] = sumo_actor_id
                self.synchronization.sumo.subscribe(sumo_actor_id)
        
        self._player_sumo_id = sumo_actor_id
        self.synchronization.sumo.player_id = sumo_actor_id
        self.synchronization.carla.player_id = carla_id 
        print("Sync dict: ", self.synchronization.carla2sumo_ids)

    def _add_vehiclepool_vehs_to_synch(self):

        for id in self._vehicle_pool.vehicles:
            carla_actor = self.synchronization.carla.get_actor(id)
            type_id = BridgeHelper.get_sumo_vtype(carla_actor)
            # color = self._player.attributes.get('color', None) 
            color = None
            if type_id is not None:
                sumo_actor_id = self.synchronization.sumo.spawn_actor(type_id, color)
                if sumo_actor_id != INVALID_ACTOR_ID:
                    self.synchronization.carla2sumo_ids[id] = sumo_actor_id
                    self.synchronization.sumo.subscribe(sumo_actor_id)


    def ready(self, ticks=10):
        # for _ in range(ticks):
        #     self.step(warmup=True)
        for x in self._actor_dict['camera']:
            x.get()

        self._time_start = time.time()
        self._tick = 0

        return True if self._player_sumo_id and self.synchronization.sumo.player_id else False 

    def get_state(self):
        '''Get the state of the driver's lane'''
        return self.synchronization.sumo.get_state()

    def step(self, warmup=False):
        # print("Sync dict: ", self.synchronization.carla2sumo_ids)

        if self.agent and self.use_agent and not warmup:
            self._player.apply_control(self.agent.run_step())
            
        self.synchronization.tick() 

        # Updates vehicle routes
        for vehicle_id in traci.vehicle.getIDList():
            route = traci.vehicle.getRoute(vehicle_id)
            index = traci.vehicle.getRouteIndex(vehicle_id)
            vclass = traci.vehicle.getVehicleClass(vehicle_id)

            if index == (len(route) - 1):
                current_edge = self.sumo_net.getEdge(route[index])
                available_edges = list(current_edge.getAllowedOutgoing(vclass).keys())
                if available_edges: 
                    next_edge = random.choice(available_edges)
                    new_route = [current_edge.getID(), next_edge.getID()]
                    traci.vehicle.setRoute(vehicle_id, new_route)

        self._tick += 1

        result = None 

        if not warmup:
            transform = self._player.get_transform()
            velocity = self._player.get_velocity()
            acceleration = self._player.get_acceleration()
            control = self._player.get_control()
            # velocity = self.synchronization.sumo.get_speed(self._player_sumo_id)
            # acceleration = self.synchronization.sumo.get_accel(self._player_sumo_id)

        # Put here for speed (get() busy polls queue).

            if self.synchronization.sumo.has_result():
                state, player_ind = self.get_state() 
                
                result = {key: val.get() for key, val in self._cameras.items()}
                result.update({
                    'wall': time.time() - self._time_start,
                    'tick': self._tick,
                    'x': transform.location.x,
                    'y': transform.location.y,
                    'theta': transform.rotation.yaw,
                    'steer': control.steer,
                    'throttle': control.throttle,
                    'brake': control.brake,
                    'speed': np.linalg.norm([velocity.x, velocity.y, velocity.z]),
                    'accel': np.linalg.norm([acceleration.x, acceleration.y, acceleration.z]),
                    'player_lane_state': state, 
                    'player_ind_in_lane': player_ind, 
                    'fuel_consumption': self.synchronization.sumo.get_playerlane_fuel_consumption()
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
        self.synchronization.close() 
        set_sync_mode(self._client, False)

    def _clean_up(self):
        # self._pedestrian_pool = None
        # self._vehicle_pool = None
        self._cameras.clear()

        # The only actor in _actor_dict should be ego vehicle, 
        # Because synchronization is providing npc vehicles
        for actor_type in list(self._actor_dict.keys()):
            self._actor_dict[actor_type].clear()

        for carla_actor_id in self.synchronization.sumo2carla_ids.values():
            self.synchronization.carla.destroy_actor(carla_actor_id)
        
        for sumo_actor_id in self.synchronization.carla2sumo_ids.values():
            self.synchronization.sumo.destroy_actor(sumo_actor_id)

        self._actor_dict.clear()

        self._tick = 0
        self._time_start = time.time()

        self._player = None
