import sys, os

from pathlib import Path

import numpy as np
import tqdm
import carla
import cv2
import pandas as pd
import argparse
import logging 
import shutil

from PIL import Image

from .t_carla_env import TrafficCarlaEnv, PRESET_WEATHERS, PRESET_WEATHERS_STRING
from .common import COLOR

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


EPISODE_LENGTH = 500
EPISODES = 20
FRAME_SKIP = 5
SAVE_PATH = Path('/scratch/2020_CARLA_challenge/data/traffic_data')
DEBUG = False
WARMUP_STEPS=50

NUM_VEHICLES_TOWN_DICT = {
    1: 100, 
    3: 100,
    4: 200, 
    5: 150, 
    6: 150, 
    7: 100, 
    10: 150

}

def collect_episode(env, save_dir):
    save_dir.mkdir()

    (save_dir / 'rgb_left').mkdir()
    (save_dir / 'rgb').mkdir()
    (save_dir / 'rgb_right').mkdir()
    (save_dir / 'map').mkdir()

    # env._client.start_recorder(str(save_dir / 'recording.log'))

    spectator = env._world.get_spectator()
    spectator.set_transform(
            carla.Transform(
                env._player.get_location() + carla.Location(z=50),
                carla.Rotation(pitch=-90)))

    measurements = list()

    for step in tqdm.tqdm(range(EPISODE_LENGTH * FRAME_SKIP + WARMUP_STEPS)):
        
        observations = env.step()

        if step < WARMUP_STEPS or step % FRAME_SKIP != 0 or not observations:
            continue

        index = (step - WARMUP_STEPS) // FRAME_SKIP

        if index % 5 == 0: 
            env._set_weather('random') # change weather 

        rgb = observations.pop('rgb')
        rgb_left = observations.pop('rgb_left')
        rgb_right = observations.pop('rgb_right')
        topdown = observations.pop('topdown')

        measurements.append(observations)

        if DEBUG:
            cv2.imshow('rgb', cv2.cvtColor(np.hstack((rgb_left, rgb, rgb_right)), cv2.COLOR_BGR2RGB))
            cv2.imshow('topdown', cv2.cvtColor(COLOR[topdown], cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

        Image.fromarray(rgb_left).save(save_dir / 'rgb_left' / ('%04d.png' % index))
        Image.fromarray(rgb).save(save_dir / 'rgb' / ('%04d.png' % index))
        Image.fromarray(rgb_right).save(save_dir / 'rgb_right' / ('%04d.png' % index))
        Image.fromarray(topdown).save(save_dir / 'map' / ('%04d.png' % index))

    pd.DataFrame(measurements).to_csv(save_dir / 'measurements.csv', index=False)

    # env._client.stop_recorder()


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--carla-host',
                           metavar='H',
                           default='127.0.0.1',
                           help='IP of the carla host server (default: 127.0.0.1)')
    argparser.add_argument('--carla-port',
                           metavar='P',
                           default=2000,
                           type=int,
                           help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--sumo-host',
                           metavar='H',
                           default=None,
                           help='IP of the sumo host server (default: 127.0.0.1)')
    argparser.add_argument('--sumo-port',
                           metavar='P',
                           default=None,
                           type=int,
                           help='TCP port to listen to (default: 8813)')
    argparser.add_argument('--sumo-gui', action='store_true', help='run the gui version of sumo')
    argparser.add_argument('--step-length',
                           default=1/10,
                           type=float,
                           help='set fixed delta seconds (default: 0.05s)')
    argparser.add_argument('--client-order',
                           metavar='TRACI_CLIENT_ORDER',
                           default=1,
                           type=int,
                           help='client order number for the co-simulation TraCI connection (default: 1)')
    argparser.add_argument('--sync-vehicle-lights',
                           action='store_true',
                           help='synchronize vehicle lights state (default: False)')
    argparser.add_argument('--sync-vehicle-color',
                           action='store_true',
                           help='synchronize vehicle color (default: False)')
    argparser.add_argument('--sync-vehicle-all',
                           action='store_true',
                           help='synchronize all vehicle properties (default: False)')
    argparser.add_argument('--tls-manager',
                           type=str, 
                           choices=['none', 'sumo', 'carla'], 
                           help="select traffic light manager (default: none)", 
                           default='carla') 
    argparser.add_argument('-n', 
                           '--number-of-vehicles', 
                           metavar='N', 
                           default=50,
                           type=int,
                           help='number of vehicles (default: 50)')
    argparser.add_argument('--safe', 
                           action='store_true', 
                           help='avoid spawning vehicles prone to accidents') 
    argparser.add_argument('--use-agent', 
                           action='store_true', 
                           help='use CARLA AI')

    argparser.add_argument('--debug', action='store_true', help='enable debug messages')
    args = argparser.parse_args()

    if args.sync_vehicle_all is True:
        args.sync_vehicle_lights = True
        args.sync_vehicle_color = True

    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    np.random.seed(0)
    
    # for nv in [80, 150]:
    #     args.number_of_vehicles = nv
    # for wi in [1]: #  8, 12
    for i in [5]: # [1, 4, 5, 10]:
        # if i == 1 and wi == 1: 
        #     continue 

        args.safe = True 

        for episode in range(EPISODES):
            if i == 10:
                town = 'Town10HD'
            else:
                town = f'Town{i:02d}'

            args.number_of_vehicles = NUM_VEHICLES_TOWN_DICT[i]

            success = False 

            # while not success: 
            #     env = None 
            #     try: 
            save_path = SAVE_PATH / ('%03d_%s' % (len(list(SAVE_PATH.glob('*'))), town))
            with TrafficCarlaEnv(args, town=town, npc_manager="sumo") as env:
                weather_setting = PRESET_WEATHERS[1]
                env.reset(
                        weather=weather_setting,
                        ticks=20, 
                        # n_vehicles=np.random.choice([100, 130, 200]),
                        n_vehicles=args.number_of_vehicles,
                        n_pedestrians=np.random.choice([50, 100, 200]),
                        seed=np.random.randint(0, 256))
                if not args.use_agent:
                    env._player.set_autopilot(True)
                collect_episode(env, save_path)
                env._clean_up()
                success = True 
                    # except Exception: # Try again if fails
                    #     if env: 
                    #         env._clean_up()
                    #     if os.path.exists(str(save_path)):
                    #         shutil.rmtree(str(save_path))
                    #     pass


if __name__ == '__main__':
    main()
