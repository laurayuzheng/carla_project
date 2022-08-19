import uuid
import argparse
import pathlib

import numpy as np
import torch
import pytorch_lightning as pl
import torchvision
import wandb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image, ImageDraw

from .traffic_map_model_steer import TrafficMapModelSteer
from .models import SegmentationModel, RawController
from .utils.heatmap import ToHeatmap
from .dataset import get_dataset
from .converter import Converter
from . import common

import gym
import math

from .traffic.d_car_following_models import IDMStepLayer
from .traffic.rewards import * 
from .traffic_map_model_accel import TrafficMapModelAccel

import torch
from torch import det
import numpy as np
from FLOW_CONFIG import *

@torch.no_grad()
def visualize(batch, out, between, out_steer, out_accel, loss_point, loss_cmd, target_heatmap):
    images = list()

    for i in range(out.shape[0]):
        _loss_point = loss_point[i] if loss_point is not None else -1
        _loss_cmd = loss_cmd[i] if loss_cmd is not None else -1
        _out = out[i] if out is not None else [-1]
        # _out_cmd = out_cmd[i]
        _out_steer = out_steer[i] if out_steer is not None else [-1]
        _out_accel = out_accel[i] if out_accel is not None else [-1]
        _between = between[i] if between is not None else [-1]

        rgb, topdown, points, target, actions, meta, _, _, _ = [x[i] for x in batch]
        
        _rgb = np.uint8(rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255)
        _target_heatmap = np.uint8(target_heatmap[i].detach().squeeze().cpu().numpy() * 255)
        _target_heatmap = np.stack(3 * [_target_heatmap], 2)
        _target_heatmap = Image.fromarray(_target_heatmap)
        _topdown = Image.fromarray(common.COLOR[topdown.argmax(0).detach().cpu().numpy()])
        _draw = ImageDraw.Draw(_topdown)

        _draw.ellipse((target[0]-2, target[1]-2, target[0]+2, target[1]+2), (255, 255, 255))

        for x, y in points:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))

        for x, y in _out:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        for x, y in _between:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw.ellipse((x-1, y-1, x+1, y+1), (0, 255, 0))

        _draw.text((5, 10), 'Point: %.3f' % _loss_point)
        _draw.text((5, 30), 'Command: %.3f' % _loss_cmd)
        _draw.text((5, 50), 'Meta: %s' % meta)

        _draw.text((5, 90), 'Steer, Accel Label: %.3f %.3f' % tuple(actions))
        _draw.text((5, 110), 'Steer Pred: %.3f' % tuple(_out_steer))
        _draw.text((5, 130), 'Accel Pred: %.3f' % tuple(_out_accel))

        image = np.array(_topdown).transpose(2, 0, 1)
        images.append((_loss_cmd, torch.ByteTensor(image)))

    images.sort(key=lambda x: x[0], reverse=True)

    result = torchvision.utils.make_grid([x[1] for x in images], nrow=4)
    result = wandb.Image(result.numpy().transpose(1, 2, 0))

    return result

class Dagger(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.to_heatmap = ToHeatmap(hparams.heatmap_radius)
        # self.steer_net = TrafficMapModelSteer(hparams)
        # self.steer_net.load_from_checkpoint(hparams.steer_checkpoint)
        # self.steer_net.freeze()
        self.net = SegmentationModel(10, 4, hack=hparams.hack, temperature=hparams.temperature)
        self.controller = RawController(4)
        self.converter = Converter()
        
        self.step_layer = IDMStepLayer.apply

    def forward(self, topdown, target, debug=False): 
        target_heatmap = self.to_heatmap(target, topdown)[:, None]
        out = self.net(torch.cat((topdown, target_heatmap), 1))

        if not debug:
            return out

        return out, (target_heatmap,)

    def d_compute_reward(self, state):
        if self.hparams.reward_type == "avg_velocity":
            return d_average_velocity(state, fail=False)
        else: 
            return d_desired_velocity(state, 30, fail=False)

    def get_actions_loss(self, q_0, rl_actions, rl_indices=[], t_delta=0.1, v0=30., s0=2., T=1.5, a=0.73, b=1.67, delta=4):
        vehicle_length = 5.
        q = torch.zeros_like(q_0).type_as(q_0)
        actions_loss = torch.zeros_like(rl_actions)
        rl_actions_i = 0
        q_clone = q_0.clone()
        
        vs = q_clone[1::2]
        xs = q_clone[0::2]

        last_xs = torch.roll(xs, 1)
        last_vs = torch.roll(vs, 1)

        s_star = s0 + vs*T + (vs * (vs - last_vs))/(2*math.sqrt(a*b))
        interaction_terms = (s_star/(last_xs - xs - vehicle_length))**2
        interaction_terms[0] = 0.

        dv = a * (1 - (vs / v0)**delta - interaction_terms) # calculate acceleration
    
        for i in rl_indices: # use RL vehicle's acceleration action
            if i == 0:
                max_accel = 30 
            else: 
                max_accel = (xs[i-1] - xs[i] + 2*t_delta*vs[i-1] - 2*t_delta*vs[i] + t_delta**2*dv[i-1] + s0) / (t_delta**2)

            actions_loss[rl_actions_i] = max_accel

            rl_actions_i += 1


        return torch.nn.functional.l1_loss(actions_loss, rl_actions, reduction='none').mean().squeeze()

    def simstep(self, last_obs, action, player_index, num_veh):
        d_start_state = last_obs
        d_action = torch.as_tensor(action)
        d_start_state.requires_grad = True
        
        obs_array = [] 
        rewards = []

        for i in range(d_start_state.size(0)):
            vehicles = int(num_veh[i])
            curr_state = d_start_state[i, 0:vehicles*2] # truncate padded state
            obs = self.step_layer(curr_state, d_action[i], [player_index[i]], 1, 0.05, 30, 2, 1, 1, 1.5, 4)
            obs_array.append(obs)
            reward = torch.sigmoid(self.d_compute_reward(obs).type_as(last_obs))
            # rewards.append(reward.unsqueeze(0))
            rewards.append(-0.5 * torch.sigmoid(self.get_actions_loss(curr_state, d_action[i], [player_index[i]], 0.05, 30, 2, 1, 1, 1.5, 4)) + reward)

        # obs_array = torch.cat(obs_array)
        rewards = torch.cat(rewards)
        # obs_array = obs_array.type_as(last_obs)
        rewards = rewards.type_as(last_obs)

        return obs_array, rewards

    def training_step(self, batch, batch_nb): # img never used in map model
        img, topdown, points, target, actions, meta, traffic_state, player_ind, num_veh = batch
        # speed_actions = actions[:,1:2]

        out, (target_heatmap,) = self.forward(topdown, target, debug=True) # Just uses feature extractor of map model
        
        alpha = torch.rand(out.shape).type_as(out)
        between = alpha * out + (1-alpha) * points # Interpolate between predicted waypoints and ground truth waypoints
        out_accel = self.controller(between)
        
        points_cam = out.clone()
        points_cam[..., 0] = (points_cam[..., 0] + 1) / 2 * img.shape[-1]
        points_cam[..., 1] = (points_cam[..., 1] + 1) / 2 * img.shape[-2]
        points_cam = points_cam.squeeze()
        points_world = self.converter.cam_to_world(points_cam)
        desired_speed = torch.norm(points_world[:,0] - points_world[:,1], dim=1) * 2.0 #  * (acceleration*0.05 + speed)
        desired_speed = torch.clamp(desired_speed, min=torch.finfo(torch.float32).eps, max=30)
        # brake = torch.minimum(torch.log(2*desired_speed), 1)
        brake = torch.log(2*desired_speed)

        states, reward = self.simstep(traffic_state, out_accel, player_ind, num_veh)
        
        obs_velocities = torch.Tensor(np.array([obs[1::2].mean() for obs in states], dtype=np.float32)).type_as(traffic_state)
        reward = brake*reward # positive reward if not braking, else negative for acceleration when braking

        loss_point = torch.nn.functional.l1_loss(out, points, reduction='none').mean((1, 2))
        loss = (-0.1 * reward + loss_point).mean()

        metrics = {
                'loss': loss.item(),
                'next_state_avg_vel': obs_velocities.mean().item(),
                'reward; '+self.hparams.reward_type: (reward).mean().item(),
                'train_point_loss': loss_point.mean().item()
                }

        if batch_nb % 250 == 0:
            metrics['train_image'] = visualize(batch, out, between, None, out_accel, loss_point, None, target_heatmap)

        self.logger.log_metrics(metrics, self.global_step)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        img, topdown, points, target, actions, meta, traffic_state, player_ind, num_veh = batch
        steer_actions = actions[:,0:1]

        out, (target_heatmap,) = self.forward(topdown, target, debug=True)

        alpha = 0.0
        between = alpha * out + (1-alpha) * points

        out_accel = self.controller(between)

        points_cam = out.clone()
        points_cam[..., 0] = (points_cam[..., 0] + 1) / 2 * img.shape[-1]
        points_cam[..., 1] = (points_cam[..., 1] + 1) / 2 * img.shape[-2]
        points_cam = points_cam.squeeze()
        points_world = self.converter.cam_to_world(points_cam)
        desired_speed = torch.norm(points_world[:,0] - points_world[:,1], dim=1) #  * (acceleration*0.05 + speed)
        desired_speed = torch.clamp(desired_speed, min=torch.finfo(torch.float32).eps)
        brake = torch.log(2*desired_speed)

        states, reward = self.simstep(traffic_state, out_accel, player_ind, num_veh)
        obs_velocities = torch.Tensor(np.array([obs[1::2].mean() for obs in states], dtype=np.float32)).type_as(traffic_state)
        reward = brake*reward # flip reward for loss (higher reward is better)

        loss_point = torch.nn.functional.l1_loss(out, points, reduction='none').mean((1, 2))
        loss = (-0.1 * reward + loss_point).mean()

        if batch_nb == 0:
            self.logger.log_metrics({
                'val_image': visualize(batch, out, between, None, out_accel, loss_point, None, target_heatmap)
                }, self.global_step)

        return {
                'val_loss': loss.item(),
                'next_state_avg_vel': obs_velocities.mean().item(),
                'reward': reward.mean().item(),
                'val_point_loss': loss_point.mean().item()
                }

    def validation_epoch_end(self, batch_metrics):
        results = dict()

        for metrics in batch_metrics:
            for key in metrics:
                if key not in results:
                    results[key] = list()

                results[key].append(metrics[key])

        summary = {key: np.mean(val) for key, val in results.items()}
        self.logger.log_metrics(summary, self.global_step)

        return summary

    def configure_optimizers(self):
        optim = torch.optim.Adam(
                list(self.net.parameters()) + list(self.controller.parameters()),
                lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode='min', factor=0.5, patience=5, min_lr=1e-6,
                verbose=True)

        return [optim], [scheduler]

    def train_dataloader(self):
        return get_dataset(self.hparams.dataset_dir, True, self.hparams.batch_size, sample_by=self.hparams.sample_by, traffic=True)

    def val_dataloader(self):
        return get_dataset(self.hparams.dataset_dir, False, self.hparams.batch_size, sample_by=self.hparams.sample_by, traffic=True)

def main(hparams):
    model = TrafficMapModelAccel(hparams)
    logger = WandbLogger(id=hparams.id, save_dir=str(hparams.save_dir), project='traffic_accel')
    checkpoint_callback = ModelCheckpoint(hparams.save_dir, save_top_k=1)

    try:
        resume_from_checkpoint = sorted(hparams.save_dir.glob('*.ckpt'))[-1]
    except:
        resume_from_checkpoint = None

    gpus = 0 if hparams.cpu else -1 

    trainer = pl.Trainer(
            gpus=gpus, max_epochs=hparams.max_epochs,
            resume_from_checkpoint=resume_from_checkpoint,
            logger=logger, checkpoint_callback=checkpoint_callback)

    trainer.fit(model)

    wandb.save(str(hparams.save_dir / '*.ckpt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--save_dir', type=pathlib.Path, default='checkpoints')
    parser.add_argument('--id', type=str, default=uuid.uuid4().hex)

    parser.add_argument('--heatmap_radius', type=int, default=5)
    parser.add_argument('--sample_by', type=str, choices=['none', 'even', 'speed', 'steer'], default='even')
    parser.add_argument('--reward_type', type=str, default="desired_velocity", choices=["avg_velocity", "desired_velocity"])
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument('--hack', action='store_true', default=False)

    # Data args.
    parser.add_argument('--dataset_dir', type=pathlib.Path, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--steer_checkpoint', type=pathlib.Path, required=True)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--cpu', action='store_true', default=False)

    parsed = parser.parse_args()
    parsed.save_dir = parsed.save_dir / parsed.id
    parsed.save_dir.mkdir(parents=True, exist_ok=True)

    main(parsed)
