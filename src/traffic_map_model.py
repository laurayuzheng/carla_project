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

from .models import SegmentationModel, RawController
from .utils.heatmap import ToHeatmap
from .dataset import get_dataset
from . import common

import gym

from .traffic.d_car_following_models import IDMStepLayer
from .traffic.rewards import *

import torch
from torch import det
import numpy as np
from FLOW_CONFIG import *

@torch.no_grad()
def visualize(batch, out, between, out_cmd, loss_point, loss_cmd, target_heatmap):
    images = list()

    for i in range(out.shape[0]):
        _loss_point = loss_point[i]
        _loss_cmd = loss_cmd[i]
        _out = out[i]
        _out_cmd = out_cmd[i]
        _between = between[i]

        rgb, topdown, points, target, actions, meta, _, _ = [x[i] for x in batch]

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

        _draw.text((5, 90), 'Raw: %.3f %.3f' % tuple(actions))
        _draw.text((5, 110), 'Pred: %.3f %.3f' % tuple(_out_cmd))

        image = np.array(_topdown).transpose(2, 0, 1)
        images.append((_loss_cmd, torch.ByteTensor(image)))

    images.sort(key=lambda x: x[0], reverse=True)

    result = torchvision.utils.make_grid([x[1] for x in images], nrow=4)
    result = wandb.Image(result.numpy().transpose(1, 2, 0))

    return result


class TrafficMapModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.to_heatmap = ToHeatmap(hparams.heatmap_radius)
        self.net = SegmentationModel(10, 4, hack=hparams.hack, temperature=hparams.temperature)
        self.controller = RawController(4)
        # constructor, env_name = env_constructor(params=flow_params, version=0)
        # eval_flow_params = flow_params.copy()
        # constructor2, env_name2 = env_constructor(params=eval_flow_params, version=1)
        # self.flow_train_env = constructor() 
        # self.flow_eval_env = constructor2()
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
            return d_desired_velocity(state, 50, fail=False)

    def simstep(self, last_obs, action, player_index):
        d_start_state = last_obs
        d_action = torch.as_tensor(action)
        d_start_state.requires_grad = True
        
        obs_array = [] 
        rewards = []

        for i in range(d_start_state.size(0)):
            obs = self.step_layer(d_start_state[i], d_action[i].unsqueeze(0), [player_index[i]], 1, 1, 30, 2, 1, 1, 1.5, 4)
            obs_array.append(obs.unsqueeze(0))
            reward = self.d_compute_reward(obs)
            rewards.append(reward.unsqueeze(0))

        obs_array = torch.cat(obs_array)
        rewards = torch.cat(rewards)
        obs_array = obs_array.type_as(last_obs)
        rewards = rewards.type_as(last_obs)

        return obs_array, rewards

    def training_step(self, batch, batch_nb): # img never used in map model
        img, topdown, points, target, actions, meta, traffic_state, player_ind = batch
        # _, traffic_state, player_ind = meta 
        steer_actions = actions[:,0:1]
        current_speed = actions[:,1]

        out, (target_heatmap,) = self.forward(topdown, target, debug=True)
        
        alpha = torch.rand(out.shape).type_as(out)
        between = alpha * out + (1-alpha) * points # Interpolate between predicted waypoints and ground truth waypoints
        out_cmd = self.controller(between) # Outputs 3 things: 'steer', 'target_speed', 'accel', 

        # print("******************************")
        # print(out_cmd.size())
        accel = out_cmd[:,1] # matters with reward maximization
        target_speeds = 0.1*accel + current_speed
        # print(accel.size())

        # out_cmd = torch.cat([out_cmd[:,0:2], out_cmd[:,3:]], dim=1) # remove for command loss: don't want controller and sim to "fight"
        compute_out_cmd = out_cmd[:,0:1]
        state, reward = self.simstep(traffic_state, accel, player_ind)
        # print(reward)
        average_velocity = state[1::2].mean()
        reward = -1*reward # flip reward for loss (higher reward is better)

        loss_point = torch.nn.functional.l1_loss(out, points, reduction='none').mean((1, 2))
        loss_cmd_raw = torch.nn.functional.l1_loss(compute_out_cmd, steer_actions, reduction='none')

        loss_cmd = loss_cmd_raw.mean(1)
        loss = ((loss_point + self.hparams.command_coefficient * loss_cmd) + self.hparams.reward_coefficient * reward).mean()

        metrics = {
                'loss': loss.item(),
                'point_loss': loss_point.mean().item(),
                # 'cmd_loss': loss_cmd.mean().item(),
                'loss_steer': loss_cmd.mean().item(),
                'next_state_avg_vel': average_velocity.item(),
                # 'loss_speed': loss_cmd_raw[:, 1].mean().item(), 
                # 'loss_accel': loss_cmd_raw[:, 2].mean().item(), 
                # 'loss_throttle': loss_cmd_raw[:, 2].mean().item(), 
                # 'loss_brake': loss_cmd_raw[:, 3].mean().item(), 
                'reward; '+self.hparams.reward_type: (reward*-1).mean().item()
                }

        if batch_nb % 250 == 0:
            metrics['train_image'] = visualize(batch, out, between, out_cmd, loss_point, loss_cmd, target_heatmap)

        self.logger.log_metrics(metrics, self.global_step)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        img, topdown, points, target, actions, meta, traffic_state, player_ind = batch
        steer_actions = actions[:,0:1]
        current_speed = actions[:,1]
        # _, traffic_state, player_ind = meta 
        out, (target_heatmap,) = self.forward(topdown, target, debug=True)

        alpha = 0.0
        between = alpha * out + (1-alpha) * points
        out_cmd = self.controller(between)
        out_cmd_pred = self.controller(out)

        accel = out_cmd_pred[:, 1] # matters with reward maximization
        target_speeds = 0.1*accel + current_speed

        compute_out_cmd_pred = out_cmd_pred[:,0:1]
        compute_out_cmd = out_cmd[:,0:1]
        state, reward = self.simstep(traffic_state, accel, player_ind)
        average_velocity = state[1::2].mean()
        reward = -1*reward # flip reward for loss (higher reward is better)


        loss_point = torch.nn.functional.l1_loss(out, points, reduction='none').mean((1, 2))
        loss_cmd_raw = torch.nn.functional.l1_loss(compute_out_cmd, steer_actions, reduction='none')
        loss_cmd_pred_raw = torch.nn.functional.l1_loss(compute_out_cmd_pred, steer_actions, reduction='none')

        loss_cmd = loss_cmd_raw.mean(1)
        loss = ((loss_point + self.hparams.command_coefficient * loss_cmd) + self.hparams.reward_coefficient * reward).mean()

        if batch_nb == 0:
            self.logger.log_metrics({
                'val_image': visualize(batch, out, between, out_cmd, loss_point, loss_cmd, target_heatmap)
                }, self.global_step)

        return {
                'val_loss': loss.item(),
                'val_point_loss': loss_point.mean().item(),

                # 'val_cmd_loss': loss_cmd_raw.mean(1).mean().item(),
                'val_steer_loss': loss_cmd.mean().item(),
                # 'val_speed_loss': loss_cmd_raw[:, 1].mean().item(),

                # 'val_cmd_pred_loss': loss_cmd_pred_raw.mean(1).mean().item(),
                'val_steer_pred_loss': loss_cmd_pred_raw[:, 0].mean().item(), 
                # 'predicted_target_speed': target_speeds.mean().item(),
                'next_state_avg_vel': average_velocity.item(),
                # 'val_speed_pred_loss': loss_cmd_pred_raw[:, 1].mean().item(),
                # 'loss_accel': loss_cmd_raw[:, 2].mean().item(), 
                # 'loss_throttle': loss_cmd_raw[:, 2].mean().item(), 
                # 'loss_brake': loss_cmd_raw[:, 3].mean().item(),
                'reward': (reward*-1).mean().item()
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
    model = TrafficMapModel(hparams)
    logger = WandbLogger(id=hparams.id, save_dir=str(hparams.save_dir), project='traffic_stage_1')
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
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--save_dir', type=pathlib.Path, default='checkpoints')
    parser.add_argument('--id', type=str, default=uuid.uuid4().hex)

    parser.add_argument('--heatmap_radius', type=int, default=5)
    parser.add_argument('--sample_by', type=str, choices=['none', 'even', 'speed', 'steer'], default='even')
    parser.add_argument('--reward_type', type=str, default="avg_velocity", choices=["avg_velocity", "desired_velocity"])
    parser.add_argument('--command_coefficient', type=float, default=0.1)
    parser.add_argument('--reward_coefficient', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument('--hack', action='store_true', default=False)

    # Data args.
    parser.add_argument('--dataset_dir', type=pathlib.Path, required=True)
    parser.add_argument('--batch_size', type=int, default=32)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--cpu', action='store_true', default=False)

    parsed = parser.parse_args()
    parsed.save_dir = parsed.save_dir / parsed.id
    parsed.save_dir.mkdir(parents=True, exist_ok=True)

    main(parsed)
