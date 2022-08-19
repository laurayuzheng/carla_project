import uuid
import argparse
import pathlib

import numpy as np
import math
import torch
import torchvision
import pytorch_lightning as pl
import wandb


from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image, ImageDraw

from .traffic_map_model_accel import TrafficMapModelAccel
from .models import SegmentationModel, RawController
from .utils.heatmap import ToHeatmap
from .dataset import get_dataset, get_dagger_dataset
from .converter import Converter
from . import common
from .scripts.cluster_points import points as RANDOM_POINTS

from .traffic.d_car_following_models import IDMStepLayer
from .traffic.rewards import *


@torch.no_grad()
def viz(batch, out, pred_accel, target_cam, lbl_cam, lbl_map, point_loss, reward):
    images = list()

    for i in range(out.shape[0]):
        _point_loss = point_loss[i]
        _reward = reward[i]

        _out = out[i]
        _target = target_cam[i]
        _lbl_cam = lbl_cam[i]
        _lbl_map = lbl_map[i]

        _out_ctrl = pred_accel[i]

        img, topdown, points, target, actions, meta, traffic_state, player_ind, num_veh = [x[i] for x in batch]

        _rgb = Image.fromarray((255 * img[:3].cpu()).byte().numpy().transpose(1, 2, 0))
        _draw_rgb = ImageDraw.Draw(_rgb)
        _draw_rgb.text((5, 10), 'Point loss: %.3f' % _point_loss)
        _draw_rgb.text((5, 30), 'Accel Reward: %.3f' % _reward)
        _draw_rgb.text((5, 50), 'Pred: %.3f ' % tuple(_out_ctrl))
        # _draw_rgb.text((5, 70), 'Label: %.3f ' % tuple(_ctrl_map))
        _draw_rgb.text((5, 70), 'Meta: %s' % meta)
        # _draw_rgb.text((5, 110), 'Expert speed: %.3f ' % _speed)
        # _draw_rgb.text((5, 130), 'Our speed: %.3f ' % _our_speed)
        _draw_rgb.ellipse((_target[0]-3, _target[1]-3, _target[0]+3, _target[1]+3), (255, 255, 255))

        for x, y in _out:
            x = (x + 1) / 2 * _rgb.width
            y = (y + 1) / 2 * _rgb.height

            _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (0, 255, 0))

        for x, y in _lbl_cam:
            x = (x + 1) / 2 * _rgb.width
            y = (y + 1) / 2 * _rgb.height

            _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        _topdown = Image.fromarray(common.COLOR[topdown.argmax(0).cpu().numpy()])
        _draw_map = ImageDraw.Draw(_topdown)

        for x, y in points:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw_map.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))

        for x, y in _lbl_map:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw_map.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        _topdown.thumbnail(_rgb.size)

        image = np.hstack((_rgb, _topdown)).transpose(2, 0, 1)
        images.append((_reward, torch.ByteTensor(image)))

    images.sort(key=lambda x: x[0], reverse=True)

    result = torchvision.utils.make_grid([x[1] for x in images], nrow=4)
    result = wandb.Image(result.numpy().transpose(1, 2, 0))

    return result


class TrafficImageModel(pl.LightningModule):
    def __init__(self, hparams, dagger=False):
        super().__init__()

        self.hparams = hparams
        self.to_heatmap = ToHeatmap(hparams.heatmap_radius)
        self.dagger = dagger

        # if teacher_path:
        #     print("Loading teacher weights from checkpoint: ", teacher_path)
            # self.teacher = TrafficMapModelAccel.load_from_checkpoint(teacher_path)
        #     self.teacher.freeze()

        self.net = SegmentationModel(10, 4, hack=hparams.hack, temperature=hparams.temperature)
        self.converter = Converter()
        self.controller = RawController(4, n_classes=1)
        self.step_layer = IDMStepLayer.apply

    def forward(self, img, target):
        target_cam = self.converter.map_to_cam(target)
        target_heatmap_cam = self.to_heatmap(target, img)[:, None]
        out = self.net(torch.cat((img, target_heatmap_cam), 1))

        return out, (target_cam, target_heatmap_cam)

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

        rewards = torch.cat(rewards)
        rewards = rewards.type_as(last_obs)

        return obs_array, rewards


    @torch.no_grad()
    def _get_labels(self, topdown, target):
        out, (target_heatmap,) = self.teacher.forward(topdown, target, debug=True)
        control_accel = self.teacher.controller(out)

        return out, control_accel, (target_heatmap,)

    def training_step(self, batch, batch_nb):
        img, topdown, points, target, actions, meta, traffic_state, player_ind, num_veh = batch
                
        # Ground truth command.
        # lbl_map, ctrl_map, (target_heatmap,) = self._get_labels(topdown, target)
        lbl_map = points # Use ground truth points here
        lbl_cam = self.converter.map_to_cam((lbl_map + 1) / 2 * 256)
        lbl_cam[..., 0] = (lbl_cam[..., 0] / 256) * 2 - 1
        lbl_cam[..., 1] = (lbl_cam[..., 1] / 144) * 2 - 1

        out, (target_cam, target_heatmap_cam) = self.forward(img, target)

        alpha = torch.rand(out.shape[0], out.shape[1], 1).type_as(out)
        between = alpha * out + (1-alpha) * lbl_cam
        out_accel = self.controller(between) # out_ctrl
        
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

        loss_point = torch.nn.functional.l1_loss(out, lbl_cam, reduction='none').mean((1, 2))
        loss = (-1 * self.hparams.command_coefficient * reward + loss_point).mean()

        loss_gt_mean = loss.mean()

        metrics = {
                'train_loss': loss_gt_mean.item(),

                'train_point': loss_point.mean().item(),
                'train_reward': reward.mean().item(),
                'train_next_state_avg_vel': obs_velocities.mean().item(),
                }

        if batch_nb % 250 == 0:
            metrics['train_image'] = viz(batch, out, out_accel, target_cam, lbl_cam, lbl_map, loss_point, reward)

        self.logger.log_metrics(metrics, self.global_step)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        img, topdown, points, target, actions, meta, traffic_state, player_ind, num_veh = batch
        # speeds = actions[:,1]

        # Ground truth command.
        # lbl_map, ctrl_map, (target_heatmap,) = self._get_labels(topdown, target)
        lbl_map = points
        lbl_cam = self.converter.map_to_cam((lbl_map + 1) / 2 * 256)
        lbl_cam[..., 0] = (lbl_cam[..., 0] / 256) * 2 - 1
        lbl_cam[..., 1] = (lbl_cam[..., 1] / 144) * 2 - 1

        out, (target_cam, target_heatmap_cam) = self.forward(img, target)
        out_accel = self.controller(out)
        # out_accel_gt = self.controller(lbl_cam)

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

        point_loss = torch.nn.functional.l1_loss(out, lbl_cam, reduction='none').mean((1, 2))

        loss_gt = (-1 * self.hparams.command_coefficient * reward + point_loss).mean()
        loss_gt_mean = loss_gt.mean()

        if batch_nb == 0:
            self.logger.log_metrics({
                'val_image': viz(batch, out, out_accel, target_cam, lbl_cam, lbl_map, point_loss, reward),
                }, self.global_step)

        return {
                'val_loss': (loss_gt_mean).item(),

                'val_point': point_loss.mean().item(),
                'val_reward': reward.mean().item(),
                'val_next_state_avg_vel': obs_velocities.mean().item(),
                }

    def validation_epoch_end(self, outputs):
        results = dict()

        for output in outputs:
            for key in output:
                if key not in results:
                    results[key] = list()

                results[key].append(output[key])

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
        if self.dagger:
            return get_dagger_dataset(self.hparams.dataset_dir, True, self.hparams.batch_size, sample_by=self.hparams.sample_by, traffic=True)
        else:
            return get_dataset(self.hparams.dataset_dir, True, self.hparams.batch_size, sample_by=self.hparams.sample_by, traffic=True)

    def val_dataloader(self):
        if self.dagger:
            return get_dagger_dataset(self.hparams.dataset_dir, False, self.hparams.batch_size, sample_by=self.hparams.sample_by, traffic=True)
        else:
            return get_dataset(self.hparams.dataset_dir, False, self.hparams.batch_size, sample_by=self.hparams.sample_by, traffic=True)

    def state_dict(self):
        return {k: v for k, v in super().state_dict().items() if 'teacher' not in k}

    def load_state_dict(self, state_dict):
        errors = super().load_state_dict(state_dict, strict=False)

        print(errors)


def main(hparams):
    try:
        resume_from_checkpoint = sorted(hparams.save_dir.glob('*.ckpt'))[-1]
    except:
        resume_from_checkpoint = None

    model = TrafficImageModel(hparams, teacher_path=hparams.teacher_path)
    logger = WandbLogger(id=hparams.id, save_dir=str(hparams.save_dir), project='stage_2')
    checkpoint_callback = ModelCheckpoint(hparams.save_dir, 
                                            # monitor="train_loss",
                                            # mode="min",
                                            save_top_k=1)

    trainer = pl.Trainer(
            gpus=-1, max_epochs=hparams.max_epochs,
            resume_from_checkpoint=resume_from_checkpoint,
            logger=logger, checkpoint_callback=checkpoint_callback)

    trainer.fit(model)

    wandb.save(str(hparams.save_dir / '*.ckpt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--save_dir', type=pathlib.Path, default='checkpoints')
    parser.add_argument('--id', type=str, default=uuid.uuid4().hex)

    parser.add_argument('--teacher_path', type=pathlib.Path, required=True)

    # Model args.
    parser.add_argument('--heatmap_radius', type=int, default=5)
    parser.add_argument('--sample_by', type=str, choices=['none', 'even', 'speed', 'steer'], default='even')
    parser.add_argument('--command_coefficient', type=float, default=0.1)
    parser.add_argument('--reward_coefficient', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=5.0)
    parser.add_argument('--hack', action='store_true', default=False)
    parser.add_argument('--reward_type', type=str, default="desired_velocity", choices=["avg_velocity", "desired_velocity"])

    # Data args.
    parser.add_argument('--dataset_dir', type=pathlib.Path, required=True)
    parser.add_argument('--batch_size', type=int, default=16)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parsed = parser.parse_args()
    parsed.teacher_path = parsed.teacher_path.resolve()
    parsed.save_dir = parsed.save_dir.resolve() / parsed.id
    parsed.save_dir.mkdir(parents=True, exist_ok=True)

    main(parsed)
