import uuid
import argparse
import pathlib

import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
import wandb
# import flow 
# from flow.core.util import ensure_dir
# from flow.utils.registry import env_constructor
# from flow.utils.rllib import FlowParamsEncoder, get_flow_params
# from flow.utils.registry import make_create_env


from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image, ImageDraw

from .map_model import MapModel
from .models import SegmentationModel, RawController
from .utils.heatmap import ToHeatmap
from .dataset import get_dataset
from .converter import Converter
from . import common
from .scripts.cluster_points import points as RANDOM_POINTS


@torch.no_grad()
def viz(batch, out, out_ctrl, target_cam, point_loss, ctrl_loss):
    images = list()

    for i in range(out.shape[0]):
        _point_loss = point_loss[i]
        _ctrl_loss = ctrl_loss[i]

        _out = out[i]
        _target = target_cam[i]
        # _lbl_cam = lbl_cam[i]
        # _lbl_map = lbl_map[i]

        _out_ctrl = out_ctrl[i]
        # _ctrl_map = ctrl_map[i]

        img, topdown, points, _, actions, meta = [x[i] for x in batch]

        _rgb = Image.fromarray((255 * img[:3].cpu()).byte().numpy().transpose(1, 2, 0))
        _draw_rgb = ImageDraw.Draw(_rgb)
        _draw_rgb.text((5, 10), 'Point loss: %.3f' % _point_loss)
        _draw_rgb.text((5, 30), 'Control loss: %.3f' % _ctrl_loss)
        _draw_rgb.text((5, 50), 'Raw: %.3f %.3f' % tuple(_out_ctrl))
        # _draw_rgb.text((5, 70), 'Pred: %.3f %.3f' % tuple(_ctrl_map))
        _draw_rgb.text((5, 90), 'Meta: %s' % meta)
        _draw_rgb.ellipse((_target[0]-3, _target[1]-3, _target[0]+3, _target[1]+3), (255, 255, 255))

        for x, y in _out:
            x = (x + 1) / 2 * _rgb.width
            y = (y + 1) / 2 * _rgb.height

            _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (0, 255, 0))

        # for x, y in _lbl_cam:
        #     x = (x + 1) / 2 * _rgb.width
        #     y = (y + 1) / 2 * _rgb.height

        #     _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        _topdown = Image.fromarray(common.COLOR[topdown.argmax(0).cpu().numpy()])
        _draw_map = ImageDraw.Draw(_topdown)

        for x, y in points:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw_map.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))

        # for x, y in _lbl_map:
        #     x = (x + 1) / 2 * 256
        #     y = (y + 1) / 2 * 256

        #     _draw_map.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        _topdown.thumbnail(_rgb.size)

        image = np.hstack((_rgb, _topdown)).transpose(2, 0, 1)
        images.append((_ctrl_loss, torch.ByteTensor(image)))

    images.sort(key=lambda x: x[0], reverse=True)

    result = torchvision.utils.make_grid([x[1] for x in images], nrow=4)
    result = wandb.Image(result.numpy().transpose(1, 2, 0))

    return result


class TrafficImageModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.to_heatmap = ToHeatmap(hparams.heatmap_radius)
        self.use_cpu = hparams.cpu

        # if teacher_path:
        #     self.teacher = MapModel.load_from_checkpoint(teacher_path)
        #     self.teacher.freeze()

        self.net = SegmentationModel(10, 4, hack=hparams.hack, temperature=hparams.temperature)
        self.converter = Converter()
        self.controller = RawController(4)

        # from flow.controllers import IDMController, StaticLaneChanger, ContinuousRouter
        # from flow.core.params import SumoParams, EnvParams, NetParams
        # from flow.core.params import VehicleParams, SumoCarFollowingParams
        # from flow.envs.ring.d_accel import ADDITIONAL_ENV_PARAMS
        # from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS
        # from flow.envs import dAccelEnv
        # from flow.networks import FigureEightNetwork

        # # Set up flow simulation here 
        # vehicles = VehicleParams()
        # vehicles.add(
        #     veh_id="idm",
        #     acceleration_controller=(IDMController, {}),
        #     lane_change_controller=(StaticLaneChanger, {}),
        #     routing_controller=(ContinuousRouter, {}),
        #     car_following_params=SumoCarFollowingParams(
        #         speed_mode="obey_safe_speed",
        #         decel=1.5,
        #     ),
        #     initial_speed=0,
        #     num_vehicles=14)


        # self.flow_params = dict(
        #     # name of the experiment
        #     exp_tag='figure8',
        #     # name of the flow environment the experiment is running on
        #     env_name=dAccelEnv,
        #     # name of the network class the experiment is running on
        #     network=FigureEightNetwork,
        #     # simulator that is used by the experiment
        #     simulator='traci',
        #     # sumo-related parameters (see flow.core.params.SumoParams)
        #     sim=SumoParams(
        #         render=True,
        #     ),
        #     # environment related parameters (see flow.core.params.EnvParams)
        #     env=EnvParams(
        #         horizon=1500,
        #         additional_params=ADDITIONAL_ENV_PARAMS.copy(),
        #     ),
        #     # network-related parameters (see flow.core.params.NetParams and the
        #     # network's documentation or ADDITIONAL_NET_PARAMS component)
        #     net=NetParams(
        #         additional_params=ADDITIONAL_NET_PARAMS.copy(),
        #     ),
        #     # vehicles to be placed in the network at the start of a rollout (see
        #     # flow.core.params.VehicleParams)
        #     veh=vehicles,
        # )

    def forward(self, img, target):
        target_cam = self.converter.map_to_cam(target)
        target_heatmap_cam = self.to_heatmap(target, img)[:, None]
        out = self.net(torch.cat((img, target_heatmap_cam), 1))

        return out, (target_cam, target_heatmap_cam)

    # @torch.no_grad()
    # def _get_labels(self, topdown, target):
    #     out, (target_heatmap,) = self.teacher.forward(topdown, target, debug=True)
    #     control = self.teacher.controller(out)

    #     return out, control, (target_heatmap,)

    def training_step(self, batch, batch_nb):
        img, topdown, points, target, actions, meta = batch

        out, (target_cam, target_heatmap_cam) = self.forward(img, target)

        alpha = torch.rand(out.shape[0], out.shape[1], 1).type_as(out)
        # between = alpha * out + (1-alpha) * lbl_cam
        between = alpha * out + (1-alpha) * points
        out_ctrl = self.controller(between)

        point_loss = torch.nn.functional.l1_loss(out, points, reduction='none').mean((1, 2))
        ctrl_loss_raw = torch.nn.functional.l1_loss(out_ctrl, actions, reduction='none')
        ctrl_loss = ctrl_loss_raw.mean(1)
        steer_loss = ctrl_loss_raw[:, 0]
        speed_loss = ctrl_loss_raw[:, 1]

        loss_gt = (point_loss + self.hparams.command_coefficient * ctrl_loss)
        loss = loss_gt.mean()

        metrics = {
                'train_loss': loss.item(),

                'train_point': point_loss.mean().item(),
                'train_ctrl': ctrl_loss.mean().item(),
                'train_steer': steer_loss.mean().item(),
                'train_speed': speed_loss.mean().item(),
                }

        if batch_nb % 250 == 0:
            metrics['train_image'] = viz(batch, out, out_ctrl, target_cam, point_loss, ctrl_loss)

        self.logger.log_metrics(metrics, self.global_step)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        img, topdown, points, target, actions, meta = batch

        out, (target_cam, target_heatmap_cam) = self.forward(img, target)
        out_ctrl = self.controller(out)
        # out_ctrl_gt = self.controller(lbl_cam)

        point_loss = torch.nn.functional.l1_loss(out, points, reduction='none').mean((1, 2))
        ctrl_loss_raw = torch.nn.functional.l1_loss(out_ctrl, actions, reduction='none')
        ctrl_loss = ctrl_loss_raw.mean(1)
        steer_loss = ctrl_loss_raw[:, 0]
        speed_loss = ctrl_loss_raw[:, 1]

        loss_gt = (point_loss + self.hparams.command_coefficient * ctrl_loss)
        loss_gt_mean = loss_gt.mean()

        if batch_nb == 0:
            self.logger.log_metrics({
                'val_image': viz(batch, out, out_ctrl, target_cam, point_loss, ctrl_loss),
                }, self.global_step)

        return {
                'val_loss': loss_gt_mean.item(),

                'val_point': point_loss.mean().item(),
                'val_ctrl': ctrl_loss.mean().item(),
                'val_steer': steer_loss.mean().item(),
                'val_speed': speed_loss.mean().item(),
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
        return get_dataset(self.hparams.dataset_dir, True, self.hparams.batch_size, sample_by=self.hparams.sample_by, use_cpu=self.use_cpu)

    def val_dataloader(self):
        return get_dataset(self.hparams.dataset_dir, False, self.hparams.batch_size, sample_by=self.hparams.sample_by, use_cpu=self.use_cpu)

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

    model = TrafficImageModel(hparams)
    logger = WandbLogger(id=hparams.id, save_dir=str(hparams.save_dir), project='traffic_model')
    checkpoint_callback = ModelCheckpoint(hparams.save_dir, save_top_k=1)

    trainer = pl.Trainer(
            # gpus=-1, 
            gpus=None, 
            accelerator="cpu",
            max_epochs=hparams.max_epochs,
            resume_from_checkpoint=resume_from_checkpoint,
            logger=logger, checkpoint_callback=checkpoint_callback)

    trainer.fit(model)

    wandb.save(str(hparams.save_dir / '*.ckpt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--save_dir', type=pathlib.Path, default='checkpoints')
    parser.add_argument('--id', type=str, default=uuid.uuid4().hex)
    parser.add_argument('--cpu', action='store_true', default=False)

    # parser.add_argument('--teacher_path', type=pathlib.Path, required=True)

    # Model args.
    parser.add_argument('--heatmap_radius', type=int, default=5)
    parser.add_argument('--sample_by', type=str, choices=['none', 'even', 'speed', 'steer'], default='even')
    parser.add_argument('--command_coefficient', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=5.0)
    parser.add_argument('--hack', action='store_true', default=False)

    # Data args.
    parser.add_argument('--dataset_dir', type=pathlib.Path, required=True)
    parser.add_argument('--batch_size', type=int, default=16)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parsed = parser.parse_args()
    # parsed.teacher_path = parsed.teacher_path.resolve()
    parsed.save_dir = parsed.save_dir.resolve() / parsed.id
    parsed.save_dir.mkdir(parents=True, exist_ok=True)

    main(parsed)
