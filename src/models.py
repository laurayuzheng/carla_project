import torch
import torch.nn.functional as F
import stable_baselines3 as sb3
from stable_baselines3.common.preprocessing import preprocess_obs
from gym import spaces

from torchvision.models.segmentation import deeplabv3_resnet50

# action, _states = model.predict(obs, deterministic=True)
class AccelAgentNetwork(torch.nn.Module):
  def __init__(self, extractor, mlp_policy, action_net, value_net):
      super(AccelAgentNetwork, self).__init__()
      self.extractor = extractor
      self.mlp = mlp_policy
      self.action_net = action_net
      self.value_net = value_net

  def forward(self, observation):
    #   observation = preprocess_obs(obs=observation,
    #                             observation_space=spaces.Dict,
    #                             normalize_images=True)
    # preprocessed_obs = {}
    # for key, _obs in observation.items():
    #     preprocessed_obs[key] = preprocess_obs(_obs, spaces.Box, normalize_images=True)
    features = self.extractor(observation)
    action_hidden, value_hidden = self.mlp(features)
    return self.action_net(action_hidden), self.value_net(value_hidden)

class RawController(torch.nn.Module):
    def __init__(self, n_input=4, k=32, n_classes=1):
        super().__init__()

        self.layers = torch.nn.Sequential(
                torch.nn.BatchNorm1d(n_input * 2),
                torch.nn.Linear(n_input * 2, k), torch.nn.ReLU(),

                torch.nn.BatchNorm1d(k),
                torch.nn.Linear(k, k), torch.nn.ReLU(),

                torch.nn.BatchNorm1d(k),
                torch.nn.Linear(k, n_classes))

    def forward(self, x):
        return self.layers(torch.flatten(x, 1))


class SpatialSoftmax(torch.nn.Module):
    def forward(self, logit, temperature):
        """
        Assumes logits is size (n, c, h, w)
        """
        flat = logit.view(logit.shape[:-2] + (-1,))
        weights = F.softmax(flat / temperature, dim=-1).view_as(logit)

        x = (weights.sum(-2) * torch.linspace(-1, 1, logit.shape[-1]).type_as(logit)).sum(-1)
        y = (weights.sum(-1) * torch.linspace(-1, 1, logit.shape[-2]).type_as(logit)).sum(-1)

        return torch.stack((x, y), -1)


class SegmentationModel(torch.nn.Module):
    def __init__(self, input_channels=3, n_steps=4, batch_norm=True, hack=False, temperature=1.0):
        super().__init__()

        self.temperature = temperature
        self.hack = hack

        self.norm = torch.nn.BatchNorm2d(input_channels) if batch_norm else lambda x: x
        self.network = deeplabv3_resnet50(pretrained=False, num_classes=n_steps)
        self.extract = SpatialSoftmax()

        old = self.network.backbone.conv1
        self.network.backbone.conv1 = torch.nn.Conv2d(
                input_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=old.bias)

    def forward(self, input, heatmap=False):
        if self.hack:
            input = torch.nn.functional.interpolate(input, scale_factor=0.5, mode='bilinear')

        x = self.norm(input)
        x = self.network(x)['out']

        if self.hack:
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode='bilinear')

        y = self.extract(x, self.temperature)

        if heatmap:
            return y, x

        return y
