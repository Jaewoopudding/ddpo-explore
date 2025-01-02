import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import copy

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class StateEmbeddingNet(nn.Module):
    def __init__(self, in_ch=4, feature_dim=512):
        super(StateEmbeddingNet, self).__init__()

        feature_output = 64 * 4 * 4
        # 초기 Conv layer
        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim//2)
        )
        
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()


    def forward(self, next_obs):
        return self.network(next_obs)


class RND(nn.Module):
    def __init__(self, in_ch, feature_dim, intrinsic_coef=1.0):
        super().__init__()

        self.random_target_network = StateEmbeddingNet(in_ch = in_ch, feature_dim = feature_dim)
        self.predictor_network = StateEmbeddingNet(in_ch = in_ch, feature_dim = feature_dim)

        # self.device = next(self.predictor_network.parameters()).device
        # self.random_target_network = self.random_target_network.to(self.device)

        self.reward_scale = intrinsic_coef
    
    def forward(self, next_obs):

        next_obs_copy = next_obs.clone()

        random_obs = self.random_target_network(next_obs)
        predicted_obs = self.predictor_network(next_obs_copy)

        return random_obs, predicted_obs

    def compute_intrinsic_reward(self, next_obs):
        random_obs, predicted_obs = self.forward(next_obs)

        intrinsic_reward = torch.norm(predicted_obs.detach() - random_obs.detach(), dim=-1, p=2)
        intrinsic_reward *= self.reward_scale

        return intrinsic_reward

    def compute_loss(self, next_obs):
        random_obs, predicted_obs = self.forward(next_obs)
        rnd_loss = torch.norm(predicted_obs - random_obs.detach(), dim=-1, p=2)
        mean_rnd_loss = torch.mean(rnd_loss)
        return mean_rnd_loss



