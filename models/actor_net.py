## Currently this file is only being used by:
## gail.py, model_expert_traj.py

import torch
from torch import nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.fc1 = nn.Linear(obs_space, 32)
        self.fc2 = nn.Linear(32, action_space)        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x