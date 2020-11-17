## Currently this file is only being used by:
## gail.py, model_expert_traj.py

import torch
from torch import nn
import torch.nn.functional as F


class CriticNet(nn.Module):
    def __init__(self, obs_space):
        super().__init__()
        self.fc1 = nn.Linear(obs_space, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x