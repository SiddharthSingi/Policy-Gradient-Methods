## Currently this file is only being used by:
## gail.py, model_expert_traj.py

import torch
from torch import nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self, obs_space, action_space):
        super().__init__()
        self.fc1 = nn.Linear(obs_space+action_space, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.sigmoid(x)

        return x
