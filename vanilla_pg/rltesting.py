import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
import gym
from collections import deque
import matplotlib.pyplot as plt 
import numpy as np

Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorNet(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.fc1 = nn.Linear(obs_space, 32)
        self.fc2 = nn.Linear(32, action_space)        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PGAgent():
    def __init__(self):
        # These values are taken from the env's state and action sizes
        self.actor_net = ActorNet(4,2)
        self.actor_net.to(device=Device)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr = 0.001)
        self.env = gym.make('CartPole-v1')
        self.render = False

        # These two will be used during the training of the policy
        self.batch_size = 500
        self.epochs = 3

        # These will be used for the agent to play using the current policy
        self.max_eps = 15
        # Max steps in mountaincar ex is 200
        self.max_steps = 201
        # Should be more than max_eps * max_steps
        self.deque_len = 2500
        # documenting the stats
        self.avg_over = 5 # episodes
        self.stats = {'episode': 0, 'ep_rew': []}

        

    # This returns a categorical torch object, which makes it easier to calculate log_prob, prob and sampling from them
    def get_logits(self, state):
        logits = self.actor_net(state)
        return Categorical(logits=logits)

    def calc_log_prob(self, states, actions, rewards):
        assert (torch.is_tensor(states) and torch.is_tensor(actions) and torch.is_tensor(rewards)),\
             "states and actions are not in the right format"

        # The negative sign is for gradient ascent
        loss = -(self.get_logits(states).log_prob(actions))*rewards
        return loss.mean()

    def get_action(self, state):
        # Sample in categorical finds porbability first and then samples values according to that prob
        return self.get_logits(state).sample().item()


    def reward_to_go(self, traj_dones, traj_rewards):
        # This gives the reward to go for each transition in the batch
        rew_to_go_list = []
        rew_sum = 0
        for rew, done in zip(reversed(traj_rewards), reversed(traj_dones)):
            if done:
                rew_sum = rew
                rew_to_go_list.append(rew_sum)
            else:
                rew_sum = rew + rew_sum
                rew_to_go_list.append(rew_sum)

        rew_to_go_list = reversed(rew_to_go_list)
        return list(rew_to_go_list)

    
    def train(self, traj_obs, traj_rewards, traj_dones, traj_actions):

        # Making sure traj lists are of the same size
        assert (len(traj_dones)==len(traj_obs)==len(traj_rewards)), "Size of traj lists don't match"
        print('size of trajectories: ', len(traj_obs))

        traj_rewards_to_go = self.reward_to_go(traj_dones, traj_rewards)
        
        for epoch in range(self.epochs):
            for batch_offs in range(0, len(traj_obs), self.batch_size):
                
                batch_obs = traj_obs[batch_offs:batch_offs + self.batch_size]
                batch_rew = traj_rewards_to_go[batch_offs:batch_offs + self.batch_size]
                batch_dones = traj_dones[batch_offs:batch_offs + self.batch_size]
                batch_actions = traj_actions[batch_offs:batch_offs + self.batch_size]

                batch_obs = torch.as_tensor(batch_obs, dtype=torch.float32).to(device=Device)
                batch_actions = torch.as_tensor(batch_actions, dtype=torch.float32).to(device=Device)
                batch_rew = torch.as_tensor(batch_rew, dtype=torch.float32).to(Device)

                # print('while training: ', batch_obs.dtype)
                # print('shape: ', batch_obs.shape)
                self.actor_optim.zero_grad()
                policy_loss = self.calc_log_prob(batch_obs, batch_actions, batch_rew)
                # print('leaf: ', policy_loss.is_leaf)
                policy_loss.backward()
                # print('gradient of actor: ', self.actor_net.fc1.weight.grad)
                self.actor_optim.step()
        
        print('Policy updated ')
        

    # The agent will play self.max_eps episodes using the current policy, and train on that data
    def play(self, rendering):
        # This consists of transitions from all episodes played
        traj_obs = []
        traj_actions = []
        traj_rewards = []
        traj_dones = []
        traj_logprobs = []

        saved_transitions = 0
        for ep in range(self.max_eps):
            obs = self.env.reset()
            ep_reward = 0

            for step in range(self.max_steps):
                
                if rendering==True:
                    self.env.render()

                traj_obs.append(obs)

                with torch.no_grad():

                    obs = torch.from_numpy(obs).float().to(device=Device)
                    action = self.get_action(obs)
                    obs, rew, done, info = self.env.step(action)
                    ep_reward += rew

                traj_actions.append(action)
                traj_rewards.append(rew)

                saved_transitions += 1

                if done:
                    # We will not save the last observation, since it is essentially a dead state
                    # This will result in having the same length of obs, action, reward and dones deque
                    traj_dones.append(done)
                    self.stats['ep_rew'].append(ep_reward)
                    self.stats['episode'] += 1
                    break

                else:
                    traj_dones.append(done)
            

        print(saved_transitions, ' transitions saved')
        self.train(traj_obs, traj_rewards, traj_dones, traj_actions)

    def run(self, policy_updates = 65, show_renders_every = 20):
        for i in range(policy_updates):
            if i%show_renders_every==0:
                vanilla_pg.play(rendering=True)
            else:
                vanilla_pg.play(rendering=False)

    def plot_rewards(self, avg_over=10):
        graph_x = np.arange(vanilla_pg.stats['episode'])[::avg_over]
        graph_y = self.stats['ep_rew'][::avg_over]
        plt.plot(graph_x, graph_y)
        plt.show()



vanilla_pg = PGAgent()
vanilla_pg.run(policy_updates = 200, show_renders_every = 20)

vanilla_pg.plot_rewards()

