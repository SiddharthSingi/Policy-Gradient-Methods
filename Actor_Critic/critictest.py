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

class PGAgent():
    def __init__(self):
        # These values are taken from the env's state and action sizes
        self.actor_net = ActorNet(4,2)
        self.actor_net.to(device=Device)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr = 0.001)

        self.critic_net = CriticNet(4)
        self.critic_net.to(Device)
        self.crtitc_net_optim = torch.optim.Adam(self.critic_net.parameters(), lr = 0.001)

        # Storing all the values used to calculate the model losses
        self.traj_obs = []
        self.traj_actions = []
        self.traj_rewards = []
        self.traj_dones = []
        self.traj_logprobs = []
        self.traj_logits = []
        self.traj_state_values = []

        # Discount factor
        self.gamma = 0.99
        # Bias Variance tradeoff (higher value results in high variance, low bias)
        self.gae_lambda = 0.80


        self.env = gym.make('CartPole-v1')
        self.render = False

        # These two will be used during the training of the policy
        self.batch_size = 500
        self.epochs = 3

        # These will be used for the agent to play using the current policy
        self.max_eps = 15
        # Max steps in mountaincar ex is 200
        self.max_steps = 400
        # Should be more than max_eps * max_steps
        self.deque_len = 2500
        # documenting the stats
        self.avg_over = 5 # episodes
        self.stats = {'episode': 0, 'ep_rew': []}

    def clear_lists(self):
        self.traj_obs = []
        self.traj_actions = []
        self.traj_rewards = []
        self.traj_dones = []
        self.traj_logprobs = []
        self.traj_logits = []
        self.traj_state_values = []

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
        
        # Finding the logits and state value using the actor and critic net
        logits = self.get_logits(state)
        action = logits.sample()
        state_value = self.critic_net(state)

        # Saving the log_prob and state values for calculating the loss for optimizers
        self.traj_logprobs.append(logits.log_prob(action))
        self.traj_logits.append(logits)
        self.traj_state_values.append(state_value)

        # Sample in categorical finds probability first and then samples values according to that prob
        return action.item()


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

    def gae_loss(self):
        # print('len of traj: ', len(self.traj_logprobs))
        # print('type: ', type(self.traj_logprobs[0]))
        # print('is leaf: ', self.traj_logprobs[0].is_leaf)
        # print('traj lens: ', len(self.traj_obs), len(self.traj_logprobs), len(self.traj_actions), len(self.traj_rewards))
        # logit = self.traj_logits[10]
        # print(type(logit))
        # print('Logits: ', logit.probs)
        # print('entropy: ', logit.entropy())
        # print(logit.log_prob(torch.tensor(0).to(Device)), logit.log_prob(torch.tensor(1).to(Device)))


        policy_loss = []
        critic_loss = []
        entropy_loss = []

        # Important to know that rewards and actions are disconnected from the graph.
        # They are converted from int values and hence gradients do not flow past them
        self.traj_actions = torch.tensor(self.traj_actions).to(Device)


        for i, val, next_val, logit, action, reward, done in \
        zip(range(len(self.traj_dones)), \
        reversed(self.traj_state_values), \
        reversed(self.traj_state_values[1:] + [None]), \
        reversed(self.traj_logits), \
        reversed(self.traj_actions), \
        reversed(self.traj_rewards), \
        reversed(self.traj_dones)):

            if done or i==0:
                delta = reward - val
                last_gae = delta
            else:
                delta = reward + self.gamma*next_val - val
                last_gae = delta + self.gamma*self.gae_lambda*last_gae
            

            entropy_loss.append(-logit.entropy()) # Negative of entropy in order to perform gradient ascent
            policy_loss.append(-last_gae*(logit.log_prob(action))) # Negative of policy loss in order to perform gradient ascent
            critic_loss.append(F.mse_loss(val, last_gae + val))

        # reset gradients
        self.actor_optim.zero_grad()
        self.crtitc_net_optim.zero_grad()

        # summing up policy and critic losses
        loss = torch.stack(policy_loss).sum() + torch.stack(critic_loss).sum() + torch.stack(entropy_loss).sum()
        
        # backpropogating the losses
        loss.backward()

        self.crtitc_net_optim.step()
        self.actor_optim.step()

    
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
        
        self.clear_lists()
        saved_transitions = 0
        for ep in range(self.max_eps):
            obs = self.env.reset()
            ep_reward = 0

            for step in range(self.max_steps):
                
                if rendering==True:
                    self.env.render()

                self.traj_obs.append(obs)
                obs = torch.from_numpy(obs).float().to(device=Device)
                
                # get_action() will find the action to take and also append the TENSOR of
                # logprob and state value and action
                # to the traj_log_probs, traj_state_values and traj_actions respectively
                action = self.get_action(obs)
                
                obs, rew, done, info = self.env.step(action)
                ep_reward += rew

                self.traj_actions.append(action)
                self.traj_rewards.append(rew)

                saved_transitions += 1

                if done:
                    # We will not save the last observation, since it is essentially a dead state
                    # This will result in having the same length of obs, action, reward and dones deque
                    self.traj_dones.append(done)
                    self.stats['ep_rew'].append(ep_reward)
                    self.stats['episode'] += 1
                    break

                else:
                    self.traj_dones.append(done)
            print(f" {ep} episodes over.", end='\r')
            
        self.gae_loss()


    def run(self, policy_updates = 65, show_renders_every = 20, renders = True):
        for i in range(policy_updates):
            if i%show_renders_every==0:
                vanilla_pg.play(rendering=renders)
            else:
                vanilla_pg.play(rendering=False)
            print(f" Policy updated {i} times")

    def plot_rewards(self, avg_over=10):
        graph_x = np.arange(vanilla_pg.stats['episode'])[::avg_over]
        graph_y = self.stats['ep_rew'][::avg_over]
        plt.plot(graph_x, graph_y)
        plt.show()



vanilla_pg = PGAgent()
# vanilla_pg.play(rendering=False)
vanilla_pg.run(policy_updates = 50, show_renders_every = 20, renders=False)
# Comment out when you want to see plotted rewards over episodes
vanilla_pg.plot_rewards()

