import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
import os
from schedule import Scheduler, Action
import env as envi

class ResourceCNN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(ResourceCNN, self).__init__()
        
        # Input shape should be (batch, 1, H, W)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Calculate flattened size
        self._get_conv_output(input_shape)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self._features_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_shape)
        )
        
    def _get_conv_output(self, shape):
        x = torch.zeros(1, 1, *shape)
        x = self.conv_layers(x)
        self._features_size = int(np.prod(x.shape))
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

class DQN:
    def __init__(self, input_shape, output_shape, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gamma = 0.99
        self.lr = 1e-3
        self.batch_size = 32
        self.memory = deque(maxlen=10000)
        
        self.policy_net = ResourceCNN(input_shape, output_shape).to(self.device)
        self.target_net = ResourceCNN(input_shape, output_shape).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, epsilon):
        if random.random() < epsilon:
            action = random.randrange(self.policy_net.fc_layers[-1].out_features)
            return action
            
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
            
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
        
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save(self, path='__cache__/model/deeprm.pth'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)
        
    def load(self, path='__cache__/model/deeprm.pth'):
        if os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            return True
        return False

class ReinforceScheduler(Scheduler):
    def __init__(self, environment, train=True):
        input_shape = environment.summary().shape[:2]  # Height, Width
        n_actions = environment.queue_size * len(environment.nodes) + 1
        
        self.env = environment
        self.dqn = DQN(input_shape, n_actions)
        
        if train:
            self.train()
            
    def train(self, episodes=1000):
        epsilon = 1.0
        min_epsilon = 0.01
        epsilon_decay = 0.995
        for episode in range(episodes):
            self.env, _ = envi.load(load_scheduler=False)
            episode_reward = 0
            
            while not self.env.terminated():
                state = self.env.summary()
                action = self.dqn.act(state, epsilon)
                
                task_index, node_index = self._explain(action)
                # invalid action, proceed to the next timestep
                if task_index < 0 or node_index < 0:
                    self.env.timestep()
                    continue

                if task_index >= 0 and node_index >= 0:
                    if task_index < len(self.env.queue):
                        task = self.env.queue[task_index]
                        node = self.env.nodes[node_index]
                        scheduled = node.schedule(task)
                        if not scheduled:
                            self.env.timestep()
                            continue
                        else:
                            del self.env.queue[task_index]
                
                next_state = self.env.summary()
                reward = self.env.reward()
                done = self.env.terminated()
                
                self.dqn.remember(state, action, reward, next_state, done)
                loss = self.dqn.replay()
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
                    
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            
            if episode % 10 == 0:
                self.dqn.update_target()
                self.dqn.save()
                
            print(f"Episode {episode}: Reward={episode_reward:.2f}, Epsilon={epsilon:.2f}")
            
    def schedule(self):
        actions = []
        
        while True:
            state = self.env.summary()
            action = self.dqn.act(state, 0.0)  # No exploration during inference
            
            task_index, node_index = self._explain(action)
            if task_index < 0 or node_index < 0:
                break
                
            if task_index >= len(self.env.queue):
                break
                
            task = self.env.queue[task_index]
            node = self.env.nodes[node_index]
            
            if not node.schedule(task):
                break
                
            del self.env.queue[task_index]
            actions.append(Action(task, node))
            
        self.env.timestep()
        return actions
        
    def _explain(self, action_index):
        task_limit = self.env.queue_size
        node_limit = len(self.env.nodes)
        
        if action_index == task_limit * node_limit:
            return -1, -1
            
        task_index = action_index % task_limit
        node_index = action_index // task_limit
        
        if task_index >= len(self.env.queue):
            return -1, -1
            
        return task_index, node_index