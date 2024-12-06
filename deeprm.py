"""DeepRM Scheduler using REINFORCE algorithm."""
import env
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from schedule import Scheduler, Action

class PolicyNetwork(nn.Module):
    """Policy Network for REINFORCE algorithm."""
    
    def __init__(self, input_shape, output_shape):
        super(PolicyNetwork, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        
        # Calculate flattened size
        flatten_size = 16 * (input_shape[0]//2) * (input_shape[1]//2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(flatten_size, 256)
        self.fc2 = nn.Linear(256, output_shape)
        
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        # Add channel dimension and convert to float
        x = x.unsqueeze(1).float()
        
        # CNN forward pass
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Output probabilities
        return F.softmax(x, dim=1)

class ReinforceScheduler(Scheduler):
    """Scheduler using REINFORCE algorithm."""

    def __init__(self, environment, train=True):
        self.environment = environment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize network
        input_shape = (environment.summary().shape[0], environment.summary().shape[1])
        output_shape = environment.queue_size * len(environment.nodes) + 1
        self.policy = PolicyNetwork(input_shape, output_shape).to(self.device)
        
        if os.path.exists('__cache__/model/reinforce.pt'):
            self.policy.load_state_dict(torch.load('__cache__/model/reinforce.pt'))
        
        if train:
            self.train()

    def select_action(self, state):
        """Select action using policy network."""
        state = torch.from_numpy(state).to(self.device)
        probs = self.policy(state.unsqueeze(0))
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def finish_episode(self, optimizer, gamma=0.99, eps=1e-8):
        """Update policy network using REINFORCE algorithm."""
        R = 0
        policy_loss = []
        returns = []
        
        # Calculate discounted returns
        for r in self.policy.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        
        # Calculate policy loss
        for log_prob, R in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        # Update policy
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        
        # Clear memory
        self.policy.saved_log_probs = []
        self.policy.rewards = []

    def train(self, num_episodes=1000, lr=1e-3):
        """Train the policy network."""
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        running_reward = 0
        
        if not os.path.exists('__cache__/summary'):
            os.makedirs('__cache__/summary')
            
        for episode in range(num_episodes):
            # Reset environment
            self.environment, _ = env.load(load_scheduler=False)
            episode_reward = 0
            
            while not self.environment.terminated():
                # Get state and select action
                state = self.environment.summary()
                action = self.select_action(state)
                task_index, node_index = self._explain(action)
                
                # Execute action
                if task_index >= 0 and node_index >= 0 and task_index < len(self.environment.queue):
                    scheduled_task = self.environment.queue[task_index]
                    scheduled_node = self.environment.nodes[node_index]
                    if scheduled_node.schedule(scheduled_task):
                        del self.environment.queue[task_index]
                
                # Get reward and move to next state
                self.environment.timestep()
                reward = self.environment.reward()
                episode_reward += reward
                self.policy.rewards.append(reward)
            
            # Update running reward
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            
            # Update policy
            self.finish_episode(optimizer)
            
            # Log results
            if episode % 10 == 0:
                print(f'Episode {episode}, Running reward: {running_reward:.2f}')
            
            # Save model
            if episode % 100 == 0:
                if not os.path.exists('__cache__/model'):
                    os.makedirs('__cache__/model')
                torch.save(self.policy.state_dict(), '__cache__/model/reinforce.pt')

    def schedule(self):
        """Schedule tasks using trained policy."""
        actions = []
        
        while True:
            state = self.environment.summary()
            with torch.no_grad():
                action = self.select_action(state)
            
            task_index, node_index = self._explain(action)
            
            if task_index < 0 or node_index < 0:
                break
                
            if task_index >= len(self.environment.queue):
                break
                
            scheduled_task = self.environment.queue[task_index]
            scheduled_node = self.environment.nodes[node_index]
            
            if not scheduled_node.schedule(scheduled_task):
                break
                
            actions.append(Action(scheduled_task, scheduled_node))
            del self.environment.queue[task_index]
        
        self.environment.timestep()
        return actions

    def _explain(self, action_index):
        """Convert action index to task and node indices."""
        task_limit = self.environment.queue_size
        node_limit = len(self.environment.nodes)
        
        if action_index == task_limit * node_limit:
            return -1, -1
            
        task_index = action_index % task_limit
        node_index = action_index // task_limit
        
        return task_index, node_index