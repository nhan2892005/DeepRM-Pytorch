"""DeepRM Scheduler using REINFORCE algorithm."""
import env
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from schedule import Scheduler, Action
from collections import deque
import random

class PolicyNetwork(nn.Module):
    """Policy Network for REINFORCE algorithm."""
    
    def __init__(self, input_shape, output_shape):
        super(PolicyNetwork, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        
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
        
        # Add batch size and memory management
        self.batch_size = 32
        self.max_memory = 10000
        self.lr = 3e-4
        self.eps = 1e-8
        self.max_episodes = 10000
        self.early_stop_patience = 1000
        self.max_steps = 10000

        # Initialize network
        input_shape = (environment.summary().shape[0], environment.summary().shape[1])
        output_shape = environment.queue_size * len(environment.nodes) + 1
        self.policy = PolicyNetwork(input_shape, output_shape).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        if os.path.exists('__cache__/model/reinforce.pth'):
            self.policy.load_state_dict(torch.load('__cache__/model/reinforce.pth'))
        
        if train:
            self.train()

    def select_action(self, state):
        """Select action using policy network with gradient tracking."""
        try:
            state = torch.FloatTensor(state).to(self.device)
            # Enable gradient tracking
            with torch.enable_grad():
                # Get action probabilities
                probs = self.policy(state.unsqueeze(0))
                
                # Add exploration noise
                noise = torch.rand_like(probs) * 0.1
                probs = F.softmax(probs + noise, dim=-1)
            
                m = Categorical(probs)
                action = m.sample()
                # Save log probability with gradient tracking
                self.policy.saved_log_probs.append(m.log_prob(action))
                
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
                return action.item()
            
        except Exception as e:
            print(f"Error in select_action: {e}")
            return 0  # Return safe default
            
    def finish_episode(self, optimizer, gamma=0.99, eps=1e-8):
        """Update policy network using REINFORCE with proper gradient handling."""
        try:
            # Check if we have any experiences to learn from
            if len(self.policy.saved_log_probs) == 0 or len(self.policy.rewards) == 0:
                return
                
            # Convert rewards to tensor and calculate returns
            rewards = torch.FloatTensor(self.policy.rewards).to(self.device)
            returns = []
            R = 0
            
            # Calculate discounted returns
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
                
            returns = torch.FloatTensor(returns).to(self.device)
            
            # Normalize returns if we have more than one return
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + eps)
            
            # Calculate policy loss
            policy_loss = 0
            for log_prob, R in zip(self.policy.saved_log_probs, returns):
                policy_loss = policy_loss - log_prob * R  # Negative for gradient ascent
                
            # Only backpropagate if we have a valid loss
            # if isinstance(policy_loss, torch.Tensor):
                # Optimize
            optimizer.zero_grad()
            policy_loss.backward()
                
                # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                
            optimizer.step()
            
            # Clear memory
            self.policy.saved_log_probs = []
            self.policy.rewards = []
            
            # Clear CUDA cache if using GPU
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error in finish_episode: {e}")
            self.policy.saved_log_probs = []
            self.policy.rewards = []

    def _execute_action(self, task_index, node_index):
        """Execute the selected action in the environment.
        
        Args:
            task_index (int): Index of task to schedule
            node_index (int): Index of node to schedule on
        
        Returns:
            bool: True if action was executed successfully, False otherwise
        """
        try:
            # Check for invalid indices
            if task_index < 0 or node_index < 0:
                return False
                
            if task_index >= len(self.environment.queue):
                return False
                
            if node_index >= len(self.environment.nodes):
                return False
            
            # Get task and node
            scheduled_task = self.environment.queue[task_index]
            scheduled_node = self.environment.nodes[node_index]
            
            # Try to schedule
            if scheduled_node.schedule(scheduled_task):
                del self.environment.queue[task_index]
                return True
                
            return False
            
        except Exception as e:
            print(f"Error executing action: {e}")
            return False

    def train(self, num_episodes=10000, lr=1e-3):
        """Train with memory optimization and early stopping."""
        if num_episodes is None:
            num_episodes = self.max_episodes
            
        running_reward = 0
        best_reward = float(-33.54)
        patience_counter = 0
        
        # try:
        for episode in range(num_episodes):
                self.environment, _ = env.load(load_scheduler=False)
                episode_reward = 0
                state = self.environment.summary()
                
                for step in range(self.max_steps):
                # while not self.environment.terminated():
                    action = self.select_action(state)
                    
                    task_index, node_index = self._explain(action)
                    if self._execute_action(task_index, node_index):
                        reward = self.environment.reward()
                        reward = reward * 0.1
                        episode_reward += reward
                        self.policy.rewards.append(reward)
                    
                    self.environment.timestep()
                    state = self.environment.summary()
                    if self.environment.terminated():
                        self._save_model()
                        break

                    if len(self.policy.saved_log_probs) >= self.batch_size:
                        self.finish_episode(self.optimizer)

                running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    patience_counter = 0
                    self._save_model()
                    print(f"New best reward: {best_reward:.2f}")
                    patience_counter += 1
                else:
                    if patience_counter >= self.early_stop_patience:
                        print(f"Early stopping at episode {episode}")
                        break

                if episode % 10 == 0:
                    print(f'Episode {episode}, Reward: {episode_reward:.2f}')
                
                
                    
        #except Exception as e:
            #print(f"Training error: {e}")
            # self._save_model()  # Save current progress

    def _save_model(self):
        """Safe model saving."""
        try:
            if not os.path.exists('__cache__/model'):
                os.makedirs('__cache__/model')
            torch.save(self.policy.state_dict(), '__cache__/model/reinforce.pth')
        except Exception as e:
            print(f"Error saving model: {e}")

    def schedule(self):
        """Schedule tasks using trained policy."""
        actions = []
        
        while True:
            state = self.environment.summary()
            with torch.no_grad():
                action = self.select_action(state)
            
            task_index, node_index = self._explain(action)
            
            print(task_index, node_index)
            print(self.environment.queue)
            if task_index < 0 or node_index < 0:
                break
                
            if task_index >= len(self.environment.queue):
                if len(self.environment.queue) == 0:
                    break
                task_index = 0
                
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