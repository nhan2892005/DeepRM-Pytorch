import datetime
import os
from abc import ABC, abstractmethod

import numpy as np

import env
from schedule import Scheduler
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.autograd import Variable

class TorchCNNModel(nn.Module):
    """CNN Model."""
    def __init__(self, input_shape, output_shape):
        super(TorchCNNModel, self).__init__()
        self.model_path = '__cache__/model/deeprm.pth'
        self.conv1 = nn.Conv2d(kernel_size=(3, 3), stride=1, padding='same', out_channels=16, in_channels=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self._compute_flatten_size(input_shape), 256)
        self.fc2 = nn.Linear(256, output_shape)

    def forward(self, x):
        """Forward pass."""
        x = func.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self):
        """Save the model to a file."""
        if not os.path.exists('__cache__/model'):
            os.makedirs('__cache__/model')
        torch.save(self.state_dict(), self.model_path)

    def load(self):
        """Load the model from a file."""
        self.load_state_dict(torch.load(self.model_path))
    
    def _compute_flatten_size(self, input_shape):
        """Compute the size of the flattened features after convolution and pooling."""
        # Create a dummy input to calculate the flattened size
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)  # Batch size 1
            x = func.relu(self.conv1(x))  # Apply convolution
            x = self.pool(x)  # Apply pooling
            x = self.dropout(x)  # Apply dropout
            x = self.flatten(x)  # Flatten the output
            return x.shape[1]  # Return the number of features after flattening
        
class TorchDQN(object):
    """DQN Implementation."""

    def __init__(self, input_shape, output_shape):
        self.lr = 0.01
        self.gamma = 0.99
        self.batch_size = 32
        self.min_experiences = 100
        self.max_experiences = 10000
        self.criterion = nn.MSELoss()
        self.num_actions = output_shape
        self.model = TorchCNNModel(input_shape, output_shape)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}

    def predict(self, input_data):
        """Predict Q value given state."""
        input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(1)
        return self.model(input_data)

    def train(self, dqn_target):
        """Train DQN."""
        if len(self.experience['s']) < self.min_experiences:
            return
        
        # samples
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])

        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        # use target model to calculate actual values
        values_next = torch.max(dqn_target.predict(states_next), dim=1)[0]
        actual_values = torch.where(dones, rewards, rewards+self.gamma*values_next)

        # Compute predicted Q values and loss
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        actual_values_tensor = torch.tensor(actual_values, dtype=torch.float32)

        predicted_values = self.predict(states_tensor)
        predicted_values = predicted_values.gather(1, actions_tensor.unsqueeze(1))  # Select the predicted Q values for actions

        loss = func.mse_loss(predicted_values.squeeze(1), actual_values_tensor)

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, states, epsilon):
        """Predict action according to state (espilon for exploration)."""
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            states_tensor = torch.tensor(states, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            return torch.argmax(self.model(states_tensor)).item()

    def add_experience(self, exp):
        """Add experience into replay buffer."""
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, dqn_src):
        """Copy weights between models."""
        for param1, param2 in zip(self.model.parameters(), dqn_src.model.parameters()):
            param1.data.copy_(param2.data)

    def save_weights(self):
        """Save model."""
        self.model.save()

class TorchDeepRMTrainer(object):
    """DeepRM Trainer."""

    def __init__(self, environment):
        self.episodes = 10000
        self.copy_steps = 32
        self.save_steps = 32
        self.epsilon = 0.99
        self.decay = 0.99
        self.min_epsilon = 0.1
        input_shape = (1, environment.summary().shape[0], environment.summary().shape[1])
        output_shape = environment.queue_size * len(environment.nodes) + 1
        self.dqn_train = TorchDQN(input_shape, output_shape)
        self.dqn_target = TorchDQN(input_shape, output_shape)
        self.total_rewards = np.empty(self.episodes)
        self.environment = environment
        if not os.path.exists('__cache__/summary'):
            os.makedirs('__cache__/summary')
        self.summary_writer = SummaryWriter(log_dir='__cache__/summary/dqn-{0}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    def train(self):
        """Train process."""
        for i in range(self.episodes):
            self.epsilon = max(self.min_epsilon, self.epsilon*self.decay)
            self.total_rewards[i] = self.train_episode()
            self.summary_writer.add_scalar('Episode Job Slowdown', -self.total_rewards[i], i)
            print(f'Episode {i} Job Slowdown {-self.total_rewards[i]}')

    def train_episode(self):
        """Train process of single episode."""
        rewards = 0
        step = 0
        self.environment, _ = env.load(load_scheduler=False)
        while not self.environment.terminated():
            # observe state and predict action
            observation = self.environment.summary()
            action_index = self.dqn_train.get_action(observation, self.epsilon)
            task_index, node_index = self._explain(action_index)

            # invalid action, proceed to the next timestep
            if task_index < 0 or node_index < 0:
                self.environment.timestep()
                continue
            scheduled_task = self.environment.queue[task_index]
            scheduled_node = self.environment.nodes[node_index]
            scheduled = scheduled_node.schedule(scheduled_task)
            if not scheduled:
                self.environment.timestep()
                continue

            # apply action, calculate reward and train model
            del self.environment.queue[task_index]
            prev_observation = observation
            reward = self.environment.reward()
            observation = self.environment.summary()
            rewards = rewards + reward
            exp = {'s': prev_observation, 'a': action_index, 'r': reward, 's2': observation, 'done': self.environment.terminated()}
            self.dqn_train.add_experience(exp)
            self.dqn_train.train(self.dqn_target)

            step += 1
            # copy weights from train model to target model periodically
            if step % self.copy_steps == 0:
                self.dqn_target.copy_weights(self.dqn_train)
            # save model periodically
            if step % self.save_steps == 0:
                self.dqn_target.save_weights()

        return rewards

    def _explain(self, action_index):
        """Explain action."""
        task_limit = self.environment.queue_size
        node_limit = len(self.environment.nodes)
        if action_index == task_limit*node_limit:
            task_index = -1
            node_index = -1
        else:
            task_index = action_index % task_limit
            node_index = action_index // task_limit
        if task_index >= len(self.environment.queue):
            task_index = -1
            node_index = -1
        return (task_index, node_index)

class DeepRMSchedulerTorch(Scheduler):
    """DeepRM scheduler."""

    def __init__(self, environment, train=True):
        if train:
            TorchDeepRMTrainer(environment).train()
        input_shape = (environment.summary().shape[0], environment.summary().shape[1], 1)
        output_shape = environment.queue_size * len(environment.nodes) + 1
        self.dqn_train = DQN(input_shape, output_shape)
        self.environment = environment

    def schedule(self):
        """Schedule with trained model."""
        actions = []

        # apply actions until there's an invalid one
        while True:
            observation = self.environment.summary()
            action_index = self.dqn_train.get_action(observation, 0)
            task_index, node_index = self._explain(action_index)
            if task_index < 0 or node_index < 0:
               break
            scheduled_task = self.environment.queue[task_index]
            scheduled_node = self.environment.nodes[node_index]
            scheduled = scheduled_node.schedule(scheduled_task)
            if not scheduled:
                break
            del self.environment.queue[task_index]
            actions.append(Action(scheduled_task, scheduled_node))

        # proceed to the next timestep
        self.environment.timestep()

        return actions

    def _explain(self, action_index):
        """Explain action."""
        task_limit = self.environment.queue_size
        node_limit = len(self.environment.nodes)
        if action_index == task_limit*node_limit:
            task_index = -1
            node_index = -1
        else:
            task_index = action_index % task_limit
            node_index = action_index // task_limit
        if task_index >= len(self.environment.queue):
            task_index = -1
            node_index = -1
        return (task_index, node_index)
