import gymnasium as gym
import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e7)  # Replay memory size
BATCH_SIZE = 128  # Number of experiences to sample from memory
GAMMA = 0.99  # Discount factor
TAU = 1e-3  # Soft update parameter for updating fixed q network
LR = 1e-4  # Q Network learning rate
UPDATE_EVERY = 6  # How often to update Q network


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        """
        Replay memory allow agent to record experiences and learn from them

        Parametes
        ---------
        buffer_size (int): maximum size of internal memory
        batch_size (int): sample size from experience
        seed (int): random seed
        """
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    def add(self, state, action, reward, next_state, done):
        """Add experience"""
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        """
        Sample randomly and return (state, action, reward, next_state, done) tuple as torch tensors
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert to torch tensors
        states = torch.from_numpy(
            np.vstack(
                [
                    experience.state
                    for experience in experiences
                    if experience is not None
                ]
            )
        ).float()
        actions = torch.from_numpy(
            np.vstack(
                [
                    experience.action
                    for experience in experiences
                    if experience is not None
                ]
            )
        ).long()
        rewards = torch.from_numpy(
            np.vstack(
                [
                    experience.reward
                    for experience in experiences
                    if experience is not None
                ]
            )
        ).float()
        next_states = torch.from_numpy(
            np.vstack(
                [
                    experience.next_state
                    for experience in experiences
                    if experience is not None
                ]
            )
        ).float()

        # Convert done from boolean to int
        dones = torch.from_numpy(
            np.vstack(
                [
                    experience.done
                    for experience in experiences
                    if experience is not None
                ]
            ).astype(np.uint8)
        ).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """
        Build a fully connected neural network

        Parameters
        ----------
        state_size (int): State dimension
        action_size (int): Action dimension
        seed (int): random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        """Forward pass"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Agent:
    """
    Reinforcement learning agent in an environment. Uses q-network to learn.
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.action_size = action_space.n
        self.action = 0
        self.iterations = 1

        state_size = observation_space.shape[0]
        action_size = action_space.n
        seed = 0

        self.q_network = QNetwork(state_size, action_size, seed)
        self.fixed_network = QNetwork(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.q_network.parameters())

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.timestep = 0

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Choose a action from the action space based on the q network or by random choice.
        """
        state = observation
        rnd = random.random()
        eps = 0.2 + 1 / np.sqrt(self.iterations)

        if rnd < eps:
            action = np.random.randint(self.action_size)
            self.action = action
            state = torch.from_numpy(state).float().unsqueeze(0)
            self.observation_space = state
            return action
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            self.observation_space = state
            # set the network into evaluation mode
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            # Back to training mode
            self.q_network.train()
            action = np.argmax(action_values.cpu().data.numpy())
            self.action = action
            return action

    def checkpoint(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def learning(self, experiences):
        """
        Learn from experience by training the q_network

        Parameters
        ----------
        experiences (array_like): List of experiences sampled from agent's memory
        """
        states, actions, rewards, next_states, dones = experiences
        # Get the action with max Q value
        action_values = self.fixed_network(next_states).detach()
        # Notes
        # tensor.max(1)[0] returns the values, tensor.max(1)[1] will return indices
        # unsqueeze operation --> np.reshape
        # Here, we make it from torch.Size([64]) -> torch.Size([64, 1])
        max_action_values = action_values.max(1)[0].unsqueeze(1)

        # If done just use reward, else update Q_target with discounted action values
        Q_target = rewards + (GAMMA * max_action_values * (1 - dones))
        Q_expected = self.q_network(states).gather(1, actions)

        # Calculate loss
        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()
        # backward pass
        loss.backward()
        # update weights
        self.optimizer.step()

        # Update fixed weights
        self.update_fixed_network(self.q_network, self.fixed_network)

    def update_fixed_network(self, q_network, fixed_network):
        """
        Update fixed network by copying weights from Q network using TAU param

        Parameters
        ----------
        q_network (PyTorch model): Q network
        fixed_network (PyTorch model): Fixed target network
        """
        for source_parameters, target_parameters in zip(
            q_network.parameters(), fixed_network.parameters()
        ):
            target_parameters.data.copy_(
                TAU * source_parameters.data + (1.0 - TAU) * target_parameters.data
            )

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Learn from experience by training the q_network
        """

        state = self.observation_space
        next_state = observation
        done = terminated or truncated
        action = self.action

        self.memory.add(state, action, reward, next_state, done)
        if done:
            self.iterations += 1
        self.timestep += 1
        if self.timestep % UPDATE_EVERY == 0:
            if len(self.memory) > BATCH_SIZE:
                sampled_experiences = self.memory.sample()
                self.learning(sampled_experiences)
        self.observation_space = next_state
