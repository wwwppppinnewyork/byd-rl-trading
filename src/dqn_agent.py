"""Deep Q-network agent (PyTorch).

Adapted from an open-source DQN stock-trading example by "hcb" (2020); see the README.
Fixes against the first version: greedy actions are returned as plain ints, the target
network output is detached when building the TD target, and the epsilon schedule no
longer adds to None.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
    def __init__(self, n_features, n_actions, units=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, units), nn.ReLU(), nn.Linear(units, n_actions))

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, n_actions, n_features, learning_rate=0.02, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=4000, batch_size=512, seed=0):
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.n_actions, self.n_features = n_actions, n_features
        self.gamma, self.epsilon = reward_decay, e_greedy
        self.replace_target_iter, self.memory_size, self.batch_size = replace_target_iter, memory_size, batch_size
        self.learn_step_counter = self.memory_counter = 0
        self.memory = np.zeros((memory_size, n_features * 2 + 2))     # [s, a, r, s_]
        self.net, self.tgt_net = QNet(n_features, n_actions), QNet(n_features, n_actions)
        self.tgt_net.load_state_dict(self.net.state_dict())
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

    def store_transition(self, s, a, r, s_):
        self.memory[self.memory_counter % self.memory_size] = np.hstack((s, [a, r], s_))
        self.memory_counter += 1

    def choose_action(self, observation, train=True):
        """Epsilon-greedy while training; always greedy in the backtest."""
        if (not train) or self.rng.uniform() < self.epsilon:
            with torch.no_grad():
                q = self.net(torch.as_tensor(observation[np.newaxis, :], dtype=torch.float32))
            return int(q.argmax(dim=1).item())
        return int(self.rng.integers(0, self.n_actions))

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.tgt_net.load_state_dict(self.net.state_dict())
        high = min(self.memory_counter, self.memory_size)
        batch = self.memory[self.rng.choice(high, size=self.batch_size)]
        s = torch.as_tensor(batch[:, :self.n_features], dtype=torch.float32)
        a = torch.as_tensor(batch[:, self.n_features].astype(int))
        r = torch.as_tensor(batch[:, self.n_features + 1], dtype=torch.float32)
        s_ = torch.as_tensor(batch[:, -self.n_features:], dtype=torch.float32)
        q_eval = self.net(s)
        with torch.no_grad():
            q_target = q_eval.clone()
            q_target[torch.arange(self.batch_size), a] = r + self.gamma * self.tgt_net(s_).max(dim=1).values
        loss = nn.functional.mse_loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1
        return float(loss.item())
