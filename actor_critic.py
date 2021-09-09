import argparse
import copy

import gym
from gym.spaces import Discrete, Box
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)


class MLPLogits(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.logits_net = mlp(sizes)

    def _get_distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def forward(self, obs, act=None):
        with torch.no_grad():
            pi = self._get_distribution(obs)
            return pi.sample().item()

    def get_logp(self, obs, act):
        pi = self._get_distribution(obs)
        return pi.log_prob(torch.as_tensor(act, dtype=torch.int32))


class MLPGaussian(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.mu_net = mlp(sizes)
        self.log_std = nn.Parameter(torch.ones(sizes[-1]) * -0.5)

    def _get_distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def forward(self, obs, act=None):
        with torch.no_grad():
            pi = self._get_distribution(obs)
            return pi.sample().numpy()

    def get_logp(self, obs, act):
        pi = self._get_distribution(obs)
        return pi.log_prob(torch.as_tensor(act, dtype=torch.float32)).sum(axis=-1)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [1]
        self.v_net = mlp(sizes)

    def forward(self, obs):
        return self.v_net(obs).flatten()


def train(env_name, gamma, lam, pi_lr, v_lr, render, hidden_sizes=[32],
          epochs=50, batch_size=5000, train_v_iter=80):
    # Environment
    env = gym.make(env_name)

    assert isinstance(env.observation_space, Box), \
        "Environment does not have continuous state space"
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, Box):
        act_dim = env.action_space.shape[0]
        pi_model = MLPGaussian
    elif isinstance(env.action_space, Discrete):
        act_dim = env.action_space.n
        pi_model = MLPLogits

    # Seed
    env.seed(0)
    torch.manual_seed(0)

    # Model
    pi = pi_model(obs_dim, act_dim, hidden_sizes)
    v = MLPCritic(obs_dim, hidden_sizes)
    pi_optimizer = Adam(pi.parameters(), lr=pi_lr)
    v_optimizer = Adam(v.parameters(), lr=v_lr)

    def sample_batch(batch_size):
        obs_batch, act_batch, rew_batch = [], [], []
        obs_ep, rew_ep = [], []
        batch_len = 0
        obs = env.reset()

        first_episode = True

        while batch_len < batch_size:
            if render and first_episode:
                env.render()

            obs_ep.append(obs)
            act = pi(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            act_batch.append(act)
            rew_ep.append(rew)

            if done:
                first_episode = False
                obs_batch.append(obs_ep)
                rew_batch.append(rew_ep)
                obs_ep, rew_ep = [], []
                batch_len += len(act_batch)
                obs = env.reset()

        return obs_batch, act_batch, rew_batch

    def compute_loss_pi(obs, act, rew):
        obs_tensor = torch.as_tensor(np.concatenate(obs), dtype=torch.float32)
        logp = pi.get_logp(obs_tensor, act)
        gae = compute_gae(obs, rew)
        loss = -(logp * gae).mean()
        return loss

    def compute_gae(obs_batch, rew_batch):
        gae = []
        for obs, rew in zip(obs_batch, rew_batch):
            obs = torch.as_tensor(obs, dtype=torch.float32)
            vf = v(obs).detach().numpy()
            vf = np.append(vf, 0)
            delta = np.array(rew) + gamma * vf[1:] - vf[:-1]
            gae.append(discount_cumsum(delta, gamma*lam))
        return torch.as_tensor(np.concatenate(gae), dtype=torch.float32)

    def discount_cumsum(x, discount):
        res = np.array(x)
        for i in reversed(range(len(x) - 1)):
            res[i] += discount * res[i + 1]
        return res

    def compute_rtg(rew_batch):
        weights = []
        for rew in rew_batch:
            weights.append(discount_cumsum(rew, gamma))
        return torch.as_tensor(np.concatenate(weights), dtype=torch.float32)

    def compute_loss_v(obs, rtg):
        obs_tensor = torch.as_tensor(np.concatenate(obs), dtype=torch.float32)
        return ((v(obs_tensor) - rtg) ** 2).mean()

    def train_one_epoch():
        obs_batch, act_batch, rew_batch = sample_batch(batch_size)
        pi_loss = compute_loss_pi(obs_batch, act_batch, rew_batch)
        pi_optimizer.zero_grad()
        pi_loss.backward()
        pi_optimizer.step()

        rtg = compute_rtg(rew_batch)
        for i in range(train_v_iter):
            v_loss = compute_loss_v(obs_batch, rtg)
            v_optimizer.zero_grad()
            v_loss.backward()
            v_optimizer.step()

        batch_lens = [len(rew) for rew in rew_batch]
        rew_ep_sum = [np.sum(rew) for rew in rew_batch]
        return np.mean(batch_lens), np.mean(rew_ep_sum), v_loss.item()

    for i in range(epochs):
        ep_len, ep_rew, v_loss = train_one_epoch()
        print(f'Epoch: {i:2d}, \t Vf loss: {v_loss:6.2f}, \t Reward: {ep_rew:6.2f}, \t Length: {ep_len:6.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '-env', type=str, default='InvertedPendulum-v2')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--pi_lr', type=float, default=1e-2)
    parser.add_argument('--v_lr', type=float, default=1e-2)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    train(args.env_name, args.gamma, args.lam, args.pi_lr, args.v_lr, args.render)
