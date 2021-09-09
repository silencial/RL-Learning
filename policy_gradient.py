import argparse

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
    def __init__(self, sizes):
        super().__init__()
        self.logits_net = mlp(sizes)

    def _get_distribution(self, obs):
        logits = self.logits_net(torch.as_tensor(obs, dtype=torch.float32))
        return Categorical(logits=logits)

    def forward(self, obs, act=None):
        with torch.no_grad():
            pi = self._get_distribution(obs)
            return pi.sample().item()

    def get_logp(self, obs, act):
        pi = self._get_distribution(obs)
        return pi.log_prob(torch.as_tensor(act, dtype=torch.int32))


class MLPGaussian(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.mu_net = mlp(sizes)
        self.log_std = nn.Parameter(torch.ones(sizes[-1]) * -0.5)

    def _get_distribution(self, obs):
        mu = self.mu_net(torch.as_tensor(obs, dtype=torch.float32))
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def forward(self, obs, act=None):
        with torch.no_grad():
            pi = self._get_distribution(obs)
            return pi.sample().numpy()

    def get_logp(self, obs, act):
        pi = self._get_distribution(obs)
        return pi.log_prob(torch.as_tensor(act, dtype=torch.float32)).sum(axis=-1)


def train(env_name, lr, rtg_flag, render, hidden_size=[32], epochs=50, batch_size=5000):
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
    sizes = [obs_dim] + hidden_size + [act_dim]
    pi = pi_model(sizes)
    optimizer = Adam(pi.parameters(), lr=lr)

    def sample_batch(batch_size):
        obs_batch, act_batch, rew_batch = [], [], []
        obs_episode, act_episode, rew_episode = [], [], []
        batch_len = 0
        obs = env.reset()

        first_episode = True

        while batch_len < batch_size:
            if render and first_episode:
                env.render()

            obs_episode.append(obs)
            act = pi(obs)
            obs, rew, done, _ = env.step(act)

            act_episode.append(act)
            rew_episode.append(rew)

            if done:
                first_episode = False

                batch_len += len(act_episode)
                obs_batch.append(obs_episode)
                act_batch.append(act_episode)
                rew_batch.append(rew_episode)

                obs_episode, act_episode, rew_episode = [], [], []
                obs = env.reset()

        return obs_batch, act_batch, rew_batch

    def compute_loss(obs_batch, act_batch, rew_batch):
        loss = 0.0
        for obs, act, rew in zip(obs_batch, act_batch, rew_batch):
            logp = pi.get_logp(obs, act)
            if rtg_flag:
                rtg = compute_rtg(rew)
                loss += -(logp * rtg).sum()
            else:
                loss += -logp.sum() * np.sum(rew)
        return loss / len(rew_batch)

    def compute_rtg(rew):
        rtg = np.copy(rew)
        for i in reversed(range(len(rtg) - 1)):
            rtg[i] += rtg[i + 1]
        return torch.as_tensor(rtg, dtype=torch.float32)

    def train_one_epoch():
        obs_batch, act_batch, rew_batch = sample_batch(batch_size)
        loss = compute_loss(obs_batch, act_batch, rew_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_lens = [len(obs) for obs in obs_batch]
        rew_episode = [np.sum(rew) for rew in rew_batch]
        return loss, np.mean(batch_lens), np.mean(rew_episode)

    for i in range(epochs):
        ep_loss, ep_len, ep_rew = train_one_epoch()
        print(f'Epoch: {i:2d}, \t Loss: {ep_loss:7.2f}, \t Reward: {ep_rew:6.2f}, \t Length: {ep_len:6.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '-env', type=str, default='CartPole-v0')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    train(args.env_name, args.lr, args.reward_to_go, args.render)
