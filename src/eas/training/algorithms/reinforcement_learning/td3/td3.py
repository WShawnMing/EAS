
import copy
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -------------------------------
# 1. Replay Buffer
# -------------------------------
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)

        def tensor(x):
            return torch.as_tensor(np.array(x), dtype=torch.float32, device=DEVICE)

        return (
            tensor(s),
            tensor(a),
            tensor(r).unsqueeze(1),
            tensor(ns),
            tensor(d).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# -------------------------------
# 2. Actor: state -> action
# -------------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, s):
        return self.net(s) * self.max_action


# -------------------------------
# 3. Twin Critic: (s,a) -> Q1,Q2
# -------------------------------
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        def make_q():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

        self.q1 = make_q()
        self.q2 = make_q()

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)

    def Q1(self, s, a):
        return self.q1(torch.cat([s, a], dim=-1))


# -------------------------------
# 4. TD3 Agent
# -------------------------------
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action

        self.actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.policy_delay = 2
        self.updates = 0

    def select_action(self, state, explore=True):
        s = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(s).cpu().numpy()[0]

        if explore:
            action += np.random.normal(0, 0.1 * self.max_action, size=action.shape)

        return np.clip(action, -self.max_action, self.max_action).astype(np.float32)

    def train(self, replay, batch_size=256):
        s, a, r, ns, done = replay.sample(batch_size)

        # Step 1: Bellman Target
        with torch.no_grad():
            noise = torch.randn_like(a) * self.policy_noise
            2 = noise.clamp(-self.noise_clip, self.noise_clip)

            next_a = (self.actor_target(ns) + noise).clamp(
                -self.max_action, self.max_action
            )

            tq1, tq2 = self.critic_target(ns, next_a)
            target_q = r + self.gamma * (1 - done) * torch.min(tq1, tq2)

        # Step 2: update Twin Critic
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self.updates += 1

        # Step 3: delayed Actor update
        if self.updates % self.policy_delay == 0:
            actor_loss = -self.critic.Q1(s, self.actor(s)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Step 4: soft-update target networks
            with torch.no_grad():
                for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                    tp.lerp_(p, self.tau)

                for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                    tp.lerp_(p, self.tau)


# -------------------------------
# 5. Training
# -------------------------------
def main():
    env = gym.make("Pendulum-v1")
    env.action_space.seed(SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3(state_dim, action_dim, max_action)
    replay = ReplayBuffer()

    total_steps = 100_000
    start_steps = 5_000
    batch_size = 256

    state, _ = env.reset(seed=SEED)
    episode_reward = 0
    episode = 0

    for step in range(total_steps):
        if step < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, explore=True)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Time-limit truncation is not an MDP terminal state
        replay.add(state, action, reward, next_state, float(terminated))

        state = next_state
        episode_reward += reward

        if len(replay) >= batch_size and step >= start_steps:
            agent.train(replay, batch_size)

        if done:
            episode += 1
            print(f"TD3 | Episode {episode:3d} | Step {step+1:6d} | Reward {episode_reward:8.2f}")
            state, _ = env.reset()
            episode_reward = 0

    torch.save(agent.actor.state_dict(), "td3_actor.pt")
    env.close()

    # Evaluation
    test_env = gym.make("Pendulum-v1")
    rewards = []
    for seed in range(5):
        s, _ = test_env.reset(seed=100 + seed)
        total = 0
        for _ in range(200):
            a = agent.select_action(s, explore=False)
            s, r, terminated, truncated, _ = test_env.step(a)
            total += r
            if terminated or truncated:
                break
        rewards.append(total)

    print(f"TD3 Evaluation: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    test_env.close()


if __name__ == "__main__":
    main()
