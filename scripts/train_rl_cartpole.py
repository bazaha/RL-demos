"""Classic RL validation: DQN on CartPole-v1 (gymnasium), pure PyTorch, on GPU.

Writes results/rl_metrics.json with per-episode reward, moving average, loss,
epsilon, and greedy-policy state trajectories (for animation) at several
checkpoints during training.
"""
import json
import os
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
EPISODES = 400
GAMMA = 0.99
LR = 1e-3
BATCH = 128
BUFFER = 50_000
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.02, 8_000  # exponential decay in steps
TARGET_SYNC = 500  # steps
SOLVED_AVG = 475.0  # 100-episode moving average threshold for CartPole-v1
TRAJ_CHECKPOINTS = [1, 50, 100, 200, 300, 400]  # record greedy rollout after these episodes
OUT = "results/rl_metrics.json"


class QNet(nn.Module):
    def __init__(self, obs, act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act),
        )

    def forward(self, x):
        return self.net(x)


def greedy_rollout(env_name, policy, device, seed):
    """Run one greedy episode, return states [(x, theta), ...] and total reward."""
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    states, total = [], 0.0
    for _ in range(500):
        states.append([round(float(obs[0]), 4), round(float(obs[2]), 4)])
        with torch.no_grad():
            a = policy(torch.as_tensor(obs, dtype=torch.float32, device=device)).argmax().item()
        obs, r, term, trunc, _ = env.step(a)
        total += r
        if term or trunc:
            break
    env.close()
    return states, total


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = QNet(obs_dim, act_dim).to(device)
    target = QNet(obs_dim, act_dim).to(device)
    target.load_state_dict(policy.state_dict())
    opt = torch.optim.AdamW(policy.parameters(), lr=LR)
    buf = deque(maxlen=BUFFER)

    metrics = {
        "algo": "DQN", "env": "CartPole-v1", "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "episodes": EPISODES, "solved_avg": SOLVED_AVG,
        "phase_start": time.time(),
        "episode_stats": [],  # {episode, reward, avg100, epsilon, loss}
        "trajectories": {},   # {episode: {states: [[x, theta]...], reward}}
    }

    step = 0
    rewards_hist = deque(maxlen=100)
    solved_at = None
    obs, _ = env.reset(seed=SEED)

    for ep in range(1, EPISODES + 1):
        ep_reward, done = 0.0, False
        losses = []
        while not done:
            eps = EPS_END + (EPS_START - EPS_END) * np.exp(-step / EPS_DECAY)
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy(torch.as_tensor(obs, dtype=torch.float32, device=device)).argmax().item()
            nobs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            buf.append((obs, action, r, nobs, float(term)))
            obs = nobs
            ep_reward += r
            step += 1

            if len(buf) >= 1_000:
                batch = random.sample(buf, BATCH)
                s, a, rew, ns, d = map(np.array, zip(*batch))
                s = torch.as_tensor(s, dtype=torch.float32, device=device)
                a = torch.as_tensor(a, dtype=torch.int64, device=device)
                rew = torch.as_tensor(rew, dtype=torch.float32, device=device)
                ns = torch.as_tensor(ns, dtype=torch.float32, device=device)
                d = torch.as_tensor(d, dtype=torch.float32, device=device)
                q = policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    nq = target(ns).max(1).values
                    tgt = rew + GAMMA * nq * (1 - d)
                loss = F.smooth_l1_loss(q, tgt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                opt.step()
                losses.append(loss.item())
                if step % TARGET_SYNC == 0:
                    target.load_state_dict(policy.state_dict())

        obs, _ = env.reset()
        rewards_hist.append(ep_reward)
        avg100 = float(np.mean(rewards_hist))
        metrics["episode_stats"].append({
            "episode": ep, "reward": ep_reward, "avg100": round(avg100, 1),
            "epsilon": round(float(eps), 3),
            "loss": round(float(np.mean(losses)), 5) if losses else None,
        })
        if solved_at is None and len(rewards_hist) == 100 and avg100 >= SOLVED_AVG:
            solved_at = ep
        if ep in TRAJ_CHECKPOINTS:
            states, total = greedy_rollout("CartPole-v1", policy, device, seed=SEED + ep)
            metrics["trajectories"][str(ep)] = {"states": states, "reward": total}
            print(f"  [traj @ep{ep}] greedy reward={total}", flush=True)
        if ep % 20 == 0:
            print(f"ep {ep}/{EPISODES} reward={ep_reward:.0f} avg100={avg100:.1f} eps={eps:.3f}", flush=True)

    metrics["phase_end"] = time.time()
    metrics["solved_at"] = solved_at
    metrics["final_avg100"] = round(float(np.mean(rewards_hist)), 1)
    env.close()
    os.makedirs("results", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(metrics, f)
    print(f"wrote {OUT}; solved_at={solved_at} final_avg100={metrics['final_avg100']}")


if __name__ == "__main__":
    main()
