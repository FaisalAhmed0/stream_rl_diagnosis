import os, pickle, argparse
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from minigrid.wrappers import ImgObsWrapper 
from optim import ObGD as Optimizer
from normalization_wrappers import NormalizeObservation, ScaleReward
from sparse_init import sparse_init
import wandb

class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(obs_shape[2], obs_shape[0], obs_shape[1]), dtype=env.observation_space.dtype
        )
    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class LayerNormalization(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return F.layer_norm(input, input.size())

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

class StreamQ(nn.Module):
    def __init__(self, n_actions=3, hidden_size=256, lr=1.0, epsilon_target=0.01, epsilon_start=1.0, exploration_fraction=0.1, total_steps=1_000_000, gamma=0.99, lamda=0.8, kappa_value=2.0):
        super(StreamQ, self).__init__()
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_target = epsilon_target
        self.epsilon = epsilon_start
        self.exploration_fraction = exploration_fraction
        self.total_steps = total_steps
        self.time_step = 0
        
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=1), # 3 canaux (RGB), pas 4 (FrameStack)
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, hidden_size), # 1024 entrées
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, n_actions)
        )
        
        self.apply(initialize_weights)
        self.optimizer = Optimizer(list(self.parameters()), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

    def q(self, x):
        x = torch.tensor(np.array(x), dtype=torch.float).unsqueeze(0) # Ajout unsqueeze pour batch dim
        return self.network(x)

    def sample_action(self, s):
        self.time_step += 1
        self.epsilon = linear_schedule(self.epsilon_start, self.epsilon_target, self.exploration_fraction * self.total_steps, self.time_step)
        
        if np.random.rand() < self.epsilon:
            q_values = self.q(s)
            greedy_action = torch.argmax(q_values, dim=-1).item()
            random_action = np.random.randint(0, self.n_actions)
            if greedy_action == random_action:
                return random_action
            else:
                return random_action
        else:
            q_values = self.q(s)
            return torch.argmax(q_values, dim=-1).item()

    def update_params(self, s, a, r, s_prime, done, overshooting_info=False):
        done_mask = 0 if done else 1
        r = torch.tensor(r, dtype=torch.float)
        done_mask = torch.tensor(done_mask, dtype=torch.float)
        
        q_sa = self.q(s).squeeze()[a]
        max_q_s_prime_a_prime = torch.max(self.q(s_prime).squeeze(), dim=-1).values
        
        td_target = r + self.gamma * max_q_s_prime_a_prime * done_mask
        delta = td_target - q_sa

        q_output = -q_sa
        self.optimizer.zero_grad()
        q_output.backward()
        self.optimizer.step(delta.item(), reset=done)

        metrics = {
            "training/value": q_sa.item(),
            "training/delta": delta.item(),
            "training/td_target": td_target.item(),
            "training/epsilon": self.epsilon
        }
        return metrics

def main(env_name, seed, lr, gamma, lamda, total_steps, epsilon_target, epsilon_start, exploration_fraction, kappa_value, debug, overshooting_info, render=False, track=0, args=None):
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make(env_name, render_mode='human') if render else gym.make(env_name)
    
    env = ImgObsWrapper(env)
    env = TransposeImage(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = ScaleReward(env, gamma=gamma)
    env = NormalizeObservation(env)

    agent = StreamQ(n_actions=env.action_space.n, lr=lr, gamma=gamma, lamda=lamda, epsilon_target=epsilon_target, epsilon_start=epsilon_start, exploration_fraction=exploration_fraction, total_steps=total_steps, kappa_value=kappa_value)
    
    if debug:
        print("seed: {}".format(seed), "env: {}".format(env.spec.id))
        
    if track:
        wandb.init(
            project="Stream_Q_Minigrid_2",
            mode="online",
            entity="streaming-x-diagnosis",
            config=vars(args) if args else {},
            name=f"Stream Q(λ)_env_{env_name}_lambda_{lamda}_seed_{seed}"
        )
        
    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)
    
    for t in range(1, total_steps+1):
        a = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        
        metrics = agent.update_params(s, a, r, s_prime, done, overshooting_info)
        s = s_prime
        
        if track:
             # Log à chaque step
            if done and "episode" in info:
                metrics["training/episodic_return"] = info['episode']['r'][0]
                metrics["training/episode_length"] = info['episode']['l'][0]
            wandb.log(metrics, step=t)

        if done:
            if debug and "episode" in info:
                print("Episodic Return: {}, Time Step {}, Epsilon {:.4f}".format(info['episode']['r'][0], t, agent.epsilon))
            
            if "episode" in info:
                returns.append(info['episode']['r'][0])
                term_time_steps.append(t)
            s, _ = env.reset()
            
    env.close()
    if track:
        wandb.finish()
        
    save_dir = "data_stream_q_{}_lr{}_gamma{}_lamda{}".format(env.spec.id, lr, gamma, lamda)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream Q(λ)')
    parser.add_argument('--env_name', type=str, default='MiniGrid-Empty-5x5-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--epsilon_target', type=float, default=0.01)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--exploration_fraction', type=float, default=0.5)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--total_steps', type=int, default=200_000)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overshooting_info', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--track', type=int, default=0)
    args = parser.parse_args()
        
    main(args.env_name, args.seed, args.lr, args.gamma, args.lamda, args.total_steps, args.epsilon_target, args.epsilon_start, args.exploration_fraction, args.kappa_value, args.debug, args.overshooting_info, args.render, args.track, args)