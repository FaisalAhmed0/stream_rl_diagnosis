import os, pickle, argparse
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions import Normal
from optim import ObGD as Optimizer
from time_wrapper import AddTimeInfo
from normalization_wrappers import NormalizeObservation, ScaleReward
from sparse_init import sparse_init
import wandb 

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def create_spare_initializer(sparsity=0.9):
    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            sparse_init(m.weight, sparsity=sparsity)
            m.bias.data.fill_(0.0)
    return initialize_weights

class StreamQ(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, lr=1.0, epsilon_target=0.01, epsilon_start=1.0, exploration_fraction=0.1, total_steps=1_000_000, gamma=0.99, lamda=0.8, kappa_value=2.0, sparsity=0.9):
        super(StreamQ, self).__init__()
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_target = epsilon_target
        self.epsilon = epsilon_start
        self.exploration_fraction = exploration_fraction
        self.total_steps = total_steps
        self.time_step = 0

        self.fc1_mu = nn.Linear(n_obs, hidden_size)
        self.hidden_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, n_actions)

        self.fc1_q = nn.Linear(n_obs + n_actions, hidden_size)
        self.hidden_q = nn.Linear(hidden_size, hidden_size)
        self.fc_q = nn.Linear(hidden_size, 1)

        self.log_std = nn.Parameter(torch.zeros(1, n_actions))
        
        initialize_weights = create_spare_initializer(sparsity)
        self.apply(initialize_weights)
        
        self.optimizer = Optimizer(list(self.parameters()), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

    def get_action_distribution(self, x):
        """Get mean and standard deviation for action distribution"""
        x_mu = self.fc1_mu(x)
        x_mu = F.layer_norm(x_mu, x_mu.size())
        x_mu = F.leaky_relu(x_mu)
        x_mu = self.hidden_mu(x_mu)
        x_mu = F.layer_norm(x_mu, x_mu.size())
        x_mu = F.leaky_relu(x_mu)
        mu = self.fc_mu(x_mu)
        std = torch.exp(self.log_std)
        return mu, std

    def q_value(self, state, action):
        """Compute Q-value for state-action pair"""
        x = torch.cat([state, action], dim=-1)
        x = self.fc1_q(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_q(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        return self.fc_q(x)

    def sample_action(self, s):
        self.time_step += 1
        self.epsilon = linear_schedule(self.epsilon_start, self.epsilon_target, 
                                      self.exploration_fraction * self.total_steps, 
                                      self.time_step)
        
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        
        mu, std = self.get_action_distribution(s)
        dist = Normal(mu, std)
        
        if np.random.rand() < self.epsilon:
            random_action = np.random.uniform(-1, 1, self.n_actions)
            random_action = torch.tensor(random_action, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                greedy_action = mu
            
            if torch.allclose(greedy_action, random_action, atol=0.1):
                return random_action.squeeze(0).numpy(), False, dist
            else:
                return random_action.squeeze(0).numpy(), True, dist
        else:
            action = dist.sample()
            return action.squeeze(0).numpy(), False, dist

    def compute_entropy(self, dist):
        """Compute entropy of action distribution"""
        return dist.entropy().mean()

    def compute_log_prob(self, dist, action):
        """Compute log probability of action under distribution"""
        return dist.log_prob(action).sum(dim=-1, keepdim=True).mean()

    def update_params(self, s, a, r, s_prime, done, is_nongreedy, entropy_coeff=0.01, overshooting_info=False):
        done_mask = 0 if done else 1
        
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        a = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
        r = torch.tensor(r, dtype=torch.float32)
        s_prime = torch.tensor(s_prime, dtype=torch.float32).unsqueeze(0)
        done_mask = torch.tensor(done_mask, dtype=torch.float32)

        mu, std = self.get_action_distribution(s)
        dist = Normal(mu, std)
        
        mu_prime, std_prime = self.get_action_distribution(s_prime)
        dist_prime = Normal(mu_prime, std_prime)

        a_prime = dist_prime.sample()

        q_sa = self.q_value(s, a)
        q_s_prime_a_prime = self.q_value(s_prime, a_prime)

        td_target = r + self.gamma * q_s_prime_a_prime * done_mask
        delta = td_target - q_sa
        
        q_loss = -q_sa
        
        log_prob = dist.log_prob(a).sum(dim=-1, keepdim=True)
        policy_loss = -log_prob * delta.detach()
        
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        entropy_loss = -entropy_coeff * entropy
        
        total_loss = q_loss + 0.01 * policy_loss + entropy_coeff * entropy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step(delta.item(), reset=(done or is_nongreedy))
        
        with torch.no_grad():
            q_values = [q_sa.item(), q_s_prime_a_prime.item()]
            
            action_mean = mu.mean().item()
            action_std = std.mean().item()
            log_prob_value = log_prob.mean().item()
            entropy_value = entropy.mean().item()
            
            grad_norm = 0.0
            for param in self.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            action_magnitude = torch.norm(a).item()
            action_max = a.abs().max().item()
        
        metrics = {
            "training/q_value": q_sa.item(),
            "training/q_target": q_s_prime_a_prime.item(),
            "training/td_target": td_target.item(),
            "training/delta": delta.item(),
            "training/td_error": delta.abs().item(),
            

            "training/policy_loss": policy_loss.mean().item(),
            "training/log_prob": log_prob_value,
            "training/entropy": entropy_value,
            "training/entropy_coeff": entropy_coeff,
            

            "training/q_loss": q_loss.item(),
            "training/total_loss": total_loss.item(),
            

            "training/action_mean": action_mean,
            "training/action_std": action_std,
            "training/action_magnitude": action_magnitude,
            "training/action_max": action_max,
            

            "training/epsilon": self.epsilon,
            "training/is_nongreedy": float(is_nongreedy),
            

            "training/grad_norm": grad_norm,
            

            "training/q_ratio": q_s_prime_a_prime.item() / max(q_sa.item(), 1e-8),
            "training/td_ratio": td_target.item() / max(q_sa.item(), 1e-8),
        }

        if overshooting_info:

            with torch.no_grad():
                q_sa_new = self.q_value(s, a)
                q_s_prime_a_prime_new = self.q_value(s_prime, a_prime)
                td_target_new = r + self.gamma * q_s_prime_a_prime_new * done_mask
                delta_bar = td_target_new - q_sa_new
                overshooting = torch.sign(delta_bar * delta).item() == -1
                if overshooting:
                    print("Overshooting Detected!")
                    metrics["training/overshooting"] = 1.0
                else:
                    metrics["training/overshooting"] = 0.0
        
        return metrics

def main(env_name, seed, lr, gamma, lamda, total_steps, epsilon_target, epsilon_start, exploration_fraction, kappa_value, debug, overshooting_info, render=False, track=False, args=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    

    env = gym.make(env_name, render_mode='human', max_episode_steps=10_000) if render else gym.make(env_name, max_episode_steps=10_000)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env) 
    env = ScaleReward(env, gamma=gamma)
    env = NormalizeObservation(env)
    env = AddTimeInfo(env)
    

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    gradient_steps_per_step = args.gradient_steps_per_step if args else 1


    agent = StreamQ(
        n_obs=obs_dim, 
        n_actions=action_dim, 
        lr=lr, 
        gamma=gamma, 
        lamda=lamda, 
        epsilon_target=epsilon_target, 
        epsilon_start=epsilon_start, 
        exploration_fraction=exploration_fraction, 
        total_steps=total_steps, 
        kappa_value=kappa_value,
        sparsity=args.sparsity if args else 0.9
    )
    
    if debug:
        print(f"seed: {seed}, env: {env.spec.id}")
        print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
        print(f"Total parameters: {sum(p.numel() for p in agent.parameters()):,}")
    
    if track:
        wandb.init(
            project="Stream Q(λ) Continuous",
            mode="online",
            config=vars(args),
            name=f"Stream Q(λ)_env_{env_name}_seed_{seed}_gradient_updates_per_step{args.gradient_steps_per_step}_sparsity_{args.sparsity}"
        )
    
    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)
    episode_num = 1
    

    episode_metrics = {
        'episode_returns': [],
        'episode_lengths': [],
        'mean_q_values': [],
        'mean_entropy': [],
        'exploration_rate': []
    }
    
    for t in range(1, total_steps+1):
        a, is_nongreedy, dist = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        

        with torch.no_grad():
            entropy = agent.compute_entropy(dist)
        
        print(gradient_steps_per)
        
        for grad_step in range(gradient_steps_per_step):
            metrics = agent.update_params(s, a, r, s_prime, terminated or truncated, 
                                          is_nongreedy, entropy_coeff=0.01, 
                                          overshooting_info=overshooting_info)

            if grad_step == gradient_steps_per_step - 1 and track and t % 100 == 0:
                metrics.update({
                    "training/step_entropy": entropy.item(),
                    "training/step_reward": r,
                    "training/step": t,
                    "training/gradient_step": grad_step + 1,
                    "training/total_gradient_steps": t * gradient_steps_per_step,
                })

        s = s_prime
        

        metrics.update({
            "training/step_entropy": entropy.item(),
            "training/step_reward": r,
            "training/step": t,
        })
        
        if terminated or truncated:
            episode_return = info['episode']['r'][0]
            episode_length = info['episode']['l'][0]
            
            if debug:
                print(f"Episode {episode_num}: "
                      f"Return: {episode_return:.2f}, "
                      f"Length: {episode_length}, "
                      f"Epsilon: {agent.epsilon:.3f}, "
                      f"Step: {t}")
            

            episode_metrics = {
                "training/episodic_return": episode_return,
                "training/episode_length": episode_length,
                "training/episode": episode_num,
                "training/mean_episode_q": np.mean(episode_metrics.get('mean_q_values', [0])) if episode_metrics['mean_q_values'] else 0,
                "training/mean_episode_entropy": np.mean(episode_metrics.get('mean_entropy', [0])) if episode_metrics['mean_entropy'] else 0,
                "training/exploration_rate": np.mean(episode_metrics.get('exploration_rate', [0])) if episode_metrics['exploration_rate'] else 0,
            }
            
            if track:
                metrics.update(episode_metrics)
                wandb.log(metrics, step=t)
            
            returns.append(episode_return)
            term_time_steps.append(t)
            

            s, _ = env.reset()
            episode_num += 1
            

            episode_metrics = {
                'mean_q_values': [],
                'mean_entropy': [],
                'exploration_rate': []
            }
        elif track and t % 100 == 0: 

            metrics.update({
                "running/mean_return": np.mean(returns[-10:]) if returns else 0,
                "running/max_return": np.max(returns[-50:]) if returns else 0,
                "running/min_return": np.min(returns[-50:]) if returns else 0,
                "running/std_return": np.std(returns[-50:]) if returns else 0,
            })
            wandb.log(metrics, step=t)
    
    env.close()
    
    if track:

        if returns:
            wandb.log({
                "final/mean_return": np.mean(returns),
                "final/max_return": np.max(returns),
                "final/min_return": np.min(returns),
                "final/std_return": np.std(returns),
                "final/total_episodes": len(returns),
            }, step=total_steps)
        wandb.finish()
    

    save_dir = f"data_stream_q_{env.spec.id}_lr{lr}_gamma{gamma}_lamda{lamda}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(os.path.join(save_dir, f"seed_{seed}.pkl"), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)
    
    if debug:
        print(f"\nTraining completed!")
        print(f"Total episodes: {len(returns)}")
        print(f"Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
        print(f"Best return: {np.max(returns):.2f}")
        print(f"Final epsilon: {agent.epsilon:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream Q(λ) for Continuous Control')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--epsilon_target', type=float, default=0.01)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--exploration_fraction', type=float, default=0.05)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--total_steps', type=int, default=2_000_000)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overshooting_info', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--track', type=int, default=0)
    parser.add_argument('--sparsity', type=float, default=0.9, help="Amount of sparsity in the neural network initialization")
    parser.add_argument('--gradient_steps_per_step', type=int, default=1, help='Number of gradient steps per environment step')

    parser.add_argument('--entropy_coeff', type=float, default=0.01,
                       help="Entropy regularization coefficient")
    args = parser.parse_args()
    
    main(args.env_name, args.seed, args.lr, args.gamma, args.lamda, 
         args.total_steps, args.epsilon_target, args.epsilon_start, 
         args.exploration_fraction, args.kappa_value, args.debug, 
         args.overshooting_info, args.render, args.track, args)