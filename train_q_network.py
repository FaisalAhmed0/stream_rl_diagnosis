from math import trunc
import os, pickle, argparse
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from optim import ObGD as Optimizer
from time_wrapper import AddTimeInfo
from normalization_wrappers import NormalizeObservation, ScaleReward
from sparse_init import sparse_init
from tabular_envs import CliffwalkEnv, InvertedPendulum, MountainCar, GymnasiumTabularWrapper
import wandb 


Expert_returns = {
    "MountainCar": 164, 
    "InvertedPendulum": 191,
    "CliffwalkEnv": 12
}

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

def evaluation(network, num_states, tabular_q_model, device="cpu", batch_size=1024, num_evals=5):
    validation_losses = []
    for _ in range(num_evals):
        states = np.random.choice(num_states, size=batch_size)
        states_onehot = F.one_hot(torch.tensor(states, device=device), num_classes=num_states).float()
        target_values = torch.tensor(tabular_q_model[states], device=device, dtype=torch.float)
        predicted_values = network(states_onehot)
        loss = F.mse_loss(predicted_values, target_values)
        validation_losses.append(loss.item())
    return np.mean(validation_losses)
    
    

def load_q_model(env_name, seed=1):
    print("load the file")
    file_path = f"env_name_{env_name}_value_iteration.txt"
    return load_array(file_path)
    

def load_array(filepath):
    """
    Load a numpy array from a human-readable text file.
    
    Args:
        filepath: path to the file to load
        
    Returns:
        numpy array with the original shape restored
    """
    shape = None
    
    # Read the file to get the shape from the header
    with open(filepath, 'r') as f:
        # Read first few lines to find shape info
        for line in f:
            if line.startswith('# Shape:'):
                shape_str = line.replace('# Shape:', '').strip()
                shape = tuple(map(int, shape_str.split(',')))
                break
    
    # Load the data (skip comment lines)
    data = np.loadtxt(filepath, comments='#')
    
    # Reshape if shape was specified
    if shape is not None and data.size == np.prod(shape):
        data = data.reshape(shape)
    
    return data

class StreamQ(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=32, layer_norm=1, args=None):
        super(StreamQ, self).__init__()
        assert args is not None
        self.args = args
        self.n_actions = n_actions
        self.layer_norm = layer_norm
        self.fc1_v   = nn.Linear(n_obs, hidden_size)
        self.hidden_v  = nn.Linear(hidden_size, hidden_size)
        self.fc_v  = nn.Linear(hidden_size, n_actions)
        # self.apply(initialize_weights)

    def q(self, x):
        x = self.fc1_v(x)
        x = F.layer_norm(x, x.size()) if self.layer_norm else nn.Identity()(x)
        x = F.leaky_relu(x)
        x = self.hidden_v(x)
        x = F.layer_norm(x, x.size()) if self.layer_norm else  nn.Identity()(x)
        x = F.leaky_relu(x)
        return self.fc_v(x)

    def forward(self, x):
        return self.q(x)

    

def main(env_name, seed, lr, gamma, lamda, total_steps, epsilon_target, epsilon_start, exploration_fraction, kappa_value, debug, overshooting_info, render=False, track=False, args=None, tabular_env=False, layer_norm=1):
    torch.manual_seed(seed); np.random.seed(seed)
    # if debug:
    #     print("seed: {}".format(seed), "env: {}".format(env.spec.id))
    if track:
        wandb.init(
            project="Stream Q(λ)_tabular_envs_offline_training",
            mode="online",
            config=vars(args),
            name=f"env_name_{env_name}_seed_{seed}_layer_norm_{args.layer_norm}",
            entity="streaming-x-diagnosis"
        )
    tabular_q_function = load_q_model(env_name, seed)
    num_states, num_actions = tabular_q_function.shape
    q_network = StreamQ(n_obs=num_states, n_actions=num_actions, layer_norm=layer_norm, args=args)
    optimizer = torch.optim.Adam(list(q_network.parameters()), lr=lr)
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network.to(device)
    # Training loop
    for t in range(1, total_steps+1):
        # Sample states and actions uniformly 
        states = np.random.choice(num_states, size=batch_size)
        states_onehot = F.one_hot(torch.tensor(states, device=device), num_classes=num_states).float()
        target_values = torch.tensor(tabular_q_function[states], device=device, dtype=torch.float)
        predicted_values = q_network(states_onehot)
        loss = F.mse_loss(predicted_values, target_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.track:
            wandb.log({"training/loss": loss.item()}, step=t)
        if debug:
            print(f"Training loss: {loss.item()}")
        if t % args.eval_freq == 0:
            with torch.no_grad():
                validation_loss = evaluation(q_network, num_states, tabular_q_function, device, batch_size=1024)
            eval_logs = {
                "eval/validation_loss": validation_loss,
            }
            if args.track:
                wandb.log(eval_logs, step=t)
            if debug:
                print(f"Validation loss: {validation_loss}")
    if track:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream Q(λ)')
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--tabular_env', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--epsilon_target', type=float, default=0.01)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--exploration_fraction', type=float, default=0.3)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--total_steps', type=int, default=500_000)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overshooting_info', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--eval_freq', type=int, default=1_000)
    parser.add_argument('--batch_size', type=int, default=1024)
    
    # track with wandb
    parser.add_argument('--track', type=int, default=0)
    ## For empirical analysis
    parser.add_argument('--sparsity', type=float, default=0, help="Amount of sparsity in the neural network intialization")
    parser.add_argument('--layer_norm', type=int, default=1)
    

    args = parser.parse_args()
    main(args.env_name, args.seed, args.lr, 
    args.gamma, args.lamda, args.total_steps, 
    args.epsilon_target, args.epsilon_start, args.exploration_fraction, 
    args.kappa_value, args.debug, args.overshooting_info, args.render, args.track, args, tabular_env=args.tabular_env, layer_norm=args.layer_norm)