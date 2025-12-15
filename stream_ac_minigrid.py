import os, pickle, argparse
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions import Categorical
from minigrid.wrappers import ImgObsWrapper # Important pour Minigrid
from optim import ObGD as Optimizer
from normalization_wrappers import NormalizeObservation, ScaleReward
from sparse_init import sparse_init
import wandb

# --- WRAPPER POUR METTRE L'IMAGE DANS LE BON SENS (C, H, W) ---
class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        # De (H, W, C) vers (C, H, W)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(obs_shape[2], obs_shape[0], obs_shape[1]), dtype=env.observation_space.dtype
        )
    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))

class LayerNormalization(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return F.layer_norm(input, input.size())

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

class StreamAC(nn.Module):
    def __init__(self, n_actions=3, hidden_size=256, lr=1.0, gamma=0.99, lamda=0.8, kappa_policy=3.0, kappa_value=2.0):
        super(StreamAC, self).__init__()
        self.gamma = gamma
        
        # --- MODIFICATION DU CNN POUR MINIGRID (7x7 pixels) ---
        # L'ancien CNN (Atari) réduisait trop l'image. Celui-ci est adapté aux petites grilles.
        # Entrée : 3 canaux (Rouge, Vert, Bleu)
        
        self.network_value = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=1), # Sortie: 16 x 6 x 6
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1), # Sortie: 32 x 5 x 5
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1), # Sortie: 64 x 4 x 4
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, hidden_size), # 1024 -> hidden
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.network_policy = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, hidden_size),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, n_actions)
        )
        # -----------------------------------------------------

        self.apply(initialize_weights)
        self.optimizer_policy = Optimizer(self.network_policy.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy)
        self.optimizer_value = Optimizer(self.network_value.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

    def pi(self, x):
        # x arrive comme (C, H, W), on ajoute la dimension batch -> (1, C, H, W)
        x = torch.tensor(np.array(x), dtype=torch.float).unsqueeze(0) 
        preferences = self.network_policy(x)
        probs = F.softmax(preferences, dim=-1)
        return probs

    def v(self, x):
        x = torch.tensor(np.array(x), dtype=torch.float).unsqueeze(0)
        return self.network_value(x)

    def sample_action(self, s):
        # sample_action n'a pas besoin de gradient, on coupe le graphe avec detach() si besoin
        # mais ici on utilise juste les probs pour sampler
        probs = self.pi(s)
        dist = Categorical(probs)
        return dist.sample().item() # .item() pour avoir un int propre

    def update_params(self, s, a, r, s_prime, done, entropy_coeff, overshooting_info=False):
        done_mask = 0 if done else 1
        
        # Conversion simple pour les scalaires
        r = torch.tensor(r, dtype=torch.float)
        done_mask = torch.tensor(done_mask, dtype=torch.float)
        a = torch.tensor(a)

        # Forward pass (attention, pi et v ajoutent déjà unsqueeze(0))
        v_s = self.v(s).squeeze() # On retire la dim batch pour avoir un scalaire
        v_prime = self.v(s_prime).squeeze()
        
        td_target = r + self.gamma * v_prime * done_mask
        delta = td_target - v_s

        probs = self.pi(s).squeeze()
        dist = Categorical(probs)

        log_prob_pi = -dist.log_prob(a)
        value_output = -v_s

        entropy = dist.entropy() 
        entropy_pi = -entropy_coeff * entropy * torch.sign(delta).item()
        
        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        
        value_output.backward()
        (log_prob_pi + entropy_pi).backward()
        
        self.optimizer_policy.step(delta.item(), reset=done)
        self.optimizer_value.step(delta.item(), reset=done)

        metrics = {
        "training/value": value_output,
        "training/td_target": td_target,
        "training/delta": delta,
        "training/log_prob_pi": log_prob_pi,
        "training/entropy": entropy_pi,
        }

        return metrics

def main(env_name, seed, lr, gamma, lamda, total_steps, entropy_coeff, kappa_policy, kappa_value, debug, overshooting_info, render=False, track=0):
    torch.manual_seed(seed); np.random.seed(seed)
    
    # 1. Chargement de l'environnement
    env = gym.make(env_name, render_mode='human') if render else gym.make(env_name)
    
    # 2. Wrappers spécifiques MINIGRID (Très important !)
    env = ImgObsWrapper(env)       # Enlève le texte "mission", garde l'image
    env = TransposeImage(env)      # Change (7,7,3) -> (3,7,7) pour PyTorch
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # Note: On n'utilise PAS ResizeObservation ni GrayScale pour Minigrid
    # On n'utilise PAS FrameStack ici (on teste la mémoire interne via Lambda, pas la mémoire visuelle)
    
    env = ScaleReward(env, gamma=gamma)
    env = NormalizeObservation(env)
    
    # Initialisation Agent
    # n_actions est souvent 7 dans minigrid (gauche, droite, avancer, prendre...)
    agent = StreamAC(n_actions=env.action_space.n, lr=lr, gamma=gamma, lamda=lamda, kappa_policy=kappa_policy, kappa_value=kappa_value)
    
    if debug:
        print(f"seed: {seed}, env: {env.spec.id}, input_shape: {env.observation_space.shape}")

    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)

    if track:
        wandb.init(
            project="Stream_AC_Minigrid",
            mode = "online",
            config=vars(args) if args else {},
            name=f"Stream AC(λ)_{env_name}_lambda{lamda}_seed{seed}"
        )
    
    for t in range(1, total_steps+1):
        a = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        
        metrics = agent.update_params(s, a, r, s_prime, done, entropy_coeff, overshooting_info)
        s = s_prime

        if track:
            # Si l'épisode est fini, on ajoute le score aux métriques
            if done and "episode" in info:
                metrics["training/episodic_return"] = info['episode']['r'][0]
                metrics["training/episode_length"] = info['episode']['l'][0]
            
            # Envoi vers le site WandB
            wandb.log(metrics, step=t)
        
        if done:
            if debug:
                 # Le wrapper RecordEpisodeStatistics met le retour dans info['episode']['r']
                 # Parfois c'est info['episode']['r'][0] selon la version de gym, à vérifier
                 ep_return = info['episode']['r']
                 if isinstance(ep_return, (list, np.ndarray)): ep_return = ep_return[0]
                 print(f"Episodic Return: {ep_return}, Time Step {t}")
                 
                 returns.append(ep_return)
                 term_time_steps.append(t)
            
            s, _ = env.reset()

    env.close()

    if track:
        wandb.finish()
    
    # Sauvegarde
    save_dir = "data_stream_ac_{}_lr{}_gamma{}_lamda{}_entropy_coeff{}".format(env.spec.id, lr, gamma, lamda, entropy_coeff)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, f"seed_{seed}.pkl"), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream AC(λ) for Minigrid')
    parser.add_argument('--env_name', type=str, default='MiniGrid-Empty-5x5-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.0)
    parser.add_argument('--total_steps', type=int, default=200_000)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--kappa_policy', type=float, default=3.0)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overshooting_info', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--track', type=int, default=0)
    args = parser.parse_args()
    
    main(args.env_name, args.seed, args.lr, args.gamma, args.lamda, args.total_steps, args.entropy_coeff, args.kappa_policy, args.kappa_value, args.debug, args.overshooting_info, args.render, args.track)