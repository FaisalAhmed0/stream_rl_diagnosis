import numpy as np
from tabular_envs import MountainCar, InvertedPendulum, CliffwalkEnv
from collections import deque
from dataclasses import dataclass
import random
import tyro

@dataclass
class Args:
    env_name: str = "MountainCar"
    # Tablular q-learning configs
    num_episodes: int = 100_000
    expl_frac: float = 0.5
    gamma: float = 0.9
    seed: int = 0
    epsilon_start: float = 1.0
    epslion_end: float = 0.01
    alpha_start: float = 1.0
    alpha_end: float = 0.0001
    decay_epsilon: bool = True
    decay_alpha: bool = True
    debug: bool = False


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)


def save_array(array, filepath):
    """
    Save a numpy array to a human-readable text file.
    
    Args:
        array: numpy array to save
        filepath: path to save the file (e.g., 'q_table.txt')
    """
    # Save shape information in a header comment
    shape_str = ','.join(map(str, array.shape))
    
    # For 1D arrays, save directly
    if array.ndim == 1:
        np.savetxt(filepath, array, header=f'Shape: {shape_str}', comments='# ')
    # For 2D arrays, save directly with shape info
    elif array.ndim == 2:
        np.savetxt(filepath, array, header=f'Shape: {shape_str}', comments='# ')
    # For higher dimensional arrays, flatten and save with shape info
    else:
        flattened = array.flatten()
        np.savetxt(filepath, flattened, header=f'Shape: {shape_str}', comments='# ')


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


def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / (duration)
    return max(end_e, start_e + slope*t)

def eps_greedy_policy(q, state, epsilon):
    num_actions = q.shape[-1]
    if np.random.uniform() < epsilon:
        return np.random.choice(num_actions)
    else:
        max_actions = np.flatnonzero(q[state] == q[state].max())
        return np.random.choice(max_actions)


def evaluation(env, q, num_evals=5, max_episode_steps=200):
    all_returns = []
    for eval in range(num_evals):
        env.reset()
        env.reset_state()
        state = env.get_state()
        done = False
        sum_of_rewards = 0.0
        t = 0
        while True:
            action = np.argmax(q[state])
            _, reward, done, info = env.step(action)
            sum_of_rewards += reward
            state = env.get_state()
            t += 1
            if done or t>=max_episode_steps:
                break
        all_returns.append(sum_of_rewards)
    return np.mean(all_returns), np.std(all_returns)



def q_learning(env, num_episodes, epsilon=1, epsilon_end=0.01,
                decay_epsilon=True, decay_alpha=True, 
                duration=1, alpha=0.1, alpha_end=0.1, gamma=0.9, debug=False, 
                max_episode_steps=200,eval_freq=1000):
    
    num_states = env.num_states
    num_actions = env.num_actions
    starting_epsilon = epsilon
    staring_alpha = alpha
    Q = np.zeros((num_states, num_actions))
    # Q-learning algorithm
    global_step = 0
    max_return = 0
    max_diff = 0
    diff_list = []
    for episode in range(num_episodes):
        env.reset()
        env.reset_state()
        state = env.get_state()
        Q_now = np.copy(Q)
        done = False
        sum_of_rewards = 0.0
        episode_steps = 0.0
        new_good_reward = False
        
        while True:
            action = eps_greedy_policy(Q, state, epsilon)
            _, reward, done, info = env.step(action)
            sum_of_rewards += reward
            next_state = env.get_state()
            Q[state, action] = ((1-alpha)*Q[state, action]) + alpha*(reward + (1 - int(done)) * gamma * Q[next_state].max())
            # if debug and reward > 0:
            #     print("========= positive reward ==========")
            #     print(Q[state, action], sum_of_rewards)
            #     print(state)
            #     print(Q[2073])
            #     new_good_reward = True
            state = next_state
            if done:
                # print(sum_of_rewards)
                break
            global_step += 1
            episode_steps +=1 
            if episode_steps >= max_episode_steps:
                # print(sum_of_rewards)
                break
        if decay_epsilon:
            epsilon = linear_schedule(starting_epsilon, epsilon_end, duration, episode+1)
        if decay_alpha:
            alpha = linear_schedule(staring_alpha, alpha_end, duration, episode+1)
        Q_after = np.copy(Q)
        # if new_good_reward:
        #     import pdb;pdb.set_trace()
        #     diff = np.max(np.abs(Q_now - Q_after))
        max_return = max(sum_of_rewards, max_return)
        diff = np.max(np.abs(Q_now - Q_after))
        diff_list.append(diff)
        mean_diff = np.mean(diff_list)
        if debug:
            debug_str = f"Episode: {episode+1}, alpha: {alpha:.5f}, epsilon: {epsilon:.5f}, return: {sum_of_rewards:.5f}, max return: {max_return:.5f}, diff: {diff:.5f}, mean diff: {mean_diff:.5f}"
            print(debug_str)
        if (episode+1) % eval_freq == 0:
            eval_return, eval_std = evaluation(env, Q, num_evals=5, max_episode_steps=max_episode_steps)
            eval_str = f"eval_return:{eval_return:.5f}, eval_std:{eval_std:.5f}"
            print(eval_str)
    return Q

if __name__ == "__main__":
    args = tyro.cli(Args)
    env = eval(args.env_name)()
    seed_everything(args.seed)
    q = q_learning(env, args.num_episodes, epsilon=args.epsilon_start, epsilon_end=args.epslion_end,
                    decay_epsilon=args.decay_epsilon, decay_alpha=args.decay_alpha, alpha=args.alpha_start, 
                    duration=args.expl_frac*args.num_episodes, debug=args.debug)
    print(q)
    print(args.env_name)
    save_array(q, f"{args.env_name}_q_star_table_seed_{args.seed}.txt")
    # import pdb;pdb.set_trace()
    # import pdb;pdb.set_trace()