import numpy as np
from tabular_envs import MountainCar, InvertedPendulum



def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / (duration)
    return max(end_e, start_e + slope*t)

def eps_greedy_policy(q, state, epsilon):
    num_actions = q.shape[-1]
    if np.random.uniform() <= epsilon:
        return np.random.choice(num_actions)
    else:
        return q[state].argmax()

def q_learning(env, num_episodes, epsilon=1, decay_epsilon=True, duration=1, alpha=0.1, gamma=0.9, debug=False, max_episode_steps=200):
    
    num_states = env.num_states
    num_actions = env.num_actions
    Q = np.zeros((num_states, num_actions))
    # Q-learning algorithm
    global_step = 0
    for episode in range(num_episodes):
        env.reset()
        state = env.get_state()
        Q_now = np.copy(Q)
        done = False
        sum_of_rewards = 0
        episode_steps = 0
        while True:
            action = eps_greedy_policy(Q, state, epsilon)
            _, reward, done, info = env.step(action)
            next_state = env.get_state()
            Q[state, action] = ((1-alpha)*Q[state, action]) + alpha*(reward + ~done * gamma * Q[next_state].max())
            state = next_state
            sum_of_rewards += reward
            if done: break
            global_step += 1
            episode_steps +=1 
            if episode_steps >= max_episode_steps:
                break
        if decay_epsilon:
            epsilon = linear_schedule(epsilon, 0.01, duration, episode)
        Q_after = np.copy(Q)
        diff = np.max(np.abs(Q_now - Q_after))
        if debug:
            debug_str = f"Episode: {episode+1}, epsilon: {epsilon}, return: {sum_of_rewards}, diff: {diff}"
            print(debug_str)
        if 0 < diff < 1e-5: break
    return Q

if __name__ == "__main__":
    # env = InvertedPendulum(state_discretization=64, action_discretization=5)
    env = MountainCar()
    ## Q-learning configs
    num_episodes = 10000
    eps_frac = 0.8
    epsilon = 1.0
    q_learning(env, num_episodes, epsilon=epsilon, decay_epsilon=True, duration=eps_frac*num_episodes, debug=True)
    # import pdb;pdb.set_trace()