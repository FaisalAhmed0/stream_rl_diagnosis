"""Base class for tabular environments.

Pure Python implementation converted from Cython.
Subclasses should implement the transitions and reward methods.
An example environment is provided in CliffwalkEnv
"""
import gym
import gym.spaces
import gymnasium
import numpy as np
from dataclasses import dataclass
from gym import Env
from typing import Optional, Tuple, Any, Dict

# from main import q_learning


@dataclass
class TimeStep:
    state: int
    reward: float
    done: bool


@dataclass
class PendulumState:
    theta: float
    thetav: float


@dataclass
class MountainCarState:
    pos: float
    vel: float


def sample_int(transitions):
    """Sample from a dictionary of transitions."""
    randnum = np.random.random()
    total = 0
    for ns, p in transitions.items():
        if (p + total) >= randnum:
            return ns
        total += p
    # Fallback to last state if floating point errors
    return ns


class TabularEnv:
    """Base class for tabular environments.

    States and actions are represented as integers ranging from
    [0, self.num_states) or [0, self.num_actions), respectively.

    Args:
      num_states: Size of the state space.
      num_actions: Size of the action space.
      initial_state_distribution: A dictionary from states to
        probabilities representing the initial state distribution.
    """

    def __init__(self, num_states, num_actions, initial_state_distribution):
        self._state = -1
        self.observation_space = gym.spaces.Discrete(num_states)
        self.observation_space = gym.spaces.Box(
            low = -1,
            high = 1,
            shape=(num_states, ))
        self.action_space = gym.spaces.Discrete(num_actions)
        self.num_states = num_states
        self.num_actions = num_actions
        self.initial_state_distribution = initial_state_distribution
        self._transition_map = {}
        # import pdb;pdb.set_trace()

    def transitions(self, state, action):
        """Computes transition probabilities p(ns|s,a).

        Args:
          state:
          action:

        Returns:
          A python dict from {next state: probability}.
          (Omitted states have probability 0)
        """
        return dict(self.transitions_cy(state, action))

    def transitions_cy(self, state, action):
        """Internal transition computation."""
        self._transition_map.clear()
        self._transition_map[state] = 1.0
        return self._transition_map

    def reward(self, state, action, next_state):
        """Return the reward

        Args:
          state:
          action: 
          next_state: 
        """
        return 0.0

    def observation(self, state):
        """Computes observation for a given state.

        Args:
          state: 

        Returns:
          observation: Agent's observation of state, conforming with observation_space
        """
        return state

    def step(self, action):
        """Simulates the environment by one timestep.

        Args:
          action: Action to take

        Returns:
          observation: Next observation
          reward: Reward incurred by agent
          done: A boolean indicating the end of an episode
          info: A debug info dictionary.
        """
        infos = {'state': self.get_state()}
        ts = self.step_state(action)
        nobs = self.observation(ts.state)
        return nobs, ts.reward, ts.done, infos

    def step_state(self, action):
        """Simulates the environment by one timestep, returning the state id
        instead of the observation.

        Args:
          action: Action taken by the agent.

        Returns:
          TimeStep with state, reward, and done flag
        """
        transitions = self.transitions_cy(self._state, action)
        next_state = sample_int(transitions)
        reward = self.reward(self.get_state(), action, next_state)
        self.set_state(next_state)
        return TimeStep(next_state, reward, False)

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns:
          observation (object): The agent's initial observation.
        """
        initial_state = self.reset_state()
        return self.observation(initial_state)

    def reset_state(self):
        """Resets the state of the environment and returns an initial state.

        Returns:
          state: The agent's initial state
        """
        initial_states = list(self.initial_state_distribution.keys())
        initial_probs = list(self.initial_state_distribution.values())
        initial_state = np.random.choice(initial_states, p=initial_probs)
        self.set_state(initial_state)
        return initial_state

    def transition_matrix(self):
        """Constructs this environment's transition matrix.

        Returns:
          A dS x dA x dS array where the entry transition_matrix[s, a, ns]
          corresponds to the probability of transitioning into state ns after taking
          action a from state s.
        """
        ds = self.num_states
        da = self.num_actions
        transition_matrix = np.zeros((ds, da, ds))
        for s in range(ds):
            for a in range(da):
                transitions = self.transitions_cy(s, a)
                for next_s, prob in transitions.items():
                    transition_matrix[s, a, next_s] = prob
        return transition_matrix

    def reward_matrix(self):
        """Constructs this environment's reward matrix.

        Returns:
          A dS x dA x dS numpy array where the entry reward_matrix[s, a, ns]
          reward given to an agent when transitioning into state ns after taking
          action s from state s.
        """
        ds = self.num_states
        da = self.num_actions
        rew_matrix = np.zeros((ds, da, ds))
        for s in range(ds):
            for a in range(da):
                for ns in range(ds):
                    rew_matrix[s, a, ns] = self.reward(s, a, ns)
        return rew_matrix

    def set_state(self, state):
        """Set the agent's internal state."""
        self._state = state

    def get_state(self):
        """Return the agent's internal state."""
        return self._state

    def render(self):
        """Render the current state of the environment."""
        print(self.get_state())


class CliffwalkEnv(TabularEnv):
    """An example env where an agent can move along a sequence of states. There is
    a chance that the agent may jump back to the initial state.

    Action 0 moves the agent back to start, and action 1 to the next state.
    The agent only receives reward in the final state and is forced to move back to the start.

    Args:
      num_states: Number of states 
      transition_noise: A float in [0, 1] representing the chance that the
        agent will be transported to the start state.
    """

    def __init__(self, num_states=16, transition_noise=0.01):
        super(CliffwalkEnv, self).__init__(num_states, 2, {0: 1.0})
        self.transition_noise = transition_noise

    def transitions_cy(self, state, action):
        self._transition_map.clear()
        if action == 0:
            self._transition_map[0] = 1.0
        else:
            if state == self.num_states - 1:
                self._transition_map[0] = 1.0
            else:
                self._transition_map[0] = self.transition_noise
                self._transition_map[state + 1] = 1.0 - self.transition_noise
        return self._transition_map

    def reward(self, state, action, next_state):
        if state == self.num_states - 1 and action == 1:
            return 1.0
        else:
            return 0.0


class RandomTabularEnv(TabularEnv):
    def __init__(self, num_states=3, num_actions=2, transitions_per_action=4, 
                 seed=0, self_loop=False):
        super(RandomTabularEnv, self).__init__(num_states, num_actions, {0: 1.0})

        rng = np.random.RandomState(seed)
        
        rewards = np.zeros((num_states, num_actions))
        reward_state = rng.randint(1, num_states)
        rewards[reward_state, :] = 1.0
        self._reward_matrix = rewards

        transition_matrix = np.zeros((num_states, num_actions, num_states), dtype=np.float64)
        scores = rng.rand(num_states, num_actions, num_states).astype(np.float64)
        scores[:, :, reward_state] *= 0.999  # reduce chance of link to goal

        for s in range(num_states):
            for a in range(num_actions):
                top_states = np.argsort(scores[s, a, :])[-transitions_per_action:]
                for ns in top_states:
                    transition_matrix[s, a, ns] = 1.0 / float(transitions_per_action)
            if self_loop:
                for ns in range(num_states):
                    transition_matrix[s, 0, ns] = 0.0
                transition_matrix[s, 0, s] = 1.0
        transition_matrix = transition_matrix / np.sum(transition_matrix, axis=2, keepdims=True)
        self._transition_matrix = transition_matrix

    def transitions_cy(self, state, action):
        self._transition_map.clear()
        for ns in range(self.num_states):
            prob = self._transition_matrix[state, action, ns]
            if prob > 0:
                self._transition_map[ns] = prob
        return self._transition_map

    def reward(self, state, action, next_state):
        return self._reward_matrix[state, action]


class InvertedPendulum(TabularEnv):
    """ 
    Dynamics and reward are based on OpenAI gym's implementation of Pendulum-v0
    """

    def __init__(self, state_discretization=64, action_discretization=5):
        self._state_disc = state_discretization
        self._action_disc = action_discretization
        self.max_vel = 4.0
        self.max_torque = 3.0

        self.action_map = np.linspace(-self.max_torque, self.max_torque, num=action_discretization)
        self.state_map = np.linspace(-np.pi, np.pi, num=state_discretization)
        self._state_min = -np.pi
        self._state_step = (2 * np.pi) / state_discretization
        self.vel_map = np.linspace(-self.max_vel, self.max_vel, num=state_discretization)
        self._vel_min = -self.max_vel
        self._vel_step = (2 * self.max_vel) / state_discretization

        initial_state = self.to_state_id(PendulumState(-np.pi / 4, 0))
        super(InvertedPendulum, self).__init__(
            state_discretization * state_discretization, 
            action_discretization, 
            {initial_state: 1.0}
        )
        # self.observation_space = gym.spaces.Box(
        #     low=np.array([0, 0, -self.max_vel]), 
        #     high=np.array([1, 1, self.max_vel]), 
        #     dtype=np.float32
        # )

    def transitions_cy(self, state, action):
        self._transition_map.clear()

        # pendulum dynamics
        g = 10.0
        m = 1.0
        l = 1.0
        dt = 0.05
        torque = self.action_to_torque(action)
        pstate = self.from_state_id(state)

        newvel = pstate.thetav + (-3 * g / (2 * l) * np.sin(pstate.theta + np.pi) + 
                                  3.0 / (m * l ** 2) * torque) * dt
        newth = pstate.theta + newvel * dt
        newvel = max(min(newvel, self.max_vel - 1e-8), -self.max_vel)
        if newth < -np.pi:
            newth += 2 * np.pi
        if newth >= np.pi:
            newth -= 2 * np.pi
        next_state = self.to_state_id(PendulumState(newth, newvel))

        self._transition_map[next_state] = 1.0
        return self._transition_map

    def reward(self, state, action, next_state):
        torque = self.action_to_torque(action)
        pstate = self.from_state_id(state)
        # OpenAI gym reward
        cost = pstate.theta ** 2 + 0.1 * (pstate.thetav ** 2) + 0.001 * (torque ** 2)
        max_cost = np.pi ** 2 + 0.1 * self.max_vel ** 2 + 0.001 * (self.max_torque ** 2)
        return (-cost + max_cost) / max_cost

    def observation(self, state):
        pstate = self.from_state_id(state)
        return np.array([np.cos(pstate.theta), np.sin(pstate.theta), pstate.thetav], dtype=np.float32)

    def from_state_id(self, state):
        th_idx = state % self._state_disc
        vel_idx = state // self._state_disc
        th = self._state_min + self._state_step * th_idx
        thv = self._vel_min + self._vel_step * vel_idx
        return PendulumState(th, thv)

    def to_state_id(self, pend_state):
        th = pend_state.theta
        thv = pend_state.thetav
        # round
        th_round = int(np.floor((th - self._state_min) / self._state_step))
        th_vel = int(np.floor((thv - self._vel_min) / self._vel_step))
        return th_round + self._state_disc * th_vel

    def action_to_torque(self, action):
        return self.action_map[action]

    def render(self):
        pend_state = self.from_state_id(self.get_state())
        th = pend_state.theta
        thv = pend_state.thetav
        print('(%f, %f) = %d' % (th, thv, self.get_state()))


class MountainCar(TabularEnv):
    """ 
    Dynamics and reward are based on OpenAI gym's implementation of MountainCar-v0
    """

    def __init__(self, posdisc=64, veldisc=64, action_discretization=5):
        self._pos_disc = posdisc
        self._vel_disc = veldisc
        self._action_disc = action_discretization
        self.max_vel = 0.06  # gym 0.07
        self.min_vel = -self.max_vel
        self.max_pos = 0.55  # gym 0.6
        self.min_pos = -1.2  # gym -1.2
        self.goal_pos = 0.5

        self._state_step = (self.max_pos - self.min_pos) / self._pos_disc
        self._vel_step = (self.max_vel - self.min_vel) / self._vel_disc

        initial_state = self.to_state_id(MountainCarState(-0.5, 0))
        print(initial_state)
        super(MountainCar, self).__init__(self._pos_disc * self._vel_disc, 3, {initial_state: 1.0})
        # self.observation_space = gym.spaces.Box(
        #     low=np.array([self.min_pos, -self.max_vel]), 
        #     high=np.array([self.max_pos, self.max_vel]), 
        #     dtype=np.float32
        # )
        # import pdb;pdb.set_trace()

    def transitions_cy(self, state, action):
        self._transition_map.clear()
        state_vec = self.from_state_id(state)
        position, velocity = state_vec.pos, state_vec.vel
        for _ in range(3):
            velocity += (action - 1) * 0.001 + np.cos(3 * position) * (-0.0025)
            velocity = max(min(velocity, self.max_vel - 1e-8), self.min_vel)
            position += velocity
            position = max(min(position, self.max_pos - 1e-8), self.min_pos)
            if (position == self.min_pos and velocity < 0):
                velocity = 0
        next_state = self.to_state_id(MountainCarState(position, velocity))
        self._transition_map[next_state] = 1.0
        return self._transition_map

    def reward(self, state, action, next_state):
        state_vec = self.from_state_id(state)
        if state_vec.pos >= self.goal_pos:
            return 1.0
        return 0.0

    def observation(self, state):
        pstate = self.from_state_id(state)
        return np.array([pstate.pos, pstate.vel], dtype=np.float32)

    def from_state_id(self, state):
        th_idx = state % self._pos_disc
        vel_idx = state // self._pos_disc
        th = self.min_pos + self._state_step * th_idx
        thv = self.min_vel + self._vel_step * vel_idx
        return MountainCarState(th, thv)

    def to_state_id(self, state_vec):
        pos = state_vec.pos
        vel = state_vec.vel
        # round
        pos_idx = int(np.floor((pos - self.min_pos) / self._state_step))
        vel_idx = int(np.floor((vel - self.min_vel) / self._vel_step))
        return pos_idx + self._pos_disc * vel_idx

    def render(self):
        state_vec = self.from_state_id(self.get_state())
        x1 = state_vec.pos
        x2 = state_vec.vel
        print('(%f, %f) = %d' % (x1, x2, self.get_state()))


class GymnasiumTabularWrapper(gymnasium.Env):
    """Gymnasium wrapper for TabularEnv environments.
    
    This wrapper converts TabularEnv instances to follow the standard Gymnasium API.
    It handles the conversion from the old Gym API (done flag) to the new Gymnasium API
    (terminated and truncated flags).
    
    Args:
        tabular_env: An instance of TabularEnv or its subclasses.
    """
    
    def __init__(self, tabular_env: TabularEnv, max_episode_steps=200):
        super().__init__()
        self.tabular_env = tabular_env
        self.max_episode_steps = max_episode_steps
        # create a new spec object
        from gymnasium.envs.registration import EnvSpec
        self.spec = EnvSpec(
                id="gymnasium_tabular_env",
                max_episode_steps=max_episode_steps
                # entry_point=env.spec.entry_point,
                # max_episode_steps=env.spec.max_episode_steps,
                # reward_threshold=env.spec.reward_threshold,
                # nondeterministic=env.spec.nondeterministic,
                # kwargs=env.spec.kwargs,
            )
        
        # if tabular_env.spec is not None:
        #     self.spec = EnvSpec(
        #         id="gymnasium_tabular_env",
        #         # entry_point=env.spec.entry_point,
        #         # max_episode_steps=env.spec.max_episode_steps,
        #         # reward_threshold=env.spec.reward_threshold,
        #         # nondeterministic=env.spec.nondeterministic,
        #         # kwargs=env.spec.kwargs,
        #     )
        # else:
        #     # if env has no spec, just create a minimal one
        #     self.spec = EnvSpec(id=None)
        
        # Convert spaces to gymnasium spaces
        if isinstance(tabular_env.observation_space, gym.spaces.Discrete):
            self.observation_space = gymnasium.spaces.Discrete(tabular_env.observation_space.n)
        elif isinstance(tabular_env.observation_space, gym.spaces.Box):
            self.observation_space = gymnasium.spaces.Box(
                low=tabular_env.observation_space.low,
                high=tabular_env.observation_space.high,
                shape=tabular_env.observation_space.shape,
                dtype=tabular_env.observation_space.dtype
            )
        else:
            # Fallback: try to convert directly
            self.observation_space = tabular_env.observation_space
        
        if isinstance(tabular_env.action_space, gym.spaces.Discrete):
            self.action_space = gymnasium.spaces.Discrete(tabular_env.action_space.n)
        elif isinstance(tabular_env.action_space, gym.spaces.Box):
            self.action_space = gymnasium.spaces.Box(
                low=tabular_env.action_space.low,
                high=tabular_env.action_space.high,
                shape=tabular_env.action_space.shape,
                dtype=tabular_env.action_space.dtype
            )
        else:
            # Fallback: try to convert directly
            self.action_space = tabular_env.action_space
        
        # Metadata for Gymnasium compatibility
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
        self.render_mode = None
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Run one timestep of the environment's dynamics.
        
        Args:
            action: An action provided by the agent.
            
        Returns:
            observation: Agent's observation of the current environment.
            reward: Amount of reward returned after previous action.
            terminated: Whether the agent reaches a terminal state (episode ends normally).
            truncated: Whether the truncation condition outside the scope of the MDP is satisfied.
            info: Contains auxiliary diagnostic information.
        """
        obs, reward, done, info = self.tabular_env.step(action)
        obs = np.zeros(self.observation_space.shape[0])
        obs[self.tabular_env.get_state()] = 1.
        self.t += 1
        if self.t > self.max_episode_steps:
            terminated = done = True
        else:
            terminated  = done = False
        # convert the observation to onehot vector
        
        # Convert done flag to terminated/truncated
        # In the old API, done=True means episode ended (could be either terminated or truncated)
        # We'll treat it as terminated by default, but this can be customized per environment
        terminated = done
        truncated = False
        
        # Update info to be compatible with Gymnasium
        if not isinstance(info, dict):
            info = {}
        
        return obs, reward, terminated, truncated, info
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Resets the environment to an initial state and returns an initial observation.
        
        Args:
            seed: The seed that is used to initialize the environment's PRNG.
            options: Additional information to specify how the environment is reset.
            
        Returns:
            observation: Observation of the initial state.
            info: Dictionary containing auxiliary information.
        """
        if seed is not None:
            np.random.seed(seed)
        self.t = 0
        
        obs = self.tabular_env.reset()
        obs = np.zeros(self.observation_space.shape[0])
        obs[self.tabular_env.get_state()] = 1.
        
        # Return observation and info dict (Gymnasium API)
        info = {}
        if options is not None:
            info.update(options)
        
        return obs, info
    
    def render(self):
        """Render the environment."""
        return self.tabular_env.render()
    
    def close(self):
        """Clean up the environment's resources."""
        pass
    
    # Expose TabularEnv methods for backward compatibility
    def get_state(self):
        """Return the agent's internal state."""
        return self.tabular_env.get_state()
    
    def set_state(self, state):
        """Set the agent's internal state."""
        return self.tabular_env.set_state(state)
    
    def transition_matrix(self):
        """Constructs this environment's transition matrix."""
        return self.tabular_env.transition_matrix()
    
    def reward_matrix(self):
        """Constructs this environment's reward matrix."""
        return self.tabular_env.reward_matrix()


# if __name__ == "__main__":
#     # print("hello envs")
#     env = MountainCar()
#     env = GymnasiumTabularWrapper(env)
#     import pdb;pdb.set_trace()
#     ## Q-learning configs
#     num_episodes = 100

#     q_learning(env, )
#     import pdb;pdb.set_trace()