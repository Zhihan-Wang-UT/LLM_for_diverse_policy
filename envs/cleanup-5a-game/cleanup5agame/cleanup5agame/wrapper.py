import gym
import numpy as np
from gym.spaces import Box

class TimeStepWrapper(gym.Wrapper):
    def __init__(self, env, max_timesteps):
        super().__init__(env)
        self.max_timesteps = max_timesteps
        self.timestep = 0
        
        # Extract the high value from the original observation space
        original_high = env.observation_space.high
        if np.isscalar(original_high):
            high_value = original_high
        else:
            # Assuming the high values are uniform across the observation space,
            # we take one value as they should all be the same in this context.
            high_value = original_high.flat[0]

        original_shape = env.observation_space.shape  # (2, 34)
        new_shape = (original_shape[0], original_shape[1] + 1)  # (2, 35)

        # Create a new observation space that includes the additional timestep
        # Convert everything to float32 to accommodate the floating-point timestep
        self.observation_space = Box(
            low=np.zeros(new_shape, dtype=np.float32),
            high=np.full(new_shape, high_value, dtype=np.float32),  # Use extracted high value
            dtype=np.float32
        )
        # Set the last column (timestep) high value to 1 to reflect its actual range
        self.observation_space.high[:, -1] = 1

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        self.timestep += 1
        
        # Ensure obs is a NumPy array before proceeding
        if isinstance(obs, tuple):
            obs = obs[0]  # Assuming the first element is the observation array; adjust as needed
        
        normalized_timestep = np.full((2, 1), self.timestep / self.max_timesteps, dtype=np.float32)
        # Now, safely convert obs to float32 and concatenate
        new_obs = np.concatenate((obs.astype(np.float32), normalized_timestep), axis=1)
        return new_obs, reward, done, done, info

    def reset(self, **kwargs):
        self.timestep = 0
        obs_info, _ = self.env.reset(**kwargs)
        
        # If obs is returned alongside other info in a tuple, extract it
        if isinstance(obs_info, tuple):
            obs = obs_info[0]  # Adjust based on the actual structure
        else:
            obs = obs_info  # Directly use obs_info as obs
        
        initial_timestep = np.zeros((2, 1), dtype=np.float32)
        new_obs = np.concatenate((obs.astype(np.float32), initial_timestep), axis=1)
        return new_obs, {}



# The original reward is [r1, r2], this wrapper will make it [(r1 + r2)/2, r2]
class SharedEgoRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, rewards, done, _, info = self.env.step(action)
        # Calculate the average of both rewards
        avg_reward = np.mean(rewards)
        # Modify the first agent's reward to be the average, second agent's reward remains the same
        modified_rewards = [avg_reward, rewards[1]]
        return obs, modified_rewards, done, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    
    
    
# The original reward is [r1, r2], this wrapper will make it [(r1 + r2)/2, (r1 + r2)/2]
class FullyCooperativeRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, rewards, done, _, info = self.env.step(action)
        # Calculate the average of both rewards and apply it to both agents
        avg_reward = np.mean(rewards)
        modified_rewards = [avg_reward, avg_reward]
        return obs, modified_rewards, done, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
# The wraper should wrap the first agents' second 6 observation row 0, idx 6-11 to 0
# and the second agent's first 6 observation to zero (row1, index 0-5)

# so that the agent only see their own location and cannot see other agents' location
class AgentLocationMaskWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # The observation space remains the same size, so no need to modify it.
        # However, we'll be modifying the content of observations in step and reset methods.

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        
        # Mask the specified parts of the observation for each agent
        # For the first agent: mask indices 6-11
        # observation = obs[0]
        obs[0, 6:12] = 0
        # For the second agent: mask indices 0-5
        obs[1, 0:6] = 0
        
        return obs, reward, done, done, info

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        
        # Apply the same masking as in the step method
        obs[0, 6:12] = 0
        obs[1, 0:6] = 0
        
        return obs, {}
