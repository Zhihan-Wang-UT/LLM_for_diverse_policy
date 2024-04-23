import gym
import cleanup5agame
import numpy as np



if __name__ == "__main__":
    env = gym.make("cleanup-game-mini-reducedaction-agentlocationmask-sharereward-v0")
    # env.init_render()
    observation = env.reset(seed=0)
    observation_space = env.observation_space
    print(f"observation_space: {observation_space}")
    print(f"observation: {observation}")
    
    action_space = env.action_space
    print(f"action_space: {action_space}")
    action = action_space.sample()
    
    counter_x = 7
    counter_y = 5
    counter_x2 = 2
    counter_y2 = 1
    
    prev_action = action
    for i in range(12000):
        # 60% chance of repeating the previous action
        if np.random.rand() < 0.5:
            action = prev_action
        else:
            action = action_space.sample()
        
            
        if counter_x > 0:
            action[0] = 4
            counter_x -= 1
        elif counter_y > 0:
            action[0] = 1
            counter_y -= 1
        elif counter_x2 > 0:
            action[0] = 2
            counter_x2 -= 1
        elif counter_y2 > 0:
            action[0] = 3
            counter_y2 -= 1
        else:
            action[0] = 0
        
        print(f"action: {action}")
        observation, reward, done, _, info = env.step(action)
        print(f"observation: {observation}")
        assert observation[0, 6:12].sum() == 0
        assert observation[1, 0:6].sum() == 0
        
        
        assert reward[0] == reward[1]
        print(f"reward: {reward}")
        print(f"done: {done}")
        print(f"info: {info}")
        if done:
            env.reset() # this trigger the env to save episode.
            print(f"Episode finished after {i+1} timesteps")
            # break