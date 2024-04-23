from typing import List
import numpy as np
import gym
import uuid

from meltingpot.configs.substrates import clean_up_mini
from meltingpot.utils.substrates import builder
import namespace
import numpy as np
from ml_collections import config_dict

# def actions_to_onehot(num_actions, actions):
#     """
#     Transfer actions to onehot representation
#     :param num_actions: list of number of actions of each agent
#     :param actions: list of actions (int) for each agent
#     :return: onehot representation of actions
#     """
#     onehot = [[0] * num_action for num_action in num_actions]
#     for ag, act in enumerate(actions):
#         onehot[ag][act] = 1
#     return onehot

ASCII_MAP = """
WWWWWWWWW
WFHFf===W
WHFf   PW
WHH> PPPW
WH>  PPPW
WTTTTTTTW
WBBBBBBBW
WWWWWWWWW
"""

# This is a simplified version of the clean up game
# 
class CleanUpReducedActionGame(gym.Env):
    def __init__(self, ascii_map=ASCII_MAP, n_agents=2, last_action_state=False, mode=None):
        # n_agents = 2 is the total number of agents in the environment
        # if mode = "adhoc-eval" then the game is in adhoc mode, which means that we control
        # one agent and the other agent is controlled by a policy (heuristics based for now)
        assert last_action_state == False
        assert mode in [None, 'adhoc-eval']
        self.mode = mode
        self.n_agents = n_agents
        
        # basic steps of creating the environment
        env_module = clean_up_mini
        env_config = env_module.get_config()
        with config_dict.ConfigDict(env_config).unlocked() as env_config:
            env_config.default_player_roles = ("default",) * n_agents # default 2 players
            roles = env_config.default_player_roles
            env_config.lab2d_settings = env_module.build(roles, env_config)
        
        # config_overrides = {} # this game is 2 players by default
        # env_config.lab2d_settings.update(config_overrides)
        
        player_count = env_config.lab2d_settings.get('numPlayers', 1)
        player_prefixes = [f'{i+1}' for i in range(player_count)]
        
        print(f'Running an episode with {player_count} players: {player_prefixes}.')
        
        # self.max_episode_length = 10000
        # self.t = 0
        self.env = builder.builder(**env_config)
        self.env.reset()
        self.reward_keys = [f"{i}.REWARD" for i in range(1, n_agents+1)]
        
        # buildin the observation converter
        self.observation_converter = CleanUpLayerToFlatVectorConverter(ascii_map, n_agents)
        self.observation_space = self.observation_converter.observation_space
        
        # build action proxy
        self.action_proxy = ActionProxy(ascii_map, n_agents)
        self.action_space = self.action_proxy.action_space
        
        self.render_video = False
        
        self.mode = mode
        if self.mode == "adhoc-eval":
            self.teammate_policy = None
            # self.teammate_obs = None
            # self.total_episodes = total_episodes
            self.episodes_elapsed = 0
            raise NotImplementedError


    def init_render(self, outfile_prefix=None):
        if self.render_video:
            return
        self.frames = []
        self.render_video = True
        self.nthvideo = 0
        # get uuid
        if outfile_prefix:
            self.outfile_prefix = outfile_prefix
        else:
            self.outfile_prefix = "./outputs/" + str(uuid.uuid4())
            import os
            # if the folder does not exist, create it
            if not os.path.exists("./outputs/"):
                os.makedirs("./outputs/")
            
    def render(self):
        self.init_render()
    
    def _make_obs(self):
        # every agent shares the same observation
        observation = self.env.observation()
        flat_obs = self.observation_converter.convert(observation)
        
        obs = np.array([flat_obs] * self.n_agents)
        self.observation = obs
        return obs
        
        # if self.last_action_state:
        #     return np.array([np.array(self.last_actions)] * self.n_agents)
        # else:
        #     return np.array([np.array(self.n_agents * [0])] * self.n_agents)
        
        

    def step(self, actions):
        '''
        actions: list of actions, each action is a number from 0 to 4
        '''
        if self.mode == "adhoc-eval":
            actions = [actions, self.teammate_policy.step(self._make_obs()[-1])]
            raise NotImplementedError
        
        actions = self._action_proxy(actions)
        action_dict = self.get_action_dict(actions)
        timestep = self.env.step(action_dict)
        rewards = [timestep.observation[key].item() for key in self.reward_keys]
        # https://github.com/google-deepmind/dm_env/blob/master/docs/index.md
        # there should be another reference whihc I canot find
        done = timestep.last() # this is what they (dmlab2d to gym wrapper by deepmind) used
        # done = self.env._env._reset_next_step # alternative
        # assert done == timestep.last()
        
        if self.render_video:
            self.frames.append(self._make_render_frame())
        
        return self._make_obs(), rewards, done, done, {}
    
    # this is translate the 5 state action to 5x3x2 state action required by the enviornment
    def _action_proxy(self, actions):
        actions = self.action_proxy.convert(actions, self._make_obs()[0]) # global observation for every agent, so we just use the first one
        return actions
        
        
    def reset(self, seed=None):
        self.env.reset()
        # self.env.reset(seed=seed)
        if self.render_video:
            if len(self.frames) > 0:
                self.save_to_video()
                self.frames = []
        foo = self._make_obs()
        return foo, {}
    
    def get_action_space(self, n_agents):
        # process the action space each agent has 4 action dimensions
        if self.mode == "adhoc-eval":
            raise NotImplementedError
        else:
            # for original game, turn is -1,0,1, actions are move(0-4), turn(-1-1), fireZap(0,1), fireClean(0,1)
            # we disregard the fireZap action
            # turn is -1,0,1, we need to make action from 0,1,2 to -1,0,1 in step
            # HOWEVER we reduce the action space to 5 for cleanup, by 
            # 1) combining movement and turning (if the 
            #    agent is not facing the absolute direction it 
            #    want to move in, it turn to the direction instead) 
            # 2) automate cleaning: if the agent can clean 
            #    something, it will automatically fireClean
            return gym.spaces.MultiDiscrete([5] * n_agents) # move
        
        
        move_actions_space = gym.spaces.Discrete(5)
        turn_actions_space = gym.spaces.Box(low=-1, high=1, shape=(), dtype=np.int32)
        fireZap_actions_space = gym.spaces.Discrete(2)
        fireClean_actions_space = gym.spaces.Discrete(2)
        action_space = gym.spaces.Tuple(
            [move_actions_space, 
                turn_actions_space, 
                fireZap_actions_space,
                fireClean_actions_space] * n_agents)
        return action_space
    
    # convert our vector of action to a dictionary of actions which dmlab2d tabes
    def get_action_dict(self, actions):
        assert len(actions) == self.n_agents * 3 # each agent has 3 action dimension
        action_name = ["move", "turn", "fireZap", "fireClean"]
        action_dict = {}
        for i in range(self.n_agents):
            action_dict[f'{i+1}.move'] = actions[i*3]
            action_dict[f'{i+1}.turn'] = actions[i*3+1] # turn is -1,0,1
            action_dict[f'{i+1}.fireZap'] = 0 # no zap
            action_dict[f'{i+1}.fireClean'] = actions[i*3+2]
        return action_dict
    
    # return the rgb observation for rendering, assume this function is completed
    def _make_render_frame(self):
        observation = self.env.observation()
        return observation['WORLD.RGB']
    
    def save_to_video(self):
        if len(self.frames) > 0:
            output_file = f'{self.outfile_prefix}_{self.nthvideo}.mp4'
            save_frames_to_video(self.frames, output_file)
            print(f"Video saved to ./outputs/{output_file}")
            self.nthvideo += 1


class ActionProxy:
    def __init__(self, ascii_map, n_agents=2) -> None:
        self.n_agents = n_agents
        lines = ascii_map.strip().split("\n")
        ascii_array = np.array([list(line) for line in lines])
        
        self.max_x = len(lines[0])
        self.max_y = len(lines)
        
        # water_loc are corresponds to the 'H' and 'F' in the ASCII_MAP
        water_indices_w = np.argwhere(ascii_array == 'H')
        water_indices_f = np.argwhere(ascii_array == 'F')
        water_locs_np = np.concatenate([water_indices_w, water_indices_f], axis=0)
        assert water_locs_np.shape[0] == len(water_indices_w) + len(water_indices_f)
        self.water_locs = water_locs_np
        
        wall_indices = np.argwhere(ascii_array == 'W')
        self.wall_locs = wall_indices
        self.action_space = gym.spaces.MultiDiscrete([5] * n_agents) # move
        
        
    # return a tuple of (move, turn) for the agent
    # the this convert the 5 state action_direction (absolute move/turn) 
    # to 5x3 state action (relative move, turn) required by the enviornment
    # curreing facing from 12 o'clock, clockwise 1,2,3,4
    # action 0 is no action
    # action_direction 1,2,3,4 is up, right, down, left
    def move_or_turn(self, current_facing, action_direction): 
        if action_direction == 0:
            return (0, 0)
        if current_facing == action_direction:
            return (1, 0)
        
        # if agent is not facing the direction it wants to move in, it turn to the direction instead
        diff = action_direction - current_facing
        if diff < 0:
            diff += 4
        if diff == 1:
            return (0, 1)
        if diff == 2: # facing opposite direction, make cw turn
            return (0, 1)
        if diff == 3:
            return (0, -1)
        
    def predict_move_or_turn_result_pos(self, action_direction, move_or_turn, current_pos):
        move = move_or_turn[0]
        y,x = current_pos
        filterf = lambda x: x[0] >= 0 and x[0] < self.max_y and x[1] >= 0 and x[1] < self.max_x
        tmp = None
        if not move:
            return current_pos
        
        if action_direction == 1:
            tmp = (y-move, x)
        elif action_direction == 2:
            tmp = (y, x+move)
        elif action_direction == 3:
            tmp = (y+move, x)
        elif action_direction == 4:
            tmp = (y, x-move)
        
        # check tmp not in wall
        if tmp in self.wall_locs:
            return current_pos
        # check tmp not out of bound
        if filterf(tmp):
            return tmp
        # if out of bound, return current pos
        return current_pos
    
    def predict_move_or_turn_result_facing(self, move_or_turn, current_facing):
        tmp = current_facing + move_or_turn[1]
        if tmp < 1:
            tmp += 4
        if tmp > 4:
            tmp -= 4
        return tmp
        
    def should_fire_clean(self, agenty, agentx, agentfacing):
        # grids that would be covered if the agent fireClean
        could_be_cleaned = []
        # suppose agent is facing up, at position (y,x) agentfacing = 1 
        # it cleans (y,x-1), (y-1,x-1), (y-2,x-1),
        #           (y,x+1), (y-1,x+1), (y-2,x+1), 
        #           (y-1,x), (y-2,x),   (y-3,x)
        if agentfacing == 1: # it is facing the negative y direction
            could_be_cleaned = [(agenty, agentx-1), (agenty-1, agentx-1), (agenty-2, agentx-1),
                                (agenty, agentx+1), (agenty-1, agentx+1), (agenty-2, agentx+1),
                                (agenty-1, agentx), (agenty-2, agentx), (agenty-3, agentx)]
        elif agentfacing == 2: # it is facing the positive x direction
            could_be_cleaned = [(agenty-1, agentx), (agenty-1, agentx+1), (agenty-1, agentx+2),
                                (agenty+1, agentx), (agenty+1, agentx+1), (agenty+1, agentx+2),
                                (agenty, agentx+1), (agenty, agentx+2), (agenty, agentx+3)]
        elif agentfacing == 3: # it is facing the positive y direction
            could_be_cleaned = [(agenty, agentx-1), (agenty+1, agentx-1), (agenty+2, agentx-1),
                                (agenty, agentx+1), (agenty+1, agentx+1), (agenty+2, agentx+1),
                                (agenty+1, agentx), (agenty+2, agentx), (agenty+3, agentx)]
        elif agentfacing == 4: # it is facing the negative x direction
            could_be_cleaned = [(agenty-1, agentx), (agenty-1, agentx-1), (agenty-1, agentx-2),
                                (agenty+1, agentx), (agenty+1, agentx-1), (agenty+1, agentx-2),
                                (agenty, agentx-1), (agenty, agentx-2), (agenty, agentx-3)]
        # filter out the out of bound positions
        # filterf = lambda x: x[0] >= 0 and x[0] < self.max_y and x[1] >= 0 and x[1] < self.max_x
        # could_be_cleaned = list(filter(filterf, could_be_cleaned))
        
        for y,x in self.water_locs:
            if (y,x) in could_be_cleaned:
                return True
        # for x in could_be_cleaned:
        #     if list(x) in self.water_locs:
        # is_water_covered_by_fireclean = any([x in self.water_locs for x in could_be_cleaned])
        return False
    
    def convert(self, actions, observations):
        # convert the 5 state action_direction (absolute move/turn) 
        # to 5x3x2x2 state action (relative move, turn, fireZap, fireClean) 
        # required by the enviornment curreing facing from 12 o'clock, clockwise 1,2,3,4
        # action 0 is no action
        # action_direction 1,2,3,4 is up, right, down, left
        move_turns = []
        for i in range(self.n_agents):
            current_y = observations[i*6]
            current_x = observations[i*6+1]
            current_facing_one_hot = observations[i*6+2:i*6+6]
            assert np.sum(current_facing_one_hot) == 1
            current_facing = np.argmax(current_facing_one_hot) + 1
            current_pos = (current_y, current_x)
            action = actions[i]
            # translate the 5 state action_direction (absolute move/turn) 
            # to 5x3 state action (relative move, turn) required by the enviornment 
            move_or_turn = self.move_or_turn(current_facing, action)
            
            # stesp to decide if the agent should fireClean
            predicted_pos_y, predicted_pos_x = self.predict_move_or_turn_result_pos(action, move_or_turn, current_pos)
            predicted_facing = self.predict_move_or_turn_result_facing(move_or_turn, current_facing)
            should_fire_clean = self.should_fire_clean(predicted_pos_y, predicted_pos_x, predicted_facing)
            
            move_turns.append(move_or_turn[0]) # relative move
            move_turns.append(move_or_turn[1]) # relative turn
            # move_turns.append(0) # never fireZap -- our environment is auto filling fireZap with 0
            move_turns.append(should_fire_clean) # fireClean
        return move_turns
        

class CleanUpLayerToFlatVectorConverter:
    def __init__(self, ascii_map, n_agents=2) -> None:
        self.n_agents = n_agents
        
        # process the ascii_map
        lines = ascii_map.strip().split("\n")
        ascii_array = np.array([list(line) for line in lines])

        # water_loc are corresponds to the 'H' and 'F' in the ASCII_MAP
        water_indices_w = np.argwhere(ascii_array == 'H')
        water_indices_f = np.argwhere(ascii_array == 'F')
        water_locs_np = np.concatenate([water_indices_w, water_indices_f], axis=0)
        assert water_locs_np.shape[0] == len(water_indices_w) + len(water_indices_f)
        self.water_locs = water_locs_np
        

        # app_loc are corresponds to the 'T' and 'B' in the ASCII_MAP
        apple_indices_t = np.argwhere(ascii_array == 'T')
        apple_indices_b = np.argwhere(ascii_array == 'B')
        apple_locs_np = np.concatenate([apple_indices_t, apple_indices_b], axis=0)
        assert apple_locs_np.shape[0] == len(apple_indices_t) + len(apple_indices_b)
        self.apple_locs = apple_locs_np
        
        # process the observation space
        # first 3*n_agents are the position(y,x) and orientation of the agents
        # next len(water_locs) are the dirty water states'
        # next len(apple_locs) are the apple states
        self.n_rows = len(lines)
        self.n_columns = len(lines[0])
        self.r_space = gym.spaces.Discrete(self.n_rows)
        self.c_space = gym.spaces.Discrete(self.n_columns)
        # self.orientation_space = gym.spaces.Discrete(4)
        self.orientation_space = gym.spaces.Box(low=1, high=4, shape=(),dtype=np.int32)
        self.binary_space = gym.spaces.MultiBinary(len(self.water_locs) + len(self.apple_locs))
        
        upper_bound_for_obs_scalar = np.max([self.n_rows, self.n_columns, 4])
        lower_bound_for_obs_scalar = 0
        self.observation_space = gym.spaces.Box(
            low=lower_bound_for_obs_scalar, 
            high=upper_bound_for_obs_scalar, 
            shape=(self.n_agents, 6*n_agents + len(self.water_locs) + len(self.apple_locs)), dtype=np.int32)
        
        # self.observation_space = gym.spaces.Tuple(
        #     [self.r_space, self.c_space, self.orientation_space] * self.n_agents 
        #     + [self.binary_space])
        
        # substrate specific parameters for converting layer notation to flat vector notation
        self.agent_layer_idx = 6 # ref clean_up_mini.py search for superOverlay
        self.apple_layer_idx = 4 # ref clean_up_mini.py search for upperPhysical
        self.dirt_layer_idx = 4 # ref clean_up_mini.py search for upperPhysical
        self.dirt_value = 21
        self.apple_value = 1
    
    def convert(self, observation):
        if type(observation) == dict:
            assert 'WORLD.LAYER' in observation
            observation = observation['WORLD.LAYER']
        elif type(observation) == np.ndarray:
            pass
        self.observation = observation
        
        # parse observation from layer notation to flat vector notation
        agents_pos_and_orientation = self.get_agent_pos_and_orientation()
        apples_exist = self.get_locs_equal(self.apple_locs, self.apple_layer_idx, self.apple_value)
        waters_dirty = self.get_locs_equal(self.water_locs, self.dirt_layer_idx, self.dirt_value)
        return np.concatenate([agents_pos_and_orientation, apples_exist, waters_dirty])
        
    # check if loc index in array is of certain value
    # return vector of 1 and 0 of the same length as locs
    def get_locs_equal(self, locs, layer_idx, value):
        layer = self.observation[:,:,layer_idx]
        return np.array([layer[loc[0], loc[1]] == value for loc in locs])
    
    def get_agent_pos_and_orientation(self):
        # for every 3 element the players_states is #rows (Y), #columns (X), orientation(0-3)
        players_states = []
        
        superOverlayLayer = self.observation[:,:,6] # ref clean_up_mini.py search for superOverlay
        for i in range(1,self.n_agents+1):
            entries = [] 
            
            # find entry in superOverlayLayer such that it is between 4i+1 and 4i+4 (inclusive)
            for j in range(1,5):
                tmp = np.argwhere(superOverlayLayer == 4*i+j)
                if len(tmp) == 0:
                    continue
                # assert len(tmp) == 1
                entries.extend(tmp[0]) # y coord, x coord
                one_hot_orientation = np.zeros(4, dtype=np.int32)
                one_hot_orientation[j-1] = 1
                entries.extend(list(one_hot_orientation)) # orientation we map orientation from 1-4 to 0-3 for the gym space (which start with zero)
                
                
            players_states.extend(entries)
            
        return players_states
    
    
def save_frames_to_video(frames, output_file='video.mp4', fps=24, scale=9):
    import imageio
    import pygame # also tried Image, but pygame does gives better quality
    pygame.init()
    
    with imageio.get_writer(output_file, fps=fps) as writer:
        for obs in frames:
            surface = pygame.surfarray.make_surface(obs)
            rect = surface.get_rect()
            surf = pygame.transform.scale(
                surface, (rect[2] * scale, rect[3] * scale))
            # get the image from the surface
            img = pygame.surfarray.array3d(surf)
            writer.append_data(img)
    pygame.quit()
    
    