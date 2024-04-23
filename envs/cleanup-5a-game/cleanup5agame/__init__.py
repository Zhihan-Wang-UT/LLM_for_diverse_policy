# from gym import register
from gym.envs.registration import register
# from cleanupgame import CleanUpGame
import gym

from cleanup5agame.cleanup5agame.wrapper import TimeStepWrapper

register(
    f"cleanup-game-mini-reducedaction-v0",
    entry_point="cleanup5agame.cleanup5agame:CleanUpReducedActionGame",
    kwargs={
        "n_agents": 2,
        "mode": None,
    },
)

# def create_wrapped_environment(n_agents=2, mode=None, max_timesteps=1000):
#     # Initialize the original environment with any required arguments
#     env = gym.make("cleanup-game-mini-reducedaction-v0", n_agents=n_agents, mode=mode)
#     # Wrap the environment with your TimeStepWrapper
#     wrapped_env = TimeStepWrapper(env, max_timesteps=max_timesteps)
#     return wrapped_env

register(
    id="cleanup-game-mini-reducedaction-timestep-v0",
    entry_point='cleanup5agame:create_wrapped_environment',  
    kwargs={
        "n_agents": 2,
        "mode": None,
        "max_timesteps": 500,  # You can adjust the default 
        # max_timesteps here, this is used to normalize timestep only
    },
)

def create_wrapped_environment_from_wrapper(n_agents=2, mode=None, wrapper_class=None, **kwargs):
    assert wrapper_class is not None, "Please provide a wrapper class"
    env = gym.make("cleanup-game-mini-reducedaction-v0", n_agents=n_agents, mode=mode, **kwargs)
    # Wrap the environment with your TimeStepWrapper
    if isinstance(wrapper_class, list):
        for wrapper in wrapper_class:
            env = wrapper(env)
    else:
        env = wrapper_class(env)
    return env

from cleanup5agame.cleanup5agame.wrapper import SharedEgoRewardWrapper
from cleanup5agame.cleanup5agame.wrapper import FullyCooperativeRewardWrapper
register(
    id="cleanup-game-mini-reducedaction-sharedego-v0",
    entry_point='cleanup5agame:create_wrapped_environment_from_wrapper', 
    kwargs={
        "n_agents": 2,
        "mode": None,
        "wrapper_class": SharedEgoRewardWrapper,
    },  
)

register(
    id="cleanup-game-mini-reducedaction-fullycooperative-v0",
    entry_point='cleanup5agame:create_wrapped_environment_from_wrapper', 
    kwargs={
        "n_agents": 2,
        "mode": None,
        "wrapper_class": FullyCooperativeRewardWrapper,
    },
)

from cleanup5agame.cleanup5agame.wrapper import AgentLocationMaskWrapper

register(
    id = "cleanup-game-mini-reducedaction-agentlocationmask-v0",
    entry_point='cleanup5agame:create_wrapped_environment_from_wrapper',
    kwargs={
        "n_agents": 2,
        "mode": None,
        "wrapper_class": AgentLocationMaskWrapper,
    },
)

register(
    id = "cleanup-game-mini-reducedaction-agentlocationmask-sharereward-v0",
    entry_point='cleanup5agame:create_wrapped_environment_from_wrapper',
    kwargs={
        "n_agents": 2,
        "mode": None,
        "wrapper_class": [AgentLocationMaskWrapper, FullyCooperativeRewardWrapper],
    },
)

