import numpy as np
import copy
from ExpReplay import EpisodicExperienceReplay
from test_llmvec import init_model
from test_llmvec import convert_transition_to_text
import torch

class LLMEpisodicExperienceReplay(EpisodicExperienceReplay):
    """
        Class that encapsulates the storage used for teammate generation and AHT training.
    """
    def __init__(self, config, ob_shape, act_shape, max_episodes=100000, max_eps_length=20):
        """
            Constructor of the experience replay class.
            :param ob_shape: Observation shape.
            :param act_shape: Action space dimensionality.
            :param max_episodes: Maximum number of stored episodes.
            :param max_eps_length: Maximum length of each episode.
        """
        
        super().__init__(
            ob_shape,
            act_shape,
            max_episodes,
            max_eps_length
        )
        self.config = config
        self.env_description = self.config.env.ENV_DESCRIPTION
        self.instruction = self.config.env.INSTRUCTION
        
        self.llm_episode_embedding = np.zeros([max_episodes, 4096])
        self.llm_embed_model = init_model("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp")
        self.old_pointer = 0
        self.num_added_transitions = 0
    
    def add_episode(self, obses, acts, rewards, dones, next_obses):
    #def add_episode(self, obses, acts, rewards, total_rew, eps_length, gamma):
        """
            A method to add an episode of experiences into the replay buffer. Target returns that are appended
            with obs are changed in hindsight according to the achieved returns.
            :param obses: List of stored obses.
            :param acts: List of stored acts.
            :param rewards: List of rewards per timestep.
            :param total_rew: The total returns for an episode.
            :param eps_length: The length of an episode.
            :param gamma: The discount rate used.
        """

        eps_length = self.max_eps_length

        rewards = rewards[:eps_length]
        self.obs[self.pointer,:eps_length] = obses[:eps_length]
        self.actions[self.pointer,:eps_length] = acts[:eps_length]
        self.rewards[self.pointer,:eps_length, :] = rewards[:eps_length, :]
        self.next_obs[self.pointer,:eps_length] = next_obses[:eps_length]
        self.dones[self.pointer, :eps_length] = dones[:eps_length]

        self.size = min(self.size + eps_length + self.max_eps_length, self.max_episodes * self.max_eps_length)
        self.num_episodes = min(self.num_episodes + 1, self.max_episodes)
        self.pointer = (self.pointer + 1) % self.max_episodes
    
    def embed_transition(self, obs, act, reward):
        transition = dict()
        transition['action'] = act
        transition['observation'] = obs
        transition['reward'] = reward
        
        return convert_transition_to_text(transition)
    
    def embed_episodes(self):
        
        if self.old_pointer < self.pointer:
            obses = self.obs[self.old_pointer:self.pointer, :self.max_eps_length]
            acts = self.actions[self.old_pointer:self.pointer, :self.max_eps_length]
            rewards = self.rewards[self.old_pointer:self.pointer, :self.max_eps_length]
        else:
            obses = self.obs[self.old_pointer:, :self.max_eps_length]
            acts = self.actions[self.old_pointer:, :self.max_eps_length]
            rewards = self.rewards[self.old_pointer:, :self.max_eps_length]
            obses = np.concatenate([obses, self.obs[:self.pointer, :self.max_eps_length]],axis=0)
            acts = np.concatenate([acts, self.actions[:self.pointer, :self.max_eps_length]],axis=0)
            rewards = np.concatenate([rewards, self.rewards[:self.pointer, :self.max_eps_length]],axis=0)
        
        
        traj_text = [self.convert_traj_to_text(obs_traj, acts_traj, rewards_traj) \
            for obs_traj, acts_traj, rewards_traj in zip(obses, acts, rewards)]
        
        embeddings = self.llm_embed_model.encode(traj_text)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        embeddings = embeddings.cpu().numpy()
        self.num_added_transitions = len(traj_text)
        
        if self.old_pointer < self.pointer:
            self.llm_episode_embedding[self.old_pointer: self.pointer, :] = embeddings
        else:
            
            self.llm_episode_embedding[self.old_pointer: , :] = embeddings[:(self.max_episodes - self.old_pointer), :]
            self.llm_episode_embedding[:self.pointer, :] = embeddings[(self.max_episodes - self.old_pointer):, :]
        
        self.augment_rewards()
        self.old_pointer = self.pointer

    def augment_rewards(self):
        
        if self.old_pointer < self.pointer:
            prev_embeddings = self.llm_episode_embedding[:self.old_pointer, :]
            new_embeddings = self.llm_episode_embedding[self.old_pointer:self.pointer, :]
        else:
            prev_embeddings = self.llm_episode_embedding[self.pointer:self.old_pointer, :]
            new_embeddings = self.llm_episode_embedding[self.old_pointer:, :]
            new_embeddings = np.concatenate([new_embeddings, self.llm_episode_embedding[:self.pointer,:]], axis = 0)

        # print(prev_embeddings.shape)
        # print(new_embeddings.shape)
        if len(prev_embeddings) != 0:
            cosine_similarity = prev_embeddings @ new_embeddings.T
            exploration_rewards = cosine_similarity.mean(axis = 0)
            exploration_rewards = exploration_rewards.reshape((exploration_rewards.shape[0],1,1))
            if self.old_pointer < self.pointer:
                self.rewards[self.old_pointer:self.pointer] += self.config.exploration_reward_weight*exploration_rewards
            else:
                self.rewards[self.old_pointer:] +=  self.config.exploration_reward_weight*exploration_rewards[:(self.max_episodes - self.old_pointer)]
                self.rewards[:self.pointer] +=  self.config.exploration_reward_weight*exploration_rewards[(self.max_episodes - self.old_pointer):]
        
        
    def sample_all(self):
        """
            A method to return everything stored in the buffer.
            :return: Everything contained in buffer.
        """
        
        start = self.pointer - self.num_added_transitions
        if start < 0:
            start += self.max_episodes
            
        if start < self.pointer:
            obses = self.obs[start:self.pointer, :self.max_eps_length]
            acts = self.actions[start:self.pointer, :self.max_eps_length]
            rewards = self.rewards[start:self.pointer, :self.max_eps_length]
            next_obs = self.next_obs[start:self.pointer, :self.max_eps_length]
            dones = self.dones[start:self.pointer, :self.max_eps_length]
            
        else:
            obses = self.obs[start:, :self.max_eps_length]
            
            acts = self.actions[start:, :self.max_eps_length]
            rewards = self.rewards[start:, :self.max_eps_length]
            next_obs = self.next_obs[start:, :self.max_eps_length]
            dones = self.dones[start:, :self.max_eps_length]
            
            obses = np.concatenate([obses, self.obs[:self.pointer, :self.max_eps_length]], axis=0)
            acts = np.concatenate([acts, self.actions[:self.pointer, :self.max_eps_length]],axis=0)
            rewards = np.concatenate([rewards, self.rewards[:self.pointer, :self.max_eps_length]],axis=0)
            next_obs = np.concatenate([next_obs, self.next_obs[:self.pointer:, :self.max_eps_length]],axis=0)
            dones = np.concatenate([dones, self.dones[:self.pointer, :self.max_eps_length]], axis = 0)
            
        return obses, acts, next_obs, dones, rewards
    
    def convert_traj_to_text(self, obses, acts, rewards):
        text = ""
        for i, (obs,act,reward) in enumerate(zip(obses, acts, rewards)):
            text += "Step" + str(i) + " - " + self.embed_transition(obs,act,reward)
        return self.env_description + "." + self.instruction + "." + text + "."

    def save(self, dir_location):
        """
            A method that stores every variable into disk.
        """
        super().save(dir_location)

    def load(self, dir_location):
        """
            A method that loads experiences stored within a disk.
        """
        super().load(dir_location)
