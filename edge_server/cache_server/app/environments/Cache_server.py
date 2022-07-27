import copy
import random
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
import scipy.stats as stats

class Cache_server(gym.Env):
    environment_name = "Cache_Server"

    def __init__(self, cache_dimension=100,total_file_number=1000,zipf_param=1.3,eval=False):

        self.action_space = spaces.Discrete(cache_dimension+1)
        self.observation_space = spaces.Discrete(cache_dimension)

        self.seed()
        self.trials = 1000
        self._max_episode_steps = 10000
        self.max_episode_steps = 10000
        self.reward_threshold = .8*self.max_episode_steps
        self.id = "cache_server"
        self.cache_dimension = cache_dimension
        self.reward_for_achieving_goal = 1.0
        self.step_reward_for_not_achieving_goal = 0
        self.total_file_number = total_file_number
        self.zipf_param = zipf_param
        self.state_long_term = 0
        self.eval =eval
        self.done = False
        # self.http_access = False
        # self.deterministic = deterministic

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # if not self.deterministic:
        #     self.desired_goal = self.randomly_pick_state_or_goal()
        #     self.state = self.randomly_pick_state_or_goal()
        # else:
            # self.desired_goal = [0 for _ in range(self.environment_dimension)]
        #     self.state = [1 for _ in range(self.environment_dimension)]
        # self.state.extend(self.desired_goal)
        self.desired_goal = 1
        # self.state_short_term = np.array(self.randomly_pick_state())
        self.state = np.array(self.randomly_pick_state())
        self.state_freq = np.ones(self.state.shape)
        self.observation = self.zipf_truncated(self.zipf_param,self.total_file_number,1)
        self.achieved_goal = np.sum(self.state==self.observation)
        self.step_count = 0
        self.episod_reward =0
        self.reward = 0
        self.state = np.concatenate((self.state_freq,self.state),axis=0)
        self.state = np.log(.001+self.state)
        self.step_count=0
        self.done = False
        self.LFU_state = np.array(self.randomly_pick_state())
        self.LFU_state_freq = np.ones(self.LFU_state.shape)
        return self.state

    def randomly_pick_state(self):
        return [random.randint(0, self.total_file_number) for _ in range(self.cache_dimension)]

    def step(self, action,observation=None):
        """Conducts the discrete action chosen and updated next_state, reward and done"""
        # action = self.action_space.sample()
        if observation is None:
            self.observation = self.zipf_truncated(self.zipf_param,self.total_file_number,1)
        else:
            self.observation = observation

        if type(action) is np.ndarray:
            action = action[0]
        # assert action <= self.cache_dimension + 1, "You picked an invalid action"
        self.step_count += 1
        self.state = (np.ceil(np.exp(self.state))-0.001).astype("int32")
        self.state_freq = self.state[0:self.cache_dimension]
        self.next_state = self.state[self.cache_dimension:]
        # self.next_state = copy.copy(self.state)
        check_file_in_cache = np.sum(self.next_state == self.observation)
        # check_file_in_cache = np.sum(self.next_state[self.cache_dimension:] == self.observation)
        # if action != 0:
        #     if not check_file_in_cache:
        #         self.next_state[action-1] = self.observation

        if check_file_in_cache:
            self.state_freq[self.next_state == self.observation] += 1
        else:
            if action!= 0:
                self.next_state[action - 1] = self.observation
                self.state_freq[action - 1] = 1

        # if action != 0:
        #     if not check_file_in_cache:
        #         self.next_state[action-1] = self.observation
        #         self.state_freq[action-1] = 1
        # else:
        #     self.state_freq[self.next_state== self.observation]+=1
        self.next_state = np.concatenate((self.state_freq,self.next_state),axis=0)

        if check_file_in_cache:
            self.reward = self.reward_for_achieving_goal
            # self.reward = (self.reward_for_achieving_goal + self.reward*(self.step_count-1))/ self.step_count
            # self.state_long_term+=self.reward
            # self.reward =
        else:
            self.reward = self.step_reward_for_not_achieving_goal
            # self.reward = (self.step_reward_for_not_achieving_goal + self.reward*(self.step_count-1))/ self.step_count
            # self.state_long_term += self.reward
        # if self.goal_achieved(self.next_state):
        #     self.reward = self.reward_for_achieving_goal
        #     self.done = True
        # else:
        #     self.reward = self.step_reward_for_not_achieving_goal
        self.episod_reward = self.episod_reward + self.reward
        if not self.eval:
            if self.step_count >=self.max_episode_steps:
                # print("step count is "+ str(self.step_count))
                self.done = True
                # self.reward = np.sum(self.state_freq)
                # self.reward = self.episod_reward
                print("episod reward is "+str(self.episod_reward/self.max_episode_steps))
                self.episod_reward = 0
        self.achieved_goal = check_file_in_cache
        self.state = np.log(0.001+self.next_state)
        self.check_file_in_cache = check_file_in_cache
        # return self.reward
        return self.state, self.reward, self.done, {}
    # def goal_achieved(self, next_state):
    #     return next_state[:self.environment_dimension] == next_state[-self.environment_dimension:]
    def get_state(self):
        return (np.ceil(np.exp(self.state)) - 0.001).astype("int32")

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
        interface to fit with the open AI gym specifications"""
        # Desired goal is the cache hit
        # The achieved is the status of the cache hit, if the requited file is available in the cache
        if (achieved_goal == desired_goal).all():
            reward = self.reward_for_achieving_goal
        else:
            reward = self.step_reward_for_not_achieving_goal
        return

    def zipf_truncated(self,a,M,N):
        # Number of files
        # sampling size
        # distribution parameters
        # N = 7
        x = np.arange(1, M + 1)
        # a = 1.1
        weights = x ** (-a)
        weights /= weights.sum()
        bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))
        samples = bounded_zipf.rvs(size=N)
        return samples
        # sample = bounded_zipf.rvs(size=10000)
        # plt.hist(sample, bins=np.arange(1, N + 2))
        # plt.show()
    def LFU(self,observation):
        self.LFU_next_state = copy.copy(self.LFU_state)
        check_file_in_cache = np.sum(self.LFU_next_state == observation)
        self.check_file_in_cache_LFU = check_file_in_cache
        if check_file_in_cache:
            self.LFU_state_freq[self.LFU_next_state == observation] += 1
        else:
            LFU_state = np.argmin(self.LFU_state_freq)
            self.LFU_next_state[LFU_state] = observation
            self.LFU_state_freq[LFU_state] = 1
        if check_file_in_cache:
            self.LFU_reward = self.reward_for_achieving_goal
        else:
            self.LFU_reward = self.step_reward_for_not_achieving_goal
        self.LFU_state = self.LFU_next_state

        return self.LFU_reward
    def test_LFU(self):
        eval_time = 10000
        reward = []
        for i in range(eval_time):
            observation = self.zipf_truncated(self.zipf_param, self.total_file_number, 1)
            reward.append(self.LFU(observation))
        cache_hit_ratio = np.sum(np.array(reward))/eval_time
        return cache_hit_ratio,reward
    # def get_from_http_client():
    #     from flask import Flask
