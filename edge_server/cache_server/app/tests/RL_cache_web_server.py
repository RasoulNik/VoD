###################################
import sys, importlib
from pathlib import Path
##https://stackoverflow.com/questions/16981921/relative-imports-in-python-3

def import_parents(level=1):
    global __package__
    file = Path(__file__).resolve()
    parent, top = file.parent, file.parents[level]

    sys.path.append(str(top))
    try:
        sys.path.remove(str(parent))
    except ValueError:  # already removed
        pass

    __package__ = '.'.join(parent.parts[len(top.parts):])
    importlib.import_module(__package__)  # won't be needed after that
if __name__ == '__main__' and __package__ is None:
    import_parents(level=3) # N = 3
########################################
import random
import matplotlib.pyplot as plt
import gym
# import SAC_Discrete
from ..agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from ..environments.Cache_server import Cache_server
from ..agents.Trainer import Trainer
from ..utilities.data_structures.Config import Config
# from agents.DQN_agents.DQN import DQN
import numpy as np
import torch

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

config = Config()
config.seed = 1
# config.environment = Bit_Flipping_Environment(4)
config.cache_dimension = 100
config.total_file_number = 1000
config.zipf_param = [1.5]

config.num_episodes_to_run = 1
config.file_to_save_data_results = None
config.file_to_save_results_graph = None
config.visualise_individual_results = False
config.visualise_overall_agent_results = False
config.randomise_random_seed = False
config.runs_per_agent = 1
config.use_GPU = False
config.hyperparameters = {

    "Actor_Critic_Agents": {

        "learning_rate": 0.0005,
        "linear_hidden_units": [200, 30, 30, 30],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 25.0,
        "discount_rate": 1,
        "epsilon_decay_rate_denominator": 10.0,
        "normalise_rewards": False,
        "automatically_tune_entropy_hyperparameter": True,
        "add_extra_noise": False,
        "min_steps_before_learning": 1,
        "do_evaluation_iterations": True,
        "clip_rewards": False,

        "Actor": {
            "learning_rate": 0.001,
            # "linear_hidden_units": [20, 20],
            "linear_hidden_units": [200,100],
            # "final_layer_activation": "TANH",
            "final_layer_activation": "Softmax",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 25
        },

        "Critic": {
            "learning_rate": 0.01,
            # "linear_hidden_units": [20, 20],
            "linear_hidden_units": [200,100],
            "final_layer_activation": "None",
            "batch_norm": False,
            "buffer_size": 100000,
            "tau": 0.005,
            "gradient_clipping_norm": 25
        },

        "batch_size": 3,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 20,
        "learning_updates_per_learning_session": 10,
        "HER_sample_proportion": 0.8,
        "exploration_worker_difference": 1.0
    },

}

# def test_agent_solve_RL_cache():
config.environment = Cache_server(config.cache_dimension, config.total_file_number, config.zipf_param[0])
AGENTS = [SAC_Discrete]
trainer = Trainer(config, AGENTS)
results = trainer.run_games_for_agents()
for agent in AGENTS:
    agent_results = results[agent.agent_name]
    agent_results = np.max(agent_results[0][1][0:])
    assert agent_results >= 0.0, "Failed for {} -- score {}".format(agent.agent_name, agent_results)
plt.plot(results["SAC"][0][0])
plt.plot(results["SAC"][0][1])
plt.show()

# Test the agent
agent_trained = trainer.agent_trained
state = agent_trained.environment.get_state()

# cache_hit_ratio = []
# LFU_cache_hit_ratio =[]

def test_caching_with_http_server(agent_trained,observation):
    # game_scores, rolling_scores, time_taken = agent.run_n_episodes()
    eval_ep = True
    agent_trained.done = False
    agent_trained.environment.eval = True
    # agent_trained.environment.done = False
    # agent_trained.environment.step_num = 0
    # eval_step =int(1e1)
    # for i in range(eval_step):
    agent_trained.action = agent_trained.pick_action(eval_ep)
    agent_trained.conduct_action(agent_trained.action,observation)
    if agent_trained.time_for_critic_and_actor_to_learn():
        for _ in range(agent_trained.hyperparameters["learning_updates_per_learning_session"]):
            agent_trained.learn()
    Mask = True
    if not eval_ep:
        agent_trained.save_experience(experience=(agent_trained.state, agent_trained.action, agent_trained.reward, agent_trained.next_state, agent_trained.done))
    agent_trained.state = agent_trained.next_state
    agent_trained.global_step_number += 1
    return
    #     LFU
    lfu_cache_hit_ratio, lfu_reward = agent_trained.environment.test_LFU()
    # LFU_cache_hit_ratio.append(lfu_cache_hit_ratio)
    # print(agent_trained.total_episode_score_so_far)
    # cache_hit_ratio.append(agent_trained.total_episode_score_so_far/(eval_step+agent_trained.environment.max_episode_steps))
# plt.rcParams.update({'font.size': 14})
# plt.plot(config.zipf_param,cache_hit_ratio)
# plt.plot(config.zipf_param,LFU_cache_hit_ratio)
# plt.xlabel("Zipf parameter")
# plt.ylabel("Cache hit ratio")
# plt.legend(["ML","LFU"])
# plt.xticks(config.zipf_param)
# plt.show()
# cache_hit = []
# for i in range(1000):
#     file = agent_trained.environment.zipf_truncated(.7, config.total_file_number, 1)
#     test_caching_with_http_server(agent_trained,float(file))
#     check_file_in_cache = agent_trained.environment.check_file_in_cache
#     cache_hit.append(check_file_in_cache)
# print(np.sum(cache_hit)/1000)

from flask import Flask

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)


# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return {"Cache_status":1}

@app.route('/<file>')
def print_name(file):
    test_caching_with_http_server(agent_trained,float(file))
    check_file_in_cache = agent_trained.environment.check_file_in_cache
    return {"Cache_status":str(check_file_in_cache)}
# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run(threaded=True)