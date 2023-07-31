import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from MADDPG import MADDPG
from main import get_env
import numpy as np
import datetime
import os
# from pettingzoo.mpe import simple_tag_v2
from pettingzoo.magent import adversarial_pursuit_v4

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_env(env_name, ep_len=25, continuous_actions=True):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_tag_v2':
        new_env = simple_tag_v2.parallel_env(max_cycles=ep_len, continuous_actions=continuous_actions)

    new_env.reset()
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        if continuous_actions:
            _dim_info[agent_id].append(new_env.action_space(agent_id).shape[0])  # 连续动作空间的
        else:
            _dim_info[agent_id].append(new_env.action_space(agent_id).n)

    return new_env, _dim_info