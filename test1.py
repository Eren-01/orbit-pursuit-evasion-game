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
from pettingzoo.mpe import simple_tag_v2

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


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='simple_tag_v2', help='name of the env')
parser.add_argument('--episode_num', type=int, default=3000,
                    help='total episode num during training procedure')
parser.add_argument('--episode_length', type=int, default=50, help='steps per episode')
parser.add_argument('--learn_interval', type=int, default=100,
                    help='steps interval between learning time')
parser.add_argument('--random_steps', type=int, default=5e4,
                    help='random steps before the agent start to learn')
parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter')
parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
parser.add_argument('--batch_size', type=int, default=1024, help='batch-size of replay buffer')
parser.add_argument('--actor_lr', type=float, default=0.01, help='learning rate of actor')
parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate of critic')
args = parser.parse_args()
# 创建文件夹保存结果
env_dir = os.path.join('./results', args.env_name)
if not os.path.exists(env_dir):
    os.makedirs(env_dir)
# total_files = len([file for file in os.listdir(env_dir)])
# result_dir = os.path.join(env_dir, f'{total_files + 1}')
result_dir = os.path.join(env_dir, f'32')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
print(result_dir)

env, dim_info = get_env('simple_tag_v2', 50, continuous_actions=True)


maddpg = MADDPG.load(dim_info, os.path.join(result_dir, 'model.pt'))
agent_num = env.num_agents
print("agent_num=",agent_num)


# buffer长度 buffer_capacity= 1e6 batch_size=1024 actor_lr,critic_lr=0,01
print("dim_info:", dim_info)

# for episode in range(args.episode_num): #训练5000个episode
obs = env.reset()
agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode 初始化

action = maddpg.select_action(obs)
action_new = {}
for ac, tens in action.items():
    tens = tens.cpu().detach().numpy()[0]
    # print("tens",tens)
    action_new[ac] = tens
#
action = action_new
next_obs, reward, done, info = env.step(action)  # 输入联合动作action

print("action= ",action)
print("obs=",obs)
print("next_obs=", next_obs)
print("reward=", reward)
print("done=", done)
print("info=", info)




# print(action)
#




# def get_running_reward(arr: np.ndarray, window=100):
#     """calculate the running reward, i.e. average of last `window` elements from rewards"""
#     running_reward = np.zeros_like(arr)
#     for i in range(window - 1):
#         running_reward[i] = np.mean(arr[:i + 1])
#     for i in range(window - 1, len(arr)):
#         running_reward[i] = np.mean(arr[i - window + 1:i + 1])
#     return running_reward
#
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# # training finishes, plot reward
# fig, ax = plt.subplots()
# x = range(1, args.episode_num + 1)
# for agent_id, reward in episode_rewards.items():
#     ax.plot(x, reward, label=agent_id)
#     ax.plot(x, get_running_reward(reward))
# ax.legend()
# ax.set_xlabel('episode')
# ax.set_ylabel('reward')
# title = f'training result of maddpg'
# ax.set_title(title)
# plt.savefig(os.path.join(result_dir, title))
