import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pettingzoo.mpe import simple_tag_v2
from MADDPG import MADDPG
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

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='simple_tag_v2', help='name of the env')
    parser.add_argument('--folder', type=str, default='38', help='name of the folder where model is saved')
    parser.add_argument('--episode-num', type=int, default=100, help='total episode num during evaluation')
    parser.add_argument('--episode-length', type=int, default=100, help='steps per episode')

    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch-size of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=0.001, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.001, help='learning rate of critic')

    args = parser.parse_args()

    model_dir = os.path.join('./results', args.env_name, args.folder)  # 文件路径
    print("model_dir:",model_dir)
    assert os.path.exists(model_dir)  # 断言
    gif_dir = os.path.join(model_dir, 'gif')
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    print("文件路径：",gif_dir)
    gif_num = len([file for file in os.listdir(gif_dir)])  # current number of gif 现有的gif
    print("gif_num:",gif_num)
    env, dim_info = get_env(args.env_name, args.episode_length)
    print("dim_info")
    print(dim_info)
    print(os.path.join(model_dir, 'model.pt'))
    #maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, model_dir)
    maddpg = MADDPG.load(dim_info, os.path.join(model_dir, 'model.pt'))
    agent_num = env.num_agents
    print("agent_num=",agent_num)
    # reward of each episode of each agent
    episode_rewards = {agent: np.zeros(args.episode_num) for agent in env.agents}
    for episode in range(args.episode_num):
        states = env.reset()
        agent_reward = {agent: 0 for agent in env.agents}  # agent reward of the current episode
        frame_list = []  # used to save gif
        while env.agents:  # interact with the env  for an episode
            action = maddpg.select_action(states)
            action_new = {}
            for ac, tens in action.items():
                tens = tens.cpu().detach().numpy()[0]
                # print("tens",tens)
                action_new[ac] = tens
            #
            action = action_new
            #action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
            # print(action)
            next_states, rewards, dones, infos = env.step(action)

            frame_list.append(Image.fromarray(env.render(mode='rgb_array')))


            for agent_id, reward in rewards.items():  # update reward
                agent_reward[agent_id] += reward

        env.close()
        message = f'episode {episode + 1}, '
        # episode finishes, record reward
        for agent_id, reward in agent_reward.items():
            episode_rewards[agent_id][episode] = reward
            message += f'{agent_id}: {reward:>4f}; '
        print(message)
        # save gif
        frame_list[0].save(os.path.join(gif_dir, f'out{gif_num + episode + 1}.gif'),
                           save_all=True, append_images=frame_list[1:], duration=1, loop=0)

        # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent_id, rewards in episode_rewards.items():
        ax.plot(x, rewards, label=agent_id)
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    total_files = len([file for file in os.listdir(model_dir)])
    title = f'evaluate result of maddpg solve pursuit-evasion'
    ax.set_title(title)
    plt.savefig(os.path.join(model_dir, title))

