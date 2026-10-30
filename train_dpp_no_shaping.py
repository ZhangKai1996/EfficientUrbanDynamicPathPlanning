import os
import random
import time
import shutil

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from algo.trainer_dqn import DPPBSRSTrainer

from env.utils import haversine
from env.environment import DPPEnvironment
from env.core import DFRGenerator
from env.geo import load_graph, get_hop_neighbors


def train_dqn(env,
              graph, p, num_action,
              gamma=0.99,
              max_step=100,
              save_rate=None,
              batch_size=32,
              other_dim=0,
              beta=0.0,
              epsilon_end=0.1,
              epsilon_decay=0.1,
              num_episodes_train=int(1e3),
              folder='exp_dpp/'):

    trainer = DPPBSRSTrainer(graph, p, num_action,
                             beta=beta,
                             gamma=gamma,
                             emb_dim=4,
                             other_dim=other_dim,
                             batch_size=batch_size,
                             epsilon_decay=int(num_episodes_train * epsilon_decay),
                             epsilon_end=epsilon_end)
    trainer.load_model('./trained/')
    if save_rate is None: save_rate = len(graph.nodes)

    log_folder = folder + 'logs'
    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)
        print('Removing all previous files!')
    writer = SummaryWriter(log_dir=log_folder)

    print('-------------Train kwargs------------')
    print('\t Number of episodes:', num_episodes_train)
    print('\t Epsilon decay:', epsilon_decay)
    print('\t Epsilon end:', epsilon_end)
    print('\t Batch size:', batch_size)
    print('\t Save rate:', save_rate)
    print('\t Folder:', folder)
    print('-------------------------------------')

    rew_stats, loss_stats, step_stats = [], [], []
    start_time = time.time()

    step = 0
    for episode in tqdm(range(1, num_episodes_train + 1)):
        done, (state, dyn, mask) = False, env.reset()

        total_shaped_reward, total_cost, total_original_reward = 0.0, 0.0, 0.0
        episode_step = 0
        while not done:
            action = trainer.choose_action(state, dyn=dyn, mask=mask, episode=episode)
            (next_state, next_dyn, next_mask), reward, done, info = env.step(action)
            [shaped_reward, original_reward, *_] = reward
            [cost, *_] = info

            experience = (state, dyn, mask,
                          action, shaped_reward, original_reward,
                          next_state, next_dyn, next_mask, float(done))
            trainer.replay_buffer.push(*experience, key=env.end_node_idx)

            if episode_step <= 2:
                loss = trainer.update_q_network(t=step)
                if loss is not None: loss_stats.append(loss)

            state, dyn, mask = next_state, next_dyn, next_mask
            total_shaped_reward += shaped_reward
            total_original_reward += original_reward
            total_cost += cost
            episode_step += 1
            step += 1
            if episode_step >= max_step: break

        dist_final = haversine(graph.nodes[env.cur_node], graph.nodes[env.end_node])
        step_stats.append([episode_step, dist_final, float(done)])
        rew_stats.append([total_cost, total_original_reward, total_shaped_reward])

        if episode % save_rate == 0:
            epoch = episode // save_rate
            end_time = time.time()
            mean_rew = np.mean(rew_stats, axis=0)
            mean_step = np.mean(step_stats, axis=0)
            eps = trainer.epsilon(episode=episode)

            print("Step: {}".format(step), end=', ')
            print("Length: {:>5.1f}".format(mean_step[0]), end=', ')
            print("OriRew: {:>6.2f}".format(mean_rew[1]), end=', ')
            print("ShaRew: {:>6.2f}".format(mean_rew[2]), end=', ')
            print("Cost: {:>5.1f}".format(mean_rew[0]), end=', ')
            print("Dist: {:>5.2f}".format(mean_step[1]), end=', ')
            print("SR: {:>5.3f}".format(mean_step[2]), end=', ')
            print("Epsilon: {:>5.3f}".format(eps), end=',')
            print("Time: {:>5.2f}".format(end_time - start_time))

            if len(loss_stats) > 0:
                mean_loss = np.mean(loss_stats)
                writer.add_scalar('Loss', mean_loss, episode)

            writer.add_scalar('Epsilon', eps, episode)
            writer.add_scalar('Mean Step', mean_step[0], episode)
            writer.add_scalar('Dist', mean_step[1], episode)
            writer.add_scalar('SR', mean_step[2], episode)
            writer.add_scalar('Cost', mean_rew[0], episode)
            writer.add_scalar('ORew', mean_rew[1], episode)
            writer.add_scalar('SRew', mean_rew[2], episode)

            trainer.save_model(folder)
            # total_epoch = num_episodes_train//save_rate
            # if epoch >= int(0.99*total_epoch):
            #     trainer.save_model(folder, prefix=epoch)

            rew_stats, loss_stats, step_stats = [], [], []
            start_time = end_time


def main():
    from parameters import alpha, gamma, seed
    from parameters import n, radius, length_limit, env_id, env_kwargs, kwargs

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args = load_graph(radius=radius, **env_kwargs)
    graph, p, num_action, od_pairs, *_ = args

    suffix = 'Static Prior'
    param_lst = [
        # {'exp_name': 'No Shaping', 'a': -1.0, 'eta': 1.0, 'e': 100.0, 'zeta': 0.0, 'c': 0.0, 'beta': 0.0},
        # {'exp_name': 'Only Static', 'a': -1.0, 'eta': 1.0, 'e': 100.0, 'zeta': 5.0, 'c': 0.0, 'beta': 0.0},
        # {'exp_name': 'No Dynamic', 'a': 4.0, 'eta': 1.0, 'e': 100.0, 'zeta': 5.0, 'c': -4.0, 'beta': 0.0},
        # {'exp_name': 'No Dynamic', 'a': 3.0, 'eta': 1.0, 'e': 100.0, 'zeta': 5.0, 'c': -3.0, 'beta': 0.0},
        # {'exp_name': 'No Dynamic', 'a': 2.0, 'eta': 1.0, 'e': 100.0, 'zeta': 5.0, 'c': -2.0, 'beta': 0.0},
        # {'exp_name': 'No Dynamic', 'a': 1.0, 'eta': 1.0, 'e': 100.0, 'zeta': 5.0, 'c': -1.0, 'beta': 0.0},

        {'exp_name': 'Only Dynamic1', 'a': -1.0, 'eta': 1.0, 'e': 100.0, 'zeta': 0.0, 'c': 0.0, 'beta': 1.0},

        # {'exp_name': 'No Clipping', 'a': -1.0, 'eta': 1.0, 'e': 100.0, 'zeta': 5.0, 'c': 0.0, 'beta': 1.0},
        # {'exp_name': 'No Clipping', 'a': -1.0, 'eta': 1.0, 'e': 100.0, 'zeta': 5.0, 'c': 0.0, 'beta': 0.5},
        # {'exp_name': 'No Clipping', 'a': -1.0, 'eta': 1.0, 'e': 100.0, 'zeta': 5.0, 'c': 0.0, 'beta': 0.1},
        # {'exp_name': 'No Clipping', 'a': -1.0, 'eta': 1.0, 'e': 100.0, 'zeta': 5.0, 'c': 0.0, 'beta': -0.1},
        # {'exp_name': 'No Clipping', 'a': -1.0, 'eta': 1.0, 'e': 100.0, 'zeta': 5.0, 'c': 0.0, 'beta': -0.5}
        # {'exp_name': 'No Clipping1', 'a': -1.0, 'eta': 1.0, 'e': 100.0, 'zeta': 5.0, 'c': 0.0, 'beta': 1.0},
    ]

    hop_neighborhoods = get_hop_neighbors(graph, max_n=n, limit=length_limit)
    dfr_generator = DFRGenerator(graph, neighbors=hop_neighborhoods, length_limit=length_limit)
    for param in param_lst:
        env = DPPEnvironment(graph, p, num_action,
                             eta=param['eta'],
                             e=param['e'],
                             zeta=param['zeta'],
                             c=param['c'],
                             a=param['a'],
                             alpha=alpha,
                             gamma=gamma,
                             dfr_generator=dfr_generator,
                             od_pairs=od_pairs)

        root = 'trained/exp_dpp_dqn_{}_{}_{}_{}/'.format(env_id, int(radius), len(graph.nodes), suffix)
        folder = param['exp_name']
        folder += '({}_{}_{}_{}_{}_{}_{}_{}_{})/'.format(param['a'],
                                                      param['eta'],
                                                      param['e'],
                                                      param['zeta'],
                                                      param['c'],
                                                      param['beta'],
                                                      n, alpha,
                                                      env_kwargs['max_step'])
        train_dqn(env,
                  graph, p,
                  num_action,
                  beta=param['beta'],
                  gamma=gamma,
                  folder=root+folder,
                  max_step=env_kwargs['max_step'],
                  other_dim=hop_neighborhoods['max_len'],
                  **kwargs)
    print('Training process is over !')
    print('-------------------------------------')


if __name__ == '__main__':
    main()
