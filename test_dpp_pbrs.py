import os

from tqdm import tqdm
import numpy as np
import pandas as pd

from algo.trainer_dqn import DPPBSRSTrainer
from algo.baselines import dijkstra_path

from env.geo import load_graph, get_hop_neighbors
from env.core import DFRGenerator
from env.environment import DPPEnvironment
from env.utils import haversine

from parameters import alpha, gamma
from parameters import n, radius, length_limit, env_id, env_kwargs

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

seed = 1234
suffix = 'Dynamic Feature'
param_lst = [
    {'exp_name': 'Only Static', 'e': 60.0, 'zeta': 5.0, 'a': -1.0, 'c': 0.0, 'beta': 0.0},
]


def get_index():
    index = []
    for param in param_lst:
        idx = param['exp_name']
        if 'Only Static' in idx:
            idx += '(n={})'.format(n)
        elif 'No Dynamic' in idx:
            idx += '(a={},c={})'.format(param['a'], param['c'])
        elif 'No Clipping' in idx:
            idx += '(beta={})'.format(param['beta'])
        elif 'Ours' in idx:
            idx += '(beta={},'.format(param['beta'])
            idx += 'a={},c={})'.format(param['a'], param['c'])
        index.append(idx)
    return index


def test_dqn(env, trainer, root, max_step=100, render=False):
    gaps, srs, paths = [], [], {}
    start_node, end_node = None, None
    for j, param in enumerate(param_lst):
        env.e, env.zeta = param['e'], param['zeta']

        folder = root + param['exp_name']
        folder += '({}_{}_{}_{}_{}_{}_{})/'.format(param['a'],
                                                   param['e'],
                                                   param['zeta'],
                                                   param['c'],
                                                   param['beta'],
                                                   n, max_step)
        if not os.path.exists(folder): continue
        trainer.beta = param['beta']
        trainer.load_model(folder)

        done_rl, (state, dyn, mask) = False, env.reset(render=render, reuse=j > 0)
        start_node, end_node = env.start_node, env.end_node

        if 'Dijkstra' not in paths.keys():
            total_cost_di, path_di = dijkstra_path(env.graph, start_node, end_node, matrix=env.dfr.get_w())
            total_cost_di *= env.alpha
            done_di = path_di[0] == start_node and path_di[-1] == end_node

            total_shaped_rew_di = 0.0
            for u, v in zip(path_di[:-1], path_di[1:]):
                shaped_reward, *_ = env.calculate_potential(u, v, verbose=False)
                total_shaped_rew_di += shaped_reward
            total_shaped_rew_di += int(done_di) * env.e

            paths['Dijkstra'] = [end_node, total_cost_di, total_shaped_rew_di, path_di, done_di]
        else:
            total_cost_di = paths['Dijkstra'][1]

        path_rl = [env.cur_node, ]
        total_cost_rl, total_shaped_rew_rl, total_origin_rew_rl = 0.0, 0.0, 0.0
        episode_step = 0
        while not done_rl:
            action = trainer.choose_action(state, dyn=dyn, mask=mask)
            (next_state, next_dyn, next_mask), reward, done_rl, info = env.step(action, verbose=False)
            [shaped_reward, original_reward, *_] = reward
            [cost, *_] = info

            diff_static = shaped_reward - original_reward
            # diff_dict[j].append([diff_static, ])

            # diff_dynamic = trainer.potential_diff(state, dyn, mask, next_state, next_dyn, next_mask, done_rl)
            # print('Reward:{:>7.4f}'.format(shaped_reward + trainer.beta * diff))
            # diff_dict[j].append([diff_static, diff_dynamic, diff_static + diff_dynamic*trainer.beta])

            path_rl.append(env.cur_node)
            total_cost_rl += cost
            total_shaped_rew_rl += shaped_reward
            total_origin_rew_rl += original_reward
            episode_step += 1

            state, dyn, mask = next_state, next_dyn, next_mask
            if episode_step >= max_step: break

        gap = 10.0
        if done_rl:
            gap = (total_cost_rl - total_cost_di) / total_cost_di
        gaps.append(gap)
        srs.append(float(done_rl))
        paths[param['exp_name']] = [end_node, total_cost_rl, total_shaped_rew_rl, path_rl, done_rl]
    return {'o': start_node,
            'd': end_node,
            'paths': paths,
            'others':[gaps, srs]}


def main():
    args = load_graph(radius=radius, **env_kwargs)
    graph, p, num_action, od_pairs, *_ = args

    hop_neighborhoods = get_hop_neighbors(graph, max_n=n, limit=length_limit)
    dfr_generator = DFRGenerator(graph, neighbors=hop_neighborhoods, length_limit=length_limit)

    root = 'trained/exp_{}_{}_{}_{}_{}_{}/'.format(env_id, int(radius),
                                                   int(length_limit[0]),
                                                   int(length_limit[1]),
                                                   suffix, seed)
    png_folder = root + 'pictures/'
    if not os.path.exists(png_folder): os.mkdir(png_folder)

    env = DPPEnvironment(graph, p, num_action,
                         alpha=alpha,
                         gamma=gamma,
                         dfr_generator=dfr_generator,
                         od_pairs=od_pairs)
    trainer = DPPBSRSTrainer(graph, p, num_action,
                             gamma=gamma,
                             other_dim=hop_neighborhoods['max_len'])
    render = True
    num_episodes = 100

    gap_stats, sr_stats = [], []
    for i in tqdm(range(num_episodes), desc='Testing ...'):
        result = test_dqn(env, trainer, root, max_step=env_kwargs['max_step'], render=render)
        gap_stats.append(result['others'][0])
        sr_stats.append(result['others'][1])

        if result['o'] is not None and render:
            name = png_folder + '{}_{}_{}'.format(result['o'], result['d'], i+1)
            env.render(paths=result['paths'], edges=env.dfr.get_w(), name=name, show=False)

    arr = np.zeros(shape=(len(param_lst), 2))
    arr[:, 0] = np.array(sr_stats).mean(axis=0)
    arr[:, 1] = np.array(gap_stats).mean(axis=0)
    df = pd.DataFrame(arr, index=get_index(), columns=['SR', 'GAP'])
    df_sorted = df.sort_values(by='SR')
    print(df_sorted)

    # plot_diff_distribution(ret=diff_dict,
    #                        arr=arr,
    #                        low=-10.0,
    #                        high=10.0,
    #                        bins=20,
    #                        index=index)


if __name__ == '__main__':
    main()


                     # SR       GAP
# Only Static(n=0)  1.0    0.417223
# Only Static(n=0)  1.0    0.404275
# Only Static(n=0)  1.0    0.409008

# Only Static(n=1)  0.842  1.862106
# Only Static(n=1)  0.867  1.634938
# Only Static(n=1)  0.883  1.472473

# Only Static(n=2)  0.867  1.584403
# Only Static(n=2)  0.87   1.551986
# Only Static(n=2)  0.87   1.552304

# Only Static(n=3)  0.915  1.113557
# Only Static(n=3)  0.902  1.242529
# Only Static(n=3)  0.888  1.390932

# Only Static(n=4)  0.909  1.20019
# Only Static(n=4)  0.913  1.147165
# Only Static(n=4)  0.917  1.106896

# Only Static(n=5)  0.905  1.231748
# Only Static(n=5)  0.882  1.44194
# Only Static(n=5)  0.899  1.280885
# Only Static(n=5)  0.896  1.320928
# Only Static(n=5)  0.905  1.225968

# Only Static(n=6)  0.922  1.050249
# Only Static(n=6)  0.914  1.142688
# Only Static(n=6)  0.917  1.110906

# Only Static(n=7)  0.938  0.914741
# Only Static(n=7)  0.929  1.027176
# Only Static(n=7)  0.927  1.054056

# Only Static(n=8)  0.914  1.137371
# Only Static(n=8)  0.934  0.957911
# Only Static(n=8)  0.909  1.197704

# Only Static(n=9)  0.98  0.546905
# Only Static(n=9)  0.95  0.90679
# Only Static(n=9)  0.91  1.18928
# Only Static(n=9)  0.94  0.882426