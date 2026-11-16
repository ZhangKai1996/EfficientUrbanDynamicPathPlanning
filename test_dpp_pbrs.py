import os

from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algo.trainer_dqn import DPPBSRSTrainer
from algo.baselines import dijkstra_path

from env.geo import load_graph, get_hop_neighbors
from env.core import DFRGenerator
from env.environment import DPPEnvironment
from env.utils import haversine

from parameters import alpha, gamma, n, n_sector
from parameters import radius, length_limit, env_id, env_kwargs, kwargs

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

seed = 1234
suffix = 'Sensitivity'
param_lst = [
    {'exp_name': 'Only Static', 'a': -1.0, 'e': 100.0, 'zeta': 5.0, 'c': 0.0, 'beta': 0.0},
    {'exp_name': 'Only Static', 'a': -1.0, 'e': 100.0, 'zeta': 4.0, 'c': 0.0, 'beta': 0.0},
    {'exp_name': 'Only Static', 'a': -1.0, 'e': 100.0, 'zeta': 3.0, 'c': 0.0, 'beta': 0.0},
    {'exp_name': 'Only Static', 'a': -1.0, 'e': 100.0, 'zeta': 2.0, 'c': 0.0, 'beta': 0.0},
    {'exp_name': 'Only Static', 'a': -1.0, 'e': 100.0, 'zeta': 1.0, 'c': 0.0, 'beta': 0.0},

    {'exp_name': 'Only Static', 'a': -1.0, 'e': 50.0, 'zeta': 5.0, 'c': 0.0, 'beta': 0.0},
    {'exp_name': 'Only Static', 'a': -1.0, 'e': 50.0, 'zeta': 4.0, 'c': 0.0, 'beta': 0.0},
    {'exp_name': 'Only Static', 'a': -1.0, 'e': 50.0, 'zeta': 3.0, 'c': 0.0, 'beta': 0.0},
    {'exp_name': 'Only Static', 'a': -1.0, 'e': 50.0, 'zeta': 2.0, 'c': 0.0, 'beta': 0.0},
    {'exp_name': 'Only Static', 'a': -1.0, 'e': 50.0, 'zeta': 1.0, 'c': 0.0, 'beta': 0.0},

    {'exp_name': 'Only Static', 'a': -1.0, 'e': 10.0, 'zeta': 5.0, 'c': 0.0, 'beta': 0.0},
    {'exp_name': 'Only Static', 'a': -1.0, 'e': 10.0, 'zeta': 4.0, 'c': 0.0, 'beta': 0.0},
    {'exp_name': 'Only Static', 'a': -1.0, 'e': 10.0, 'zeta': 3.0, 'c': 0.0, 'beta': 0.0},
    {'exp_name': 'Only Static', 'a': -1.0, 'e': 10.0, 'zeta': 2.0, 'c': 0.0, 'beta': 0.0},
    # {'exp_name': 'Only Static', 'a': -1.0, 'e': 10.0, 'zeta': 1.0, 'c': 0.0, 'beta': 0.0},
]


diff_dict = {j: [] for j, _ in enumerate(param_lst)}


def get_index():
    index = []
    for param in param_lst:
        idx = param['exp_name']
        if 'Only Static' in idx:
            idx += '(zeta={},e={})'.format(param['zeta'],param['e'])
        elif 'No Dynamic' in idx:
            idx += '(a={},c={})'.format(param['a'], param['c'])
        elif 'No Clipping' in idx:
            idx += '(beta={})'.format(param['beta'])
        elif 'Ours' in idx:
            idx += '(beta={},'.format(param['beta'])
            idx += 'a={},c={})'.format(param['a'], param['c'])
        index.append(idx)
    return index


def _make_bins(low=-10.0, high=10.0, n_bins=10):
    left_edges = np.linspace(low, 0.0, n_bins + 1)      # n_bins+1 个点
    right_edges = np.linspace(0.0, high, n_bins + 1)    # n_bins+1 个点
    edges = np.concatenate([left_edges, right_edges[1:]])  # 长度 2*n_bins+1

    bins = []
    for k in range(n_bins):
        bins.append((edges[k], edges[k + 1], True, False))
    for k in range(n_bins):
        a = edges[n_bins + k]
        b = edges[n_bins + 1 + k]
        if k == 0:
            bins.append((a, b, True, True))      # [0,1]
        else:
            bins.append((a, b, False, True))     # (a,b]
    return edges, bins


def _count_bins(F, bins):
    counts = np.zeros(len(bins), dtype=int)
    for i, (a, b, lc, rc) in enumerate(bins):
        mask = np.ones(len(F), dtype=bool)
        mask &= (F >= a) if lc else (F > a)
        mask &= (F <= b) if rc else (F < b)
        counts[i] = mask.sum()
    return counts


def plot_hist(ax, F, low=-10.0, high=10.0, n_bins=10, title=None):
    edges, bins = _make_bins(low, high, n_bins)
    counts = _count_bins(F, bins)
    total = counts.sum()
    ratios = counts / total if total else counts

    centers, widths = [], []
    for a, b, lc, rc in bins:
        centers.append((a + b) / 2)
        widths.append((b - a) * 0.95)

    bars = ax.bar(centers, ratios, width=widths, edgecolor='black', align='center')
    for bar, r in zip(bars, ratios):
        if r == 0: continue
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(), '{:.2%}'.format(r),
                ha='center', va='bottom', fontsize=7, rotation=0)

    if title is not None: ax.set_title(title)

    ax.set_xticks(edges)
    ax.set_xticklabels([int(x) for x in edges], rotation=45)

    y_max = ratios.max() if len(ratios) else 0
    ax.set_ylim(0, y_max * 1.25 if y_max > 0 else 1)


def plot_diff_distribution(ret, arr, low=-1.0, high=1.0, bins=10, index=None):
    fig, axes = plt.subplots(len(ret), 1, figsize=(18, 9), constrained_layout=True)
    for i, (key, rows) in enumerate(ret.items()):
        title = index[i]
        title += '(SR:{:>5.3f},'.format(arr[i, 0])
        title += 'Gap:{:>5.3f},'.format(arr[i, 1])
        title += 'Length:{:>5.1f})'.format(arr[i, 2])

        F = np.clip(np.array(rows, dtype=np.float32), low, high)
        plot_hist(axes[i], F, low=low, high=high, n_bins=bins, title=title)

    # plt.tight_layout()
    plt.show()


def test_dqn(env, trainer, root, max_step=100, render=False):
    gaps, srs, lens, paths = [], [], [], {}
    start_node, end_node = None, None
    for j, param in enumerate(param_lst):
        env.e, env.zeta = param['e'], param['zeta']

        num_episodes = int(kwargs['num_episodes_train'] / 1e3)
        folder = root + param['exp_name']
        folder += '({}_{}_{}_{}_{}_{}_{})/'.format(param['a'],
                                                   param['e'],
                                                   param['zeta'],
                                                   param['c'],
                                                   param['beta'],
                                                   max_step,
                                                   num_episodes)
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
            diff_dict[j].append(diff_static)

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

        gap, length = -1.0, -1
        if done_rl:
            gap = (total_cost_rl - total_cost_di) / total_cost_di
            length = len(path_rl)
        gaps.append(gap)
        lens.append(length)
        srs.append(float(done_rl))
        paths[param['exp_name']+f'_{j}'] = [end_node, total_cost_rl, total_shaped_rew_rl, path_rl, done_rl]
    return {'o': start_node,
            'd': end_node,
            'paths': paths,
            'result': [gaps, srs, lens]}


def main():
    args = load_graph(radius=radius, n_sector=n_sector, **env_kwargs)
    graph, p, num_action, od_pairs, *_ = args

    dfr_generator = DFRGenerator(graph, n=n, length_limit=length_limit)
    root = 'trained/exp_{}_{}_{}_{}_{}_{}_{}_{}/'.format(env_id, int(radius),
                                                           int(length_limit[0]),
                                                           int(length_limit[1]),
                                                           n, n_sector,
                                                           suffix, seed)
    png_folder = root + 'pictures/'
    if not os.path.exists(png_folder): os.mkdir(png_folder)

    env = DPPEnvironment(graph, p, num_action,
                         alpha=alpha,
                         gamma=gamma,
                         dfr_generator=dfr_generator,
                         od_pairs=od_pairs)
    trainer = DPPBSRSTrainer(graph, p, num_action, gamma=gamma, dyn_dim=env.dfr.dyn_dim)
    render = False
    num_episodes = 1000

    gap_stats, sr_stats, len_stats = [], [], []
    for i in tqdm(range(num_episodes), desc='Testing ...'):
        ret = test_dqn(env, trainer, root, max_step=env_kwargs['max_step'], render=render)

        if ret['o'] is not None and render:
            name = png_folder + '{}_{}_{}'.format(ret['o'], ret['d'], i+1)
            env.render(paths=ret['paths'], edges=env.dfr.get_w(), name=name, show=False)
        gap_stats.append(ret['result'][0])
        sr_stats.append(ret['result'][1])
        len_stats.append(ret['result'][2])

    sr_arr = np.array(sr_stats)
    gap_arr = np.array(gap_stats)
    len_arr = np.array(len_stats)
    print(sr_arr.shape, sr_arr.mean(axis=0).shape)

    arr = np.zeros(shape=(len(param_lst), 3))
    arr[:, 0] = sr_arr.mean(axis=0)
    for j in range(len(param_lst)):
        mask = list(set(np.where(gap_arr[:, j] != -1.0)[0]))
        arr[j, 1] = gap_arr[mask, j].mean(axis=0)
        mask = list(set(np.where(len_arr[:, j] != -1)[0]))
        arr[j, 2] = len_arr[mask, j].mean(axis=0)

    ids = get_index()
    df = pd.DataFrame(arr, index=ids, columns=['SR', 'GAP', 'Length'])
    # df_sorted = df.sort_values(by='GAP')
    print(df)

    plot_diff_distribution(diff_dict, arr, low=-10.0, high=10.0, bins=10, index=ids)

    if gap_arr.shape[1] <= 1: return

    gap_arr_copy = gap_arr.copy()
    fig, axes = plt.subplots(1, len(param_lst)-1, figsize=(18, 9), constrained_layout=True)
    mask0 = set(np.where(gap_arr[:, 0] != -1.0)[0])
    for i in range(1, gap_arr.shape[1]):
        ids.append('Delta{}'.format(i))
        deltas = gap_arr[:, 0] - gap_arr[:, i]
        gap_arr_copy = np.hstack([gap_arr_copy, deltas.reshape(-1, 1)])

        mask = set(np.where(gap_arr[:, i] != -1.0)[0])
        mask = list(mask0 & mask)

        deltas_part = deltas[mask]

        tmp = np.hstack([gap_arr[mask, 0].reshape(-1, 1),
                         gap_arr[mask, i].reshape(-1, 1),
                         deltas_part.reshape(-1, 1)])
        df = pd.DataFrame(tmp, columns=['0', '1', 'delta'])
        # print(df.to_string(float_format=lambda x: f'{x:.4f}'))

        vals = np.sort(deltas_part)
        mean_val = vals.mean()

        ax = axes[i-1]
        ax.plot(vals, marker='o', linestyle='-', markersize=3, label='Delta (sorted)')
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(mean_val, color='red', linewidth=1.2, linestyle='-', label=f'Mean = {mean_val:.4f}')
        ax.set_xlabel('Index')
        ax.set_ylabel('Deltas')
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=-1.0, top=1.0)

    df = pd.DataFrame(gap_arr_copy, columns=ids)
    # print(df.to_string(float_format=lambda x: f'{x:.4f}'))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
