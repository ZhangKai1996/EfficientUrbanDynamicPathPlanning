import os

from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algo.trainer_dqn import DPPBSRSTrainer
from algo.baselines import dijkstra_path

from env.geo import load_graph
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
suffix = 'Single Sensitivity'
param_lst = [
    # {'exp_name': 'Only Static', 'a': -1.0, 'e': 50.0, 'zeta': 9.0, 'c': 0.0, 'beta': 0.0},
    # {'exp_name': 'Only Static', 'a': -1.0, 'e': 50.0, 'zeta': 8.0, 'c': 0.0, 'beta': 0.0},
    # {'exp_name': 'Only Static', 'a': -1.0, 'e': 50.0, 'zeta': 7.0, 'c': 0.0, 'beta': 0.0},
    {'exp_name': 'Only Static', 'a': -1.0, 'e': 50.0, 'zeta': 6.0, 'c': 0.0, 'beta': 0.0},

    # {'exp_name': 'Hybrid', 'a': -1.0, 'e': 50.0, 'zeta': 9.0, 'c': 0.0, 'beta': 1.0},
    # {'exp_name': 'Hybrid', 'a': -1.0, 'e': 50.0, 'zeta': 8.0, 'c': 0.0, 'beta': 1.0},
    # {'exp_name': 'Hybrid', 'a': -1.0, 'e': 50.0, 'zeta': 7.0, 'c': 0.0, 'beta': 1.0},
    # {'exp_name': 'Hybrid', 'a': -1.0, 'e': 50.0, 'zeta': 6.0, 'c': 0.0, 'beta': 1.0},
]

diff_dict = {j: [] for j, _ in enumerate(param_lst)}
pair_records = {j: [] for j, _ in enumerate(param_lst)}   # 新增：每步 (F_d, F_v, done, success)
all_ratios = np.zeros(shape=(5, len(param_lst), 20))


def get_index():
    index = []
    for param in param_lst:
        idx = param['exp_name']
        if 'Only Static' in idx:
            idx += '(zeta={},e={})'.format(param['zeta'], param['e'])
        elif 'No Dynamic' in idx:
            idx += '(a={},c={})'.format(param['a'], param['c'])
        elif 'No Clipping' in idx:
            idx += '(beta={})'.format(param['beta'])
        elif 'Hybrid' in idx:
            idx += '(zeta={},beta={})'.format(param['zeta'], param['beta'])
        elif 'Ours' in idx:
            idx += '(beta={},'.format(param['beta'])
            idx += 'a={},c={})'.format(param['a'], param['c'])
        index.append(idx)
    return index


def _make_bins(low=-10.0, high=10.0, n_bins=10):
    left_edges = np.linspace(low, 0.0, n_bins + 1)
    right_edges = np.linspace(0.0, high, n_bins + 1)
    edges = np.concatenate([left_edges, right_edges[1:]])

    bins = []
    for k in range(n_bins):
        bins.append((edges[k], edges[k + 1], True, False))
    for k in range(n_bins):
        a = edges[n_bins + k]
        b = edges[n_bins + 1 + k]
        if k == 0:
            bins.append((a, b, True, True))
        else:
            bins.append((a, b, False, True))
    return edges, bins


def _count_bins(F, bins):
    counts = np.zeros(len(bins), dtype=int)
    for i, (a, b, lc, rc) in enumerate(bins):
        mask = np.ones(len(F), dtype=bool)
        mask &= (F >= a) if lc else (F > a)
        mask &= (F <= b) if rc else (F < b)
        counts[i] = mask.sum()
    return counts


def plot_hist(ax, F, title=None, label=None, y_max=1.0,
              verbose=True,
              low=-10.0, high=10.0, n_bins=10):
    if title is not None: ax.set_title(title)
    if F is None: return None

    edges, bins = _make_bins(low, high, n_bins)
    counts = _count_bins(F, bins)
    total = counts.sum()
    ratios = counts / total if total else counts

    centers, widths = [], []
    for a, b, lc, rc in bins:
        centers.append((a + b) / 2)
        widths.append((b - a) * 0.95)

    bars = ax.bar(centers, ratios,
                  width=widths,
                  edgecolor='black',
                  align='center',
                  alpha=0.6,
                  label=label)
    if verbose:
        for bar, r in zip(bars, ratios):
            if r == 0: continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(), '{:.2%}'.format(r),
                    ha='center', va='bottom', fontsize=7, rotation=0)

    ax.set_xticks(edges)
    ax.set_xticklabels([int(x) for x in edges], rotation=45)
    ax.set_ylim(0, y_max)
    if label is not None: ax.legend()
    return ratios


def plot_diff_distribution(ret, index=None, **kwargs_):
    fig, axes = plt.subplots(len(ret), 5, figsize=(18, 9), constrained_layout=True)

    for i, (key, rows) in enumerate(ret.items()):
        label = index[i]
        F = np.clip(np.array(rows, dtype=np.float32), kwargs_['low'], kwargs_['high'])

        title = 'Original Reward' if i == 0 else None
        ratios = plot_hist(axes[i, 0], F[:, 0], title=title, label=label, y_max=1.0, **kwargs_)
        if ratios is not None: all_ratios[0, i, :] = ratios.copy()

        title = 'Static Potentials' if i == 0 else None
        ratios = plot_hist(axes[i, 1], F[:, 1], title=title, label=label, y_max=1.0, **kwargs_)
        if ratios is not None: all_ratios[1, i, :] = ratios.copy()

        data = None if 'Static' in label else F[:, 2]
        title = 'Dynamic Potentials' if i == 0 else None
        ratios = plot_hist(axes[i, 2], data, title=title, y_max=0.5, **kwargs_)
        if ratios is not None: all_ratios[2, i, :] = ratios.copy()

        data = None if 'Static' in label else F[:, 3]
        title = 'Hybrid Potentials' if i == 0 else None
        ratios = plot_hist(axes[i, 3], data, title=title, y_max=0.5, **kwargs_)
        if ratios is not None: all_ratios[3, i, :] = ratios.copy()

        title = 'Shaped Reward' if i == 0 else None
        ratios = plot_hist(axes[i, 4], F[:, 4], title=title, label=label, y_max=1.0, **kwargs_)
        if ratios is not None: all_ratios[4, i, :] = ratios.copy()

    fig.savefig('diff_distribution.png', dpi=300)
    plt.show()


def joint_sign_stats(records, exclude_done=True):
    """对每步 (F_d, F_v, done) 做 2x2 联合符号统计。"""
    rec = [r for r in records if (not r['done']) or (not exclude_done)]
    if len(rec) == 0: return None

    fd = np.array([r['fd'] for r in rec], dtype=np.float64)
    fv = np.array([r['fv'] for r in rec], dtype=np.float64)
    fd_sign = np.sign(fd)
    fv_sign = np.sign(fv)

    fd_pos = fd_sign > 0
    fd_neg = fd_sign < 0
    fv_pos = fv_sign > 0
    fv_neg = fv_sign < 0
    fv_zero = fv_sign == 0

    N_pp = int(np.sum(fd_pos & fv_pos))
    N_pn = int(np.sum(fd_pos & fv_neg))
    N_np = int(np.sum(fd_neg & fv_pos))
    N_nn = int(np.sum(fd_neg & fv_neg))
    N_p0 = int(np.sum(fd_pos & fv_zero))
    N_n0 = int(np.sum(fd_neg & fv_zero))

    total_fd_pos = N_pp + N_pn + N_p0
    total_fd_neg = N_np + N_nn + N_n0

    # 异号时 |fv|>|fd| 的比例
    pn_mask = fd_pos & fv_neg
    np_mask = fd_neg & fv_pos
    p_override_pn = float(np.mean(np.abs(fv[pn_mask]) > np.abs(fd[pn_mask]))) if pn_mask.any() else np.nan
    p_override_np = float(np.mean(np.abs(fv[np_mask]) > np.abs(fd[np_mask]))) if np_mask.any() else np.nan

    # 同号时 |fv|/|fd| 的中位数
    pp_mask = fd_pos & fv_pos
    nn_mask = fd_neg & fv_neg
    ratio_pp = float(np.median(np.abs(fv[pp_mask]) / np.maximum(np.abs(fd[pp_mask]), 1e-8))) if pp_mask.any() else np.nan
    ratio_nn = float(np.median(np.abs(fv[nn_mask]) / np.maximum(np.abs(fd[nn_mask]), 1e-8))) if nn_mask.any() else np.nan

    # 总方向一致率
    total = fd + fv
    agree = float(np.mean(np.sign(total) == fd_sign)) if len(total) else np.nan
    # 总方向被纠正的比例（fd>0 变负，或 fd<0 变正）
    flip_pos_to_neg = int(np.sum((fd_sign > 0) & (np.sign(total) < 0)))
    flip_neg_to_pos = int(np.sum((fd_sign < 0) & (np.sign(total) > 0)))

    return {
        'n_steps': len(rec),
        'N_pp': N_pp, 'N_pn': N_pn, 'N_np': N_np, 'N_nn': N_nn,
        'N_p0': N_p0, 'N_n0': N_n0,
        'P(fv>0|fd>0)': N_pp / max(1, total_fd_pos),
        'P(fv<0|fd>0)': N_pn / max(1, total_fd_pos),
        'P(fv>0|fd<0)': N_np / max(1, total_fd_neg),
        'P(fv<0|fd<0)': N_nn / max(1, total_fd_neg),
        'P(|fv|>|fd| | fd>0,fv<0)': p_override_pn,
        'P(|fv|>|fd| | fd<0,fv>0)': p_override_np,
        'median|fv|/|fd| | ++': ratio_pp,
        'median|fv|/|fd| | --': ratio_nn,
        'total_sign_agree': agree,
        'flip_pos_to_neg': flip_pos_to_neg,
        'flip_neg_to_pos': flip_neg_to_pos,
    }


def test_dqn(env, trainer, root, max_step=100, render=False):
    gaps, srs, lens = [], [], []
    costs, rews, dists = [], [], []
    paths = {}
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
            diff_dynamic = trainer.potential_diff(state, dyn, mask, next_state, next_dyn, next_mask, done_rl)
            if 'Static' not in param['exp_name']:
                shaped_reward += diff_dynamic
            diff_dict[j].append([original_reward, diff_static, diff_dynamic, diff_static+diff_dynamic, shaped_reward])

            pair_records[j].append({'fd': float(diff_static),
                                    'fv': float(diff_dynamic),
                                    'done': bool(done_rl),})

            path_rl.append(env.cur_node)
            total_cost_rl += cost
            total_shaped_rew_rl += shaped_reward
            total_origin_rew_rl += original_reward
            episode_step += 1

            state, dyn, mask = next_state, next_dyn, next_mask
            if episode_step >= max_step: break

        gap = -1.0
        if done_rl:
            gap = (total_cost_rl - total_cost_di) / total_cost_di
        gaps.append(gap)
        costs.append(total_cost_rl)
        lens.append(len(path_rl))
        srs.append(float(done_rl))
        rews.append(total_origin_rew_rl)
        dist_final = haversine(env.graph.nodes[env.cur_node], env.graph.nodes[env.end_node])
        dists.append(dist_final)
        paths[param['exp_name']+f'_{j}'] = [end_node, total_cost_rl, total_shaped_rew_rl, path_rl, done_rl]
    return {'o': start_node,
            'd': end_node,
            'paths': paths,
            'result': [gaps, srs, lens, costs, rews, dists]}


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
    render = True
    num_episodes = 100

    gap_stats, sr_stats, len_stats = [], [], []
    cost_stats, rew_stats, dist_stats = [], [], []
    for i in tqdm(range(num_episodes), desc='Testing ...'):
        ret = test_dqn(env, trainer, root, max_step=env_kwargs['max_step'], render=render)
        gap_stats.append(ret['result'][0])
        sr_stats.append(ret['result'][1])
        len_stats.append(ret['result'][2])
        cost_stats.append(ret['result'][3])
        rew_stats.append(ret['result'][4])
        dist_stats.append(ret['result'][5])

        if not render: continue
        if ret['o'] is None: continue

        name = png_folder + '{}_{}_{}'.format(ret['o'], ret['d'], i+1)
        env.render(paths=ret['paths'], edges=env.dfr.get_w(), name=name, show=False)

    ids = get_index()
    sr_arr = np.array(sr_stats)
    gap_arr = np.array(gap_stats)
    len_arr = np.array(len_stats)
    rew_arr = np.array(rew_stats)
    cost_arr = np.array(cost_stats)
    dist_arr = np.array(dist_stats)

    arr = np.zeros(shape=(len(param_lst), 6))
    arr[:, 0] = rew_arr.mean(axis=0)
    arr[:, 1] = sr_arr.mean(axis=0)
    for j in range(len(param_lst)):
        mask = list(set(np.where(gap_arr[:, j] != -1.0)[0]))
        arr[j, 2] = gap_arr[mask, j].mean(axis=0)
        mask = list(set(np.where(len_arr[:, j] != -1)[0]))
        arr[j, 3] = len_arr[mask, j].mean(axis=0)
    arr[:, 4] = cost_arr.mean(axis=0)
    arr[:, 5] = dist_arr.mean(axis=0)

    df = pd.DataFrame(arr, index=ids, columns=['Reward', 'SR', 'GAP', 'Length', 'Cost', 'Distance'])
    print(df)
    plot_diff_distribution(diff_dict, index=ids, low=-10.0, high=10.0, n_bins=10)

    tags = ['original_reward', 'diff_static', 'diff_dynamic', 'diff_hybrid', 'shaped_reward']
    columns = ['[{},{})'.format(x, x+1) for x in np.linspace(-10.0,9.0, num=20)]
    print(columns)
    for i, tag in enumerate(tags):
        df = pd.DataFrame(all_ratios[i], index=ids, columns=columns)
        print(tag)
        print(df)

    # 新增：联合符号统计（只对 Hybrid 有意义，Static 的 fv 恒为 0）
    print('\n===== Joint sign statistics (F_d vs F_v) =====')
    joint_rows = []
    for j, param in enumerate(param_lst):
        if 'Static' in param['exp_name']: continue
        stats = joint_sign_stats(pair_records[j], exclude_done=True)
        if stats is None: continue

        stats_row = {'config': ids[j], **stats}
        joint_rows.append(stats_row)
        print('\nConfig:', ids[j])
        for k, v in stats.items():
            print('  {:>28s}: {}'.format(k, v))

    if joint_rows:
        joint_df = pd.DataFrame(joint_rows).set_index('config')
        joint_df.to_csv('joint_sign_stats.csv', index=True, encoding='utf-8-sig')
        print('\nJoint sign statistics saved to joint_sign_stats.csv')
        print(joint_df)


if __name__ == '__main__':
    main()