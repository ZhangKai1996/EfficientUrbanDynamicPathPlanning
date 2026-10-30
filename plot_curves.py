import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

from env.utils import moving_average
from parameters import n, get_kwargs, length_limit

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

tags = ['ORew', 'SR', 'Cost', 'Loss', 'Mean Step']
tags_ = ['Reward', 'Success Rate', 'Cost', 'Loss', 'Length']
lim_dict = {
    # 'SR': [-0.05, 1.05],
    # 'MGap': [0.15, 1.05],
    # 'ORew': [-35.0, None]
}
legend_pos = {
    'SR': 'right',
    'MGap': 'upper right',
    # 'ORew': [-35.0, None]
}


def read_tag(folder, tag, norm=False):
    print(folder, tag, os.path.exists(folder))

    event_file = glob.glob(os.path.join(folder, "**/events.out.tfevents.*"), recursive=True)[0]
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    data = np.array([x.value for x in ea.Scalars(tag)])
    if norm: data -= data[0]
    return data


def plot_train_curve(root, max_step, param_lst, name=None):
    fig, axes = plt.subplots(2, len(tags), figsize=(18, 9), constrained_layout=True)

    data_dict = {tag: [] for tag in tags}
    arr = np.zeros(shape=(len(param_lst), len(tags)))
    index = None
    for j, tag in enumerate(tags):
        lims = lim_dict[tag] if tag in lim_dict.keys() else None
        loc = legend_pos[tag] if tag in legend_pos.keys() else None

        ax = axes[0, j]
        index = []
        for i, param in enumerate(param_lst):
            folder = root + param['exp_name']
            folder += '({}_{}_{}_{}_{}_{}_{})/'.format(param['a'],
                                                          param['e'],
                                                          param['zeta'],
                                                          param['c'],
                                                          param['beta'],
                                                          param['n'],
                                                          max_step)
            print(folder, os.path.exists(folder))
            idx = param['exp_name']
            if 'Only Static' in idx:
                idx += '(n={})'.format(param['n'])
            elif 'No Dynamic' in idx:
                idx += '(a={},c={})'.format(param['a'],param['c'])
            elif 'No Clipping' in idx:
                idx += '(beta={})'.format(param['beta'])
            elif 'Ours' in idx:
                idx += '(beta={},'.format(param['beta'])
                idx += 'a={},c={})'.format(param['a'],param['c'])
            index.append(idx)

            if not os.path.exists(folder): continue
            data = read_tag(folder, tag)
            data_dict[tag].append(data[-100:])

            arr[i, j] = np.mean(data[-100:])
            # label = param['exp_name']
            # label += r'($a$={})'.format(param['a'])

            ax.plot(moving_average(data, window_size=6), alpha=0.6, label=idx)
            if lims is not None: ax.set_ylim(bottom=lims[0], top=lims[1])
            # if j == 0: ax.set_title(map_name, fontweight='bold', fontsize=20)
            ax.set_xlabel('Training Episodes (x100)', fontsize=16)
            ax.set_ylabel(tags_[j], fontsize=16)

            ax.legend(loc=loc, fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.25, linewidth=0.8)

    # plt.tight_layout()
    # fig.savefig(name+'.pdf', dpi=400)
    # plt.show()

    # xs = [x.split(',')[-2] for x in index]
    xs = [x for x in index]
    for j, tag in enumerate(tags):
        lims = lim_dict[tag] if tag in lim_dict.keys() else None

        ax = axes[1, j]
        groups = data_dict[tag]
        positions = [k for k, g in enumerate(groups) if len(g) > 0]
        valid = [g for g in groups if len(g) > 0]

        if len(valid) == 0:
            ax.set_title(tag + ' (no data)')
            continue

        bp = ax.boxplot(valid,
                        positions=positions,
                        showfliers=False,  # 不画异常点（离群点）
                        patch_artist=True,  # 箱子填充颜色
                        showmeans=True,  # 显示均值
                        widths=0.6)
        for box in bp['boxes']:
            box.set(facecolor='tab:blue', alpha=0.5)

        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=45, ha='right', fontsize=8)
        if lims is not None: ax.set_ylim(bottom=lims[0], top=lims[1])
        ax.set_ylabel(tags_[j], fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.25, linewidth=0.8)

    plt.show()
    if name is not None: fig.savefig(name+'.pdf', dpi=400)

    df = pd.DataFrame(arr, index=index, columns=tags_)
    df_sorted = df.sort_values(by='Cost')
    print(df_sorted)
    # print(df)


def main():
    seed = 1234
    suffix = 'Dynamic Feature'
    param_lst = [
        {'exp_name': 'Only Static', 'a': -1.0, 'e': 60.0, 'zeta': 5.0, 'c': 0.0, 'beta': 0.0, 'n': 0},
        {'exp_name': 'Only Static', 'a': -1.0, 'e': 60.0, 'zeta': 5.0, 'c': 0.0, 'beta': 0.0, 'n': 2},
        {'exp_name': 'Only Static', 'a': -1.0, 'e': 60.0, 'zeta': 5.0, 'c': 0.0, 'beta': 0.0, 'n': 4},
        {'exp_name': 'Only Static', 'a': -1.0, 'e': 60.0, 'zeta': 5.0, 'c': 0.0, 'beta': 0.0, 'n': 6},
        {'exp_name': 'Only Static', 'a': -1.0, 'e': 60.0, 'zeta': 5.0, 'c': 0.0, 'beta': 0.0, 'n': 8},
        {'exp_name': 'Only Static', 'a': -1.0, 'e': 60.0, 'zeta': 5.0, 'c': 0.0, 'beta': 0.0, 'n': 10},
        {'exp_name': 'Only Static', 'a': -1.0, 'e': 60.0, 'zeta': 5.0, 'c': 0.0, 'beta': 0.0, 'n': 12},
    ]

    env_id, radius, max_step = 2, -1, 200
    # env_id, radius, max_step = 3, -1, 250
    # env_id, radius, max_step = 1, -1, 250

    _, env_kwargs, _ = get_kwargs(env_id_=env_id, radius_=radius)

    root = 'trained/exp_{}_{}_{}_{}_{}_{}/'.format(env_id, int(radius),
                                                   int(length_limit[0]),
                                                   int(length_limit[1]),
                                                   suffix, seed)
    plot_train_curve(root, max_step, param_lst=param_lst)


if __name__ == '__main__':
    main()
