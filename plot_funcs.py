import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def _make_bins(low=-10.0, high=10.0, n_bins=10):
    left_edges = np.linspace(low, 0.0, n_bins + 1)      # n_bins+1 个点
    right_edges = np.linspace(0.0, high, n_bins + 1)    # n_bins+1 个点
    edges = np.concatenate([left_edges, right_edges[1:]])  # 长度 2*n_bins+1

    bins = []
    # 左侧 n_bins 格: [edges[k], edges[k+1])
    for k in range(n_bins):
        bins.append((edges[k], edges[k + 1], True, False))
    # 右侧 n_bins 格: 第一格 [0,1]（左闭右闭），其余 (a,a+1]
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
    print(ratios)
    for bar, r in zip(bars, ratios):
        if r == 0: continue
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(), '{:.2%}'.format(r),
                ha='center', va='bottom', fontsize=7, rotation=0)

    if title is not None: ax.set_title(title)

    ax.set_xticks(edges)
    ax.set_xticklabels([int(x) for x in edges], rotation=45)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    y_max = ratios.max() if len(ratios) else 0
    ax.set_ylim(0, y_max * 1.25 if y_max > 0 else 1)


def plot_diff_distribution(ret, arr, low=-1.0, high=1.0, bins=10, index=None):
    fig, axes = plt.subplots(len(ret), 2, figsize=(18, 9), constrained_layout=True)
    for i, (key, rows) in enumerate(ret.items()):
        title = index[i]
        title += '(SR:{:>5.3f},'.format(arr[i, 1])
        title += 'Gap:{:>5.3f},'.format(arr[i, 2])
        title += 'Length:{:>5.1f})'.format(arr[i, 4])

        F = np.clip(np.array(rows, dtype=np.float32), low, high)
        plot_hist(axes[i, 0], F, low=low, high=high, n_bins=bins, title=title)

        # F = summarize(rows_arr[:, 1], low=-10.0, high=10.0, n_bins=10)
        # plot_hist(axes[i, 1], F, low=-10.0, high=10.0, n_bins=10, title=title)
        #
        # F = summarize(rows_arr[:, 2], low=-10.0, high=10.0, n_bins=10)
        # plot_hist(axes[i, 2], F, low=-10.0, high=10.0, n_bins=10, title=title)

    plt.tight_layout()
    plt.savefig('F_hist_20bins.png', dpi=300)
    plt.show()


def plot_test_performance(ret, ):
    fig, axes = plt.subplots(1, len(tags), figsize=(20, 4), constrained_layout=True)
    xs = [param['exp_name']
          + r'($\zeta$={},'.format(param['zeta'])
          + r'$\beta$={},'.format(param['beta'])
          + r'$a$={})'.format(param['a']) for param in param_lst]
    x_pos = list(range(len(param_lst)))

    ax = axes[0]
    valid = [ret[j][tags[0]] for j, _ in enumerate(param_lst)]
    positions = [k for k, g in enumerate(valid) if len(g) > 0]
    valid_data = [g for g in valid if len(g) > 0]
    if valid_data:
        bp = ax.boxplot(valid_data, positions=positions, showfliers=False,
                        patch_artist=True, showmeans=True, widths=0.6)
        for box in bp['boxes']:
            box.set(facecolor='tab:blue', alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xs, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(tags[0], fontsize=14)
    ax.set_title(tags[0])
    ax.grid(True, linestyle='--', alpha=0.25, linewidth=0.8)

    ax = axes[1]
    success_rates = []
    for j, param in enumerate(param_lst):
        vals = ret[j][tags[1]]
        success_rates.append(sum(vals) / len(vals) if len(vals) > 0 else 0.0)
    ax.bar(x_pos, success_rates, width=0.6, color='tab:green', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xs, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(tags[1], fontsize=14)
    ax.set_title(tags[1])
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.25, linewidth=0.8)

    ax = axes[2]
    valid = [ret[j][tags[2]] for j, _ in enumerate(param_lst)]
    positions = [k for k, g in enumerate(valid) if len(g) > 0]
    valid_data = [g for g in valid if len(g) > 0]
    if valid_data:
        bp = ax.boxplot(valid_data, positions=positions, showfliers=False,
                        patch_artist=True, showmeans=True, widths=0.6)
        for box in bp['boxes']:
            box.set(facecolor='tab:orange', alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xs, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(tags[2], fontsize=14)
    ax.set_title(tags[2])
    ax.grid(True, linestyle='--', alpha=0.25, linewidth=0.8)

    ax = axes[3]
    valid = [ret[j][tags[3]] for j, _ in enumerate(param_lst)]
    positions = [k for k, g in enumerate(valid) if len(g) > 0]
    valid_data = [g for g in valid if len(g) > 0]
    if valid_data:
        bp = ax.boxplot(valid_data, positions=positions, showfliers=False,
                        patch_artist=True, showmeans=True, widths=0.6)
        for box in bp['boxes']:
            box.set(facecolor='tab:red', alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xs, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(tags[3], fontsize=14)
    ax.set_title(tags[3])
    ax.grid(True, linestyle='--', alpha=0.25, linewidth=0.8)

    ax = axes[4]
    valid = [ret[j][tags[4]] for j, _ in enumerate(param_lst)]
    positions = [k for k, g in enumerate(valid) if len(g) > 0]
    valid_data = [g for g in valid if len(g) > 0]
    if valid_data:
        bp = ax.boxplot(valid_data, positions=positions, showfliers=False,
                        patch_artist=True, showmeans=True, widths=0.6)
        for box in bp['boxes']:
            box.set(facecolor='tab:red', alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xs, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(tags[4], fontsize=14)
    ax.set_title(tags[4])
    ax.grid(True, linestyle='--', alpha=0.25, linewidth=0.8)

    # fig.savefig(root+'test_stats.pdf', dpi=400)
    plt.show()
