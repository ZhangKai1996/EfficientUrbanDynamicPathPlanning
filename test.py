import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from parameters import alpha, gamma
from parameters import n, radius, length_limit, env_id, env_kwargs

from env.geo import load_graph, get_hop_neighbors
from env.core import DFRGenerator
from env.environment import DPPEnvironment


def collect_edge_potential_diff(env):
    rows = []
    for (u, v) in env.graph.edges(data=False):
        env.a, env.c = -1.0, 0.0
        shaped_rew, cost = env.calculate_potential(u, v, verbose=False)
        delta1 = shaped_rew + cost

        env.a, env.c = 1.0, -1.0
        shaped_rew, cost = env.calculate_potential(u, v, verbose=False)
        delta2 = shaped_rew + cost

        rows.append(dict(u=u, v=v, F1=float(delta1), F2=float(delta2)))
    return rows


def summarize(rows, low=-10.0, key='F', high=10.0, n_bins=20):
    F_raw = np.array([r[key] for r in rows], dtype=float)
    F_clip = np.clip(F_raw, low, high)

    # 2) 等分 n_bins 个区间,宽度固定
    edges = np.linspace(low, high, n_bins + 1)          # 21 个边界
    counts, _ = np.histogram(F_clip, bins=edges)     # 最后一个区间右闭,正好包含被截到 hi 的点

    # 3) 打印每格
    print('边总数:', len(F_raw))
    print('截断区间: [{}, {}], 分 {} 份, 每份宽度 {:.4f}'.format(low, high, n_bins, (high - low) / n_bins))
    print('{:<6} {:<22} {:>10} {:>10}'.format('bin', 'range', 'count', 'ratio'))
    for k in range(n_bins):
        a, b = edges[k], edges[k + 1]
        right = ']' if k == n_bins - 1 else ')'
        rng = '[{:.2f}, {:.2f}{}'.format(a, b, right)
        cnt = counts[k]
        ratio = cnt / len(F_raw) if len(F_raw) else 0.0
        print('{:<6} {:<22} {:>10d} {:>10.2%}'.format(k, rng, cnt, ratio))

    # 4) 附:被截断的数量(超出 [-10,10] 的原始值)
    n_high = int(np.sum(F_raw > high))
    n_low = int(np.sum(F_raw < high))
    print('\n超出上界(>{}) 被计入 {}: {} 条'.format(high, high, n_high))
    print('低于下界(<{}) 被计入 {}: {} 条'.format(high, high, n_low))
    print('未截断条数: {}'.format(len(F_raw) - n_high - n_low))
    return F_clip


def plot_hist(ax, F, low=-10.0, high=10.0, n_bins=20):
    edges = np.linspace(low, high, n_bins + 1)
    counts, _ = np.histogram(F, bins=edges)
    total = counts.sum()
    ratios = counts / total if total else counts

    # 每格中心作为横坐标位置
    centers = (edges[:-1] + edges[1:]) / 2
    width = (high - low) / n_bins

    bars = ax.bar(centers, ratios, width=width * 0.95, edgecolor='black', align='center')

    # 在每根柱子上方显示占比(百分比)和计数
    for bar, r, c in zip(bars, ratios, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                '{:.2%}\n({})'.format(r, c),
                ha='center', va='bottom', fontsize=7, rotation=0)

    ax.set_xlabel(r"$F(s,s')=\gamma\Phi_d(s')-\Phi(s)$ (clipped to $[-10,10]$)")
    ax.set_ylabel('ratio')  # 纵坐标变为占比
    ax.set_title('Distribution of F over all edges ({} bins, ratio)'.format(n_bins))
    ax.set_xticks(edges)
    ax.set_xticklabels(['{:.0f}'.format(e) for e in edges], rotation=45)

    # 纵轴用百分比格式
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    # 给顶部留出空间放标签
    ymax = ratios.max() if len(ratios) else 0
    ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1)


def main():
    args = load_graph(radius=radius, **env_kwargs)
    graph, p, num_action, od_pairs, *_ = args

    hop_neighborhoods = get_hop_neighbors(graph, max_n=n, limit=length_limit)
    dfr_generator = DFRGenerator(graph, neighbors=hop_neighborhoods, length_limit=length_limit)

    env = DPPEnvironment(graph, p, num_action,
                         zeta=5.0,
                         alpha=alpha,
                         gamma=gamma,
                         dfr_generator=dfr_generator,
                         od_pairs=od_pairs)

    env.reset(render=False, reuse=False)

    rows = collect_edge_potential_diff(env)
    F1 = summarize(rows, key='F1', low=-10.0, high=10.0, n_bins=20)
    F2 = summarize(rows, key='F2', low=-10.0, high=10.0, n_bins=20)

    fig, axes = plt.subplots(2, 1, figsize=(18, 9), constrained_layout=True)
    plot_hist(axes[0], F1, low=-10.0, high=10.0, n_bins=20)
    plot_hist(axes[1], F2, low=-10.0, high=10.0, n_bins=20)
    plt.tight_layout()
    plt.savefig('F_hist_20bins.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()