import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# ---------------- Data ----------------
configs = [
    'Static ζ=6', 'Static ζ=7', 'Static ζ=8', 'Static ζ=9',
    'Hybrid ζ=6', 'Hybrid ζ=7', 'Hybrid ζ=8', 'Hybrid ζ=9',
]
metrics = ['Reward', 'SR', 'Cost', 'Length', 'Distance']

train = np.array([
    [-37.5676, 0.7923, 116.7976, 122.1181, 9.6423],
    [-16.4593, 0.8864, 105.0993, 102.8221, 4.3911],
    [-26.5716, 0.8699, 113.5616, 113.6781, 5.0576],
    [-53.4371, 0.7704, 130.4771, 139.1529, 10.5447],
    [9.4554, 0.9966, 90.2046, 72.4182, 0.1084],
    [9.5065, 0.9970, 90.1935, 71.7420, 0.0619],
    [9.3410, 0.9973, 90.3890, 71.2911, 0.0534],
    [7.6335, 0.9972, 92.0865, 72.3353, 0.0451],
])

test = np.array([
    [-49.6436, 0.691, 118.7436, 132.091, 13.9269],
    [-35.7321, 0.741, 109.8321, 120.697, 11.4562],
    [-55.9046, 0.671, 123.0046, 137.680, 15.1331],
    [-84.6498, 0.555, 140.1498, 164.714, 21.2280],
    [14.7598, 0.972, 82.4402, 67.108, 0.7881],
    [11.1466, 0.966, 85.4534, 68.398, 1.3511],
    [15.4386, 0.976, 82.1614, 65.745, 0.8848],
    [11.5449, 0.963, 84.7551, 69.899, 1.2143],
])

colors_static = '#4C72B0'
colors_hybrid = '#DD8452'
bar_colors_train = ['#4C72B0'] * 4 + ['#DD8452'] * 4
bar_colors_test = ['#8FB1D9'] * 4 + ['#F2B48C'] * 4

# ---------------- 1. Grouped bar chart: train vs test per metric ----------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
x = np.arange(len(configs))
w = 0.38
for ax, mi in zip(axes.flat, range(len(metrics))):
    ax.bar(x - w / 2, train[:, mi], w, label='Train', color=bar_colors_train, edgecolor='black', linewidth=0.5)
    ax.bar(x + w / 2, test[:, mi], w, label='Test', color=bar_colors_test, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_title(metrics[mi])
    if metrics[mi] == 'SR':
        ax.set_ylim(0, 1.05)
    if metrics[mi] == 'Distance':
        ax.set_yscale('log')
        ax.set_ylabel('log scale')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8)
axes.flat[-1].axis('off')
fig.suptitle('Train vs Test: Grouped Bar Chart per Metric', fontsize=14)
fig.tight_layout()
fig.savefig('fig1_grouped_bar.png', dpi=300)
plt.show()

# ---------------- 2. Dumbbell chart: train to test shift ----------------
fig, axes = plt.subplots(1, len(metrics), figsize=(22, 6), sharey=True)
y = np.arange(len(configs))
for ax, mi in zip(axes, range(len(metrics))):
    for i in range(len(configs)):
        c = colors_static if i < 4 else colors_hybrid
        ax.plot([train[i, mi], test[i, mi]], [i, i], color=c, lw=2, alpha=0.7)
        ax.scatter(train[i, mi], i, color=c, marker='o', s=50, label='Train' if i == 0 else None)
        ax.scatter(test[i, mi], i, color=c, marker='s', s=50, label='Test' if i == 0 else None)
    ax.set_title(metrics[mi])
    ax.grid(axis='x', alpha=0.3)
    if metrics[mi] == 'Distance':
        ax.set_xscale('log')
axes[0].set_yticks(y)
axes[0].set_yticklabels(configs)
handles = [
    Line2D([0], [0], color=colors_static, marker='o', label='Static Train'),
    Line2D([0], [0], color=colors_static, marker='s', label='Static Test'),
    Line2D([0], [0], color=colors_hybrid, marker='o', label='Hybrid Train'),
    Line2D([0], [0], color=colors_hybrid, marker='s', label='Hybrid Test'),
]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9)
fig.suptitle('Train to Test Shift (Dumbbell Chart)', fontsize=14)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig('fig2_dumbbell.png', dpi=300)
plt.show()

# ---------------- 3. SR-Cost Pareto scatter ----------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, data, title in zip(axes, [train, test], ['Train', 'Test']):
    for i, c in enumerate(configs):
        color = colors_static if i < 4 else colors_hybrid
        marker = 'o' if i < 4 else 's'
        ax.scatter(data[i, 2], data[i, 1], color=color, marker=marker, s=90, edgecolor='black', linewidth=0.5)
        ax.annotate(c, (data[i, 2], data[i, 1]), textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax.set_xlabel('Cost')
    ax.set_ylabel('Success Rate (SR)')
    ax.set_title(f'{title}: SR vs Cost')
    ax.grid(alpha=0.3)
fig.suptitle('SR-Cost Pareto Scatter', fontsize=14)
fig.tight_layout()
fig.savefig('fig3_sr_cost.png', dpi=300)
plt.show()


# ---------------- 4. Heatmap: column-wise min-max normalization ----------------
def normalize_col(col, higher_better):
    lo, hi = col.min(), col.max()
    if hi == lo: return np.ones_like(col)
    z = (col - lo) / (hi - lo)
    return z if higher_better else 1 - z


higher_better = {'Reward': True, 'SR': True, 'Cost': False, 'Length': False, 'Distance': False}
train_norm = np.column_stack([normalize_col(train[:, i], higher_better[m]) for i, m in enumerate(metrics)])
test_norm = np.column_stack([normalize_col(test[:, i], higher_better[m]) for i, m in enumerate(metrics)])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, mat, title in zip(axes, [train_norm, test_norm],
                          ['Train (normalized, brighter = better)',
                           'Test (normalized, brighter = better)']):
    im = ax.imshow(mat, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics)));
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.set_yticks(range(len(configs)));
    ax.set_yticklabels(configs)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=8)
    ax.set_title(title)
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label='Normalized score (1 = best)')
fig.suptitle('Configuration x Metric Heatmap', fontsize=14)
fig.savefig('fig4_heatmap.png', dpi=300)
plt.show()

# ---------------- 5. Parallel coordinates: 8 configs x 5 metrics ----------------
fig, ax = plt.subplots(figsize=(14, 7))
all_data = np.vstack([train, test])
norm_all = np.column_stack([normalize_col(all_data[:, i], higher_better[m]) for i, m in enumerate(metrics)])
n_train = len(train)
for i, c in enumerate(configs):
    color = colors_static if i < 4 else colors_hybrid
    ax.plot(range(len(metrics)), norm_all[i], color=color, marker='o', lw=1.5, alpha=0.8, label=f'{c} (Train)')
    ax.plot(range(len(metrics)), norm_all[n_train + i], color=color, marker='s', lw=1.0, linestyle='--', alpha=0.6)
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(metrics)
ax.set_ylabel('Normalized score (1 = best)')
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.3)
handles = [Line2D([0], [0], color=colors_static, lw=2, label='Static'),
           Line2D([0], [0], color=colors_hybrid, lw=2, label='Hybrid'),
           Line2D([0], [0], color='gray', marker='o', label='Train (solid)'),
           Line2D([0], [0], color='gray', marker='s', linestyle='--', label='Test (dashed)')]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=9)
fig.suptitle('Parallel Coordinates: Overall Profile of Configs x Metrics', fontsize=14)
fig.tight_layout()
fig.savefig('fig5_parallel_coords.png', dpi=300)
plt.show()
