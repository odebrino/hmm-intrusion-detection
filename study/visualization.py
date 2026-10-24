"""
Generates the 5 graphs for the HMM-IDS project.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap


STATE_COLORS = ['#2ecc71', '#e67e22', '#e74c3c']   # green / orange / red
STATE_NAMES  = ['Normal', 'Suspicious', 'Attack']


# -----------------------------------------------------------------------
# 1.  Timeline : real state vs predicted state
# -----------------------------------------------------------------------

def plot_timeline(states_real, states_pred, n_show=300, save_path='timeline_states.png'):

    real = np.array(states_real[:n_show], dtype=int)
    pred = np.array(states_pred[:n_show], dtype=int)
    T    = len(real)

    cmap3 = ListedColormap(STATE_COLORS)          # 0=green 1=orange 2=red
    error = (real != pred).astype(float)           # 1 where wrong, 0 where correct

    fig, axes = plt.subplots(
        3, 1, figsize=(16, 5),
        gridspec_kw={'height_ratios': [3, 3, 1]},
        sharex=True
    )
    fig.suptitle(
        'State Sequence: Real vs Viterbi predictions\n'
        '(red strip = classification error)',
        fontsize=13, fontweight='bold'
    )

    # --- top two panels: state bands ---
    for ax, data, title in zip(axes[:2], [real, pred],
                               ['Real States', 'Predicted States (Viterbi)']):
        # Use imshow with each column = one timestep → perfectly readable bands
        ax.imshow(
            data[np.newaxis, :],          # shape (1, T) → 1 row, T columns
            aspect='auto',
            cmap=cmap3, vmin=0, vmax=2,
            extent=[-0.5, T - 0.5, -0.5, 0.5],
            interpolation='nearest'
        )
        ax.set_yticks([0])
        ax.set_yticklabels([title], fontsize=11)
        ax.tick_params(left=False)
        ax.set_ylim(-0.5, 0.5)

    # --- bottom panel: error strip ---
    err_cmap = ListedColormap(['#ffffff', '#c0392b'])   # white=correct, red=error
    axes[2].imshow(
        error[np.newaxis, :],
        aspect='auto',
        cmap=err_cmap, vmin=0, vmax=1,
        extent=[-0.5, T - 0.5, -0.5, 0.5],
        interpolation='nearest'
    )
    axes[2].set_yticks([0])
    axes[2].set_yticklabels(['Errors'], fontsize=10)
    axes[2].tick_params(left=False)
    axes[2].set_ylim(-0.5, 0.5)
    axes[2].set_xlabel('Timestep', fontsize=11)

    # --- shared legend ---
    patches = [mpatches.Patch(color=STATE_COLORS[i], label=STATE_NAMES[i])
               for i in range(3)]
    patches.append(mpatches.Patch(color='#c0392b', label='Error'))
    fig.legend(handles=patches, loc='upper right', fontsize=10,
               bbox_to_anchor=(1.0, 0.98))

    # --- stats annotation ---
    n_errors = int(error.sum())
    n_susp   = int(np.sum(pred == 1))
    axes[0].set_title(
        f'Errors on first {T} timesteps: {n_errors} '
        f'({100*n_errors/T:.1f}%)  —  '
        f'Suspicious predictions: {n_susp}',
        fontsize=10, loc='left', color='#555555'
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[viz] Timeline saved: {save_path}")


# -----------------------------------------------------------------------
# 2.  Confusion Matrix
# -----------------------------------------------------------------------

def plot_confusion_matrix(conf_matrix, state_names=None, save_path='confusion_matrix.png'):

    if state_names is None:
        state_names = STATE_NAMES

    N = conf_matrix.shape[0]

    # Row-normalised matrix (% of each real class)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for absent classes (e.g. Suspicious row = 0)
    norm_matrix = np.divide(conf_matrix, row_sums, out=np.zeros_like(conf_matrix, dtype=float), where=row_sums != 0) * 100

    # For the absolute colorscale: cap at the largest OFF-diagonal value
    # so small errors (7, 16, 43) are not washed out by the diagonal (850)
    mask_diag    = np.eye(N, dtype=bool)
    off_diag     = conf_matrix[~mask_diag]
    vmax_abs     = max(off_diag.max(), 1)          # at least 1 to avoid vmax=0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Confusion Matrix — HMM-IDS', fontsize=14, fontweight='bold')

    titles     = ['Absolute Counts\n(colorscale ← max off-diagonal value)',
                  'Rates per Real Class (%)']
    matrices   = [conf_matrix, norm_matrix]
    vmaxes     = [vmax_abs,    100.0]
    cmaps      = ['Blues',     'Blues']

    for ax, mat, title, vmax, cmap in zip(axes, matrices, titles, vmaxes, cmaps):
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(state_names, fontsize=11)
        ax.set_yticklabels(state_names, fontsize=11)
        ax.set_xlabel('Predicted State',  fontsize=12)
        ax.set_ylabel('Real State',    fontsize=12)
        ax.set_title(title, fontsize=11)

        # Annotate each cell with count + percentage
        for i in range(N):
            for j in range(N):
                count = conf_matrix[i, j]
                pct   = norm_matrix[i, j]
                # Choose text color: white on dark background, black on light
                bg    = mat[i, j]
                color = 'white' if bg > vmax * 0.55 else 'black'

                if row_sums[i, 0] == 0:
                    # Absent class row — show a dash
                    ax.text(j, i, '—', ha='center', va='center',
                            color='#888888', fontsize=13)
                else:
                    ax.text(j, i, f'{count}\n({pct:.1f}%)',
                            ha='center', va='center', color=color,
                            fontsize=11, fontweight='bold', linespacing=1.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[viz] Confusion Matrix saved: {save_path}")


# -----------------------------------------------------------------------
# 3.  Transition matrix A
# -----------------------------------------------------------------------

def plot_transition_matrix(A_initial, A_final, state_names=None,
                           save_path='transition_matrix.png'):
    if state_names is None:
        state_names = STATE_NAMES

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Transition Matrix A — Before and After Baum-Welch',
                 fontsize=13, fontweight='bold')

    labels = [
        'Sweet spot + Supervised init.\n(Em starting point)',
        'After Baum-Welch\n(Parameters learned by EM)'
    ]

    for ax, matrix, title in zip(axes, [A_initial, A_final], labels):
        im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(state_names)))
        ax.set_yticks(range(len(state_names)))
        ax.set_xticklabels(state_names, fontsize=10)
        ax.set_yticklabels(state_names, fontsize=10)
        ax.set_xlabel('Next State $q_{t+1}$', fontsize=10)
        ax.set_ylabel('Current State $q_t$', fontsize=10)
        ax.set_title(title, fontsize=11)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                color = 'white' if matrix[i, j] > 0.55 else 'black'
                ax.text(j, i, f'{matrix[i, j]:.3f}',
                        ha='center', va='center', fontsize=10, color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[viz] Transition Matrices saved: {save_path}")


# -----------------------------------------------------------------------
# 4.  Emission matrix B
# -----------------------------------------------------------------------

def plot_emission_matrix(B, state_names=None, save_path='emission_matrix.png'):

    if state_names is None:
        state_names = STATE_NAMES

    N, M = B.shape
    symbols = np.arange(M)

    fig, axes = plt.subplots(N, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        'Emission Matrix B — After Baum-Welch\n'
        'Each subplot has its own Y scale to highlight\n'
        'the discriminant clusters (peaks) and unsupervised states (uniform)',
        fontsize=12, fontweight='bold'
    )

    bar_colors = STATE_COLORS

    for idx, (ax, name, color) in enumerate(zip(axes, state_names, bar_colors)):
        row = B[idx]
        ax.bar(symbols, row, color=color, alpha=0.85, width=0.9)
        ax.set_ylabel(f'{name}\nP(sym|state)', fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)

        # Annotate the top-5 peaks for Normal and Attack
        if name != 'Suspicious':
            top5_idx = np.argsort(row)[-5:][::-1]
            for k in top5_idx:
                ax.annotate(
                    f'#{k}',
                    xy=(k, row[k]),
                    xytext=(k, row[k] + row.max() * 0.07),
                    ha='center', fontsize=7, color='#333333'
                )

        # Dashed line at uniform level 1/M for reference
        uniform = 1.0 / M
        ax.axhline(uniform, color='black', linestyle='--', linewidth=0.8,
                   label=f'Uniform (1/{M} = {uniform:.4f})')
        ax.legend(fontsize=8, loc='upper right')

    axes[-1].set_xlabel(f'K-Means Symbol (0 … {M-1})', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[viz] Emission Matrix saved: {save_path}")


# -----------------------------------------------------------------------
# 5.  Courbe de convergence Baum-Welch  (FIXED: iter-0 pre-BW point)
# -----------------------------------------------------------------------

def plot_bw_convergence(log_likelihoods, ll_before_bw=None,
                        save_path='bw_convergence.png'):

    lls = list(log_likelihoods)

    if ll_before_bw is not None:
        all_lls = [ll_before_bw] + lls
        iters   = np.arange(0, len(all_lls))
        iter0_label = 'Iter 0 (Before BW)'
    else:
        all_lls = lls
        iters   = np.arange(1, len(all_lls) + 1)
        iter0_label = None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, all_lls, 'b-o', markersize=5, linewidth=2, label='Log-likelihood')

    # Mark iter-0 distinctly if provided
    if ll_before_bw is not None:
        ax.plot(0, ll_before_bw, 'gs', markersize=9, zorder=5,
                label=iter0_label)
        ax.annotate(
            f'Before BW\n{ll_before_bw:.1f}',
            xy=(0, ll_before_bw),
            xytext=(0.8, ll_before_bw + abs(ll_before_bw) * 0.015),
            fontsize=9, color='#27ae60',
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.2)
        )

    # Mark final value
    ax.plot(iters[-1], all_lls[-1], 'r^', markersize=9, zorder=5,
            label=f'Final (iter {iters[-1]})')
    ax.annotate(
        f'Final\n{all_lls[-1]:.1f}',
        xy=(iters[-1], all_lls[-1]),
        xytext=(iters[-1] - 2.5, all_lls[-1] - abs(all_lls[-1]) * 0.015),
        fontsize=9, color='#c0392b',
        arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2)
    )

    # Annotate total gain
    gain = all_lls[-1] - all_lls[0]
    ax.set_title(
        f'Baum-Welch Convergence  —  '
        f'Total Gain: {gain:+.1f} nats  ({100*gain/abs(all_lls[0]):.1f} %)',
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('log P(O | λ)', fontsize=12)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=10)

    # Add a zoomed inset of iterations 3-20 to show the slow tail
    n_zoom_start = 3
    if len(iters) > n_zoom_start + 2:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_ins = inset_axes(ax, width='38%', height='40%', loc='lower right',
                            bbox_to_anchor=(-0.01, 0.06, 1, 1),
                            bbox_transform=ax.transAxes)
        zoom_iters = iters[n_zoom_start:]
        zoom_lls   = all_lls[n_zoom_start:]
        ax_ins.plot(zoom_iters, zoom_lls, 'b-o', markersize=3, linewidth=1.2)
        ax_ins.set_title(f'Zoom iter {zoom_iters[0]}–{zoom_iters[-1]}',
                         fontsize=8)
        ax_ins.grid(True, alpha=0.3)
        ax_ins.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[viz] Convergence Curve saved: {save_path}")
