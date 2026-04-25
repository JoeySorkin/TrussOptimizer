"""
Truss visualization — before/after comparison and standalone plot.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from solver import solve_forces, member_lengths, truss_metrics


COLORS = {
    "tension":     "#1a6faf",
    "compression": "#c0392b",
    "zero":        "#888888",
    "pin":         "#27ae60",
    "roller":      "#27ae60",
    "load":        "#e67e22",
    "free":        "#2980b9",
}


def draw_truss(ax, X, Y, C, Sx, Sy, L_vec, W, pin, roller, load, title):
    """Draw truss on axes. pin/roller/load are 0-indexed."""
    J = len(X)
    forces, _, ok = solve_forces(X, Y, C, Sx, Sy, L_vec)
    lens = member_lengths(X, Y, C)
    m = truss_metrics(X, Y, C, Sx, Sy, L_vec, W)

    for mi in range(C.shape[1]):
        j1, j2 = np.where(C[:, mi] == 1)[0][:2]
        xv = [float(X[j1]), float(X[j2])]
        yv = [float(Y[j1]), float(Y[j2])]
        if ok and forces is not None:
            f = forces[mi]
            color = COLORS["tension"] if f > 1e-6 else (COLORS["compression"] if f < -1e-6 else COLORS["zero"])
        else:
            color = COLORS["zero"]
        ax.plot(xv, yv, color=color, lw=2.2, zorder=2)
        mx, my = (xv[0]+xv[1])/2, (yv[0]+yv[1])/2
        label = f"m{mi+1}\n{lens[mi]:.1f}\""
        if ok and forces is not None:
            label += f"\n{forces[mi]:+.1f}"
        ax.text(mx, my + 0.15, label, fontsize=5.5, ha="center",
                color="#555", zorder=3, linespacing=1.2)

    for j in range(J):
        x, y = float(X[j]), float(Y[j])
        if j == pin or j == roller:
            c, mk = COLORS["pin"], ("^" if j == pin else "s")
        elif j == load:
            c, mk = COLORS["load"], "D"
        else:
            c, mk = COLORS["free"], "o"
        ax.scatter(x, y, c=c, marker=mk, s=110, zorder=5)
        ax.text(x + 0.25, y + 0.25, f"j{j+1}", fontsize=7, fontweight="bold", zorder=6)

    lx, ly = float(X[load]), float(Y[load])
    ax.annotate("", xy=(lx, ly - 1.4), xytext=(lx, ly),
                arrowprops=dict(arrowstyle="->", color=COLORS["load"], lw=2.0))

    wf = m["w_failures"]
    cv = float(np.std(wf) / np.mean(wf)) if len(wf) > 1 else 0.0
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel(
        f"material={np.sum(lens):.1f} in   W_max={m['W_max']:.1f} oz   "
        f"cost=${m['cost']:.0f}   score={m['score']:.4f} oz/$   CV={cv:.3f}",
        fontsize=7.5,
    )
    ax.set_ylabel("y (in)", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


def plot_comparison(X_i, Y_i, X_o, Y_o, C, Sx, Sy, L_vec, W,
                    pin, roller, load, info, save_path=None):
    """Side-by-side before/after plot. pin/roller/load are 0-indexed."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.suptitle("EK301 Truss Optimizer — Geometric Optimization Result",
                 fontsize=11, fontweight="bold")

    draw_truss(axes[0], X_i, Y_i, C, Sx, Sy, L_vec, W, pin, roller, load,
               f"Initial   score = {info['score_init']:.4f} oz/$")
    draw_truss(axes[1], X_o, Y_o, C, Sx, Sy, L_vec, W, pin, roller, load,
               f"Optimized   score = {info['score_opt']:.4f} oz/$  ({info['improvement']:+.1f}%)")

    legend = [
        mpatches.Patch(color=COLORS["pin"],         label="Pin / Roller"),
        mpatches.Patch(color=COLORS["load"],        label="Load joint"),
        mpatches.Patch(color=COLORS["tension"],     label="Tension member"),
        mpatches.Patch(color=COLORS["compression"], label="Compression member"),
    ]
    axes[1].legend(handles=legend, loc="upper right", fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig
