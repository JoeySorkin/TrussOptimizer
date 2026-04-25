"""
EK301 Truss Optimizer — main workflow.

1. Edit truss.yaml with your joint positions, members, and settings.
2. Run:  python main.py
3. Outputs: optimized joint positions, validation report, cut plan, and plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml

from solver           import make_support_matrices, make_load_vector
from design_validator import validate
from optimizer        import optimize
from cut_planner      import print_cut_plan
from visualizer       import plot_comparison

SEARCH_MODES = {
    "QUICK":  dict(seeds=[42],                    maxiter=1000, popsize=15),
    "NORMAL": dict(seeds=[42, 7, 13],             maxiter=2500, popsize=20),
    "DEEP":   dict(seeds=[42, 7, 13, 99, 1337, 0], maxiter=5000, popsize=25),
}


def load_truss(path="truss.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    joint_keys = sorted(cfg["joints"].keys(), key=lambda k: int(k[1:]))
    coords = [cfg["joints"][k] for k in joint_keys]
    X = np.array([c[0] for c in coords], dtype=float)
    Y = np.array([c[1] for c in coords], dtype=float)

    J = len(X)
    members = cfg["members"]
    M = len(members)
    C = np.zeros((J, M), dtype=float)
    for m, (ja, jb) in enumerate(members):
        C[ja - 1, m] = 1
        C[jb - 1, m] = 1

    mode = cfg.get("search_mode", "NORMAL").upper()
    if mode not in SEARCH_MODES:
        raise ValueError(f"search_mode must be QUICK, NORMAL, or DEEP — got '{mode}'")

    return {
        "X":            X,
        "Y":            Y,
        "C":            C,
        "pin_joint":    int(cfg["pin_joint"]),
        "roller_joint": int(cfg["roller_joint"]),
        "load_joint":   int(cfg["load_joint"]),
        "W":            float(cfg["load_oz"]),
        "stock":        list(cfg["stock"]),
        "parallel":     bool(cfg.get("parallel", False)),
        **SEARCH_MODES[mode],
    }


def main():
    import sys
    yaml_file = sys.argv[1] if len(sys.argv) > 1 else "truss.yaml"
    print(f"  Loading: {yaml_file}")
    cfg = load_truss(yaml_file)

    X, Y, C  = cfg["X"], cfg["Y"], cfg["C"]
    PIN      = cfg["pin_joint"]
    ROLLER   = cfg["roller_joint"]
    LOAD     = cfg["load_joint"]
    W        = cfg["W"]
    STOCK    = cfg["stock"]
    SEEDS    = cfg["seeds"]
    MAXITER  = cfg["maxiter"]
    POPSIZE  = cfg["popsize"]
    PARALLEL = cfg["parallel"]

    J    = len(X)
    pin  = PIN    - 1
    roll = ROLLER - 1
    load = LOAD   - 1

    Sx, Sy = make_support_matrices(J, pin, roll)
    L_vec  = make_load_vector(J, load, W)

    # ── Step 1: validate initial design ──────────────────────────────────────
    print("\n" + "=" * 66)
    print("  Step 1: Validating initial design")
    print("=" * 66)
    valid, _ = validate(X, Y, C, Sx, Sy, L_vec, PIN, ROLLER, LOAD)
    if not valid:
        print("  Initial design has constraint violations — optimizer will still run.")

    # ── Step 2: optimize ──────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  Step 2: Optimizing joint positions")
    print("=" * 66)
    X_opt, Y_opt, info = optimize(
        X, Y, C, Sx, Sy, L_vec,
        PIN, ROLLER, LOAD, W,
        maxiter=MAXITER, popsize=POPSIZE, seeds=SEEDS, parallel=PARALLEL,
    )

    # ── Step 3: validate optimized design ────────────────────────────────────
    print("\n" + "=" * 66)
    print("  Step 3: Validating optimized design")
    print("=" * 66)
    validate(X_opt, Y_opt, C, Sx, Sy, L_vec, PIN, ROLLER, LOAD)

    # ── Step 4: cut plan ──────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  Step 4: Material cut plan")
    print("=" * 66)
    print_cut_plan(X_opt, Y_opt, C, STOCK)

    # ── Step 5: plot ──────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  Step 5: Generating plots")
    print("=" * 66)
    plot_comparison(
        X, Y, X_opt, Y_opt, C, Sx, Sy, L_vec, W,
        pin, roll, load, info,
        save_path="optimized_truss.png",
    )
    plt.show()


if __name__ == "__main__":
    main()
