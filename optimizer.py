"""
Geometric optimizer — Differential Evolution over free joint positions.
Maximizes W_max / cost while satisfying all spec constraints.

Parallelism:
  - Seeds run concurrently via ProcessPoolExecutor (each seed is independent).
  - Within each DE run, population evaluations are parallelized via workers=-1.
  - Both use all available CPU cores by default.
"""

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import differential_evolution

from solver import (
    truss_metrics, member_lengths,
    L_MIN, L_MAX, MAT_MAX, LOAD_X_MIN, LOAD_X_MAX, LOAD_Y_MAX,
)
from design_validator import _find_crossings

LAMBDA_SF = 0.5
BIG = 1e6


# =============================================================================
#  Picklable objective (required for multiprocessing workers)
# =============================================================================

class _Objective:
    """Callable class so the objective is picklable for parallel DE workers."""

    def __init__(self, X_base, Y_base, C, Sx, Sy, L_vec, W, free, pin, roll, load, J):
        self.X_base = X_base
        self.Y_base = Y_base
        self.C      = C
        self.Sx     = Sx
        self.Sy     = Sy
        self.L_vec  = L_vec
        self.W      = W
        self.free   = free
        self.pin    = pin
        self.roll   = roll
        self.load   = load
        self.J      = J

    def __call__(self, params):
        Xp, Yp = _apply_params(params, self.X_base, self.Y_base, self.free)
        penalty = _constraints(Xp, Yp, self.C, self.free, self.pin, self.roll, self.load, self.J)
        m  = truss_metrics(Xp, Yp, self.C, self.Sx, self.Sy, self.L_vec, self.W)
        sf = _sf_penalty(m["w_failures"])
        if penalty > 0:
            return penalty - m["score"] + sf
        return -m["score"] + sf


# =============================================================================
#  Per-seed worker (runs in a separate process)
# =============================================================================

def _run_seed(args):
    """Silent worker — used in parallel mode."""
    seed, X, Y, C, Sx, Sy, L_vec, W, free, pin, roll, load, J, bounds, maxiter, popsize = args
    obj = _Objective(X, Y, C, Sx, Sy, L_vec, W, free, pin, roll, load, J)
    result = differential_evolution(
        obj, bounds,
        maxiter=maxiter, popsize=popsize,
        mutation=(0.4, 1.9), recombination=0.95,
        tol=1e-10, seed=seed, polish=True, disp=False,
        workers=1,
    )
    Xr, Yr = _apply_params(result.x, X, Y, free)
    mr = truss_metrics(Xr, Yr, C, Sx, Sy, L_vec, W)
    return seed, result.x, mr


def _run_seed_verbose(args):
    """Sequential worker with live per-eval progress output."""
    seed, X, Y, C, Sx, Sy, L_vec, W, free, pin, roll, load, J, bounds, maxiter, popsize = args
    counter = [0]

    def objective(params):
        counter[0] += 1
        Xp, Yp = _apply_params(params, X, Y, free)
        penalty = _constraints(Xp, Yp, C, free, pin, roll, load, J)
        m  = truss_metrics(Xp, Yp, C, Sx, Sy, L_vec, W)
        sf = _sf_penalty(m["w_failures"])
        if counter[0] % 5000 == 0:
            print(f"    eval {counter[0]:6d}  score={m['score']:.4f}  "
                  f"W_max={m['W_max']:.1f}  cost=${m['cost']:.0f}  penalty={penalty:.0f}")
        return (penalty - m["score"] + sf) if penalty > 0 else (-m["score"] + sf)

    result = differential_evolution(
        objective, bounds,
        maxiter=maxiter, popsize=popsize,
        mutation=(0.4, 1.9), recombination=0.95,
        tol=1e-10, seed=seed, polish=True, disp=False,
    )
    Xr, Yr = _apply_params(result.x, X, Y, free)
    mr = truss_metrics(Xr, Yr, C, Sx, Sy, L_vec, W)
    return seed, result.x, mr


# =============================================================================
#  Main optimizer
# =============================================================================

def optimize(X, Y, C, Sx, Sy, L_vec, pin_joint, roller_joint, load_joint, W,
             maxiter=5000, popsize=25, seeds=(42,), parallel=False):
    """
    Optimize free joint positions to maximize load/cost ratio.

    parallel=True  — seeds run simultaneously via ProcessPoolExecutor (faster, silent)
    parallel=False — seeds run sequentially with live per-eval progress output

    Returns (X_opt, Y_opt, info_dict).
    """
    J    = len(X)
    pin  = pin_joint    - 1
    roll = roller_joint - 1
    load = load_joint   - 1
    free = [j for j in range(J) if j not in (pin, roll)]

    init      = truss_metrics(X, Y, C, Sx, Sy, L_vec, W)
    bounds    = _make_bounds(X, free, pin, load)
    n_workers = min(len(seeds), os.cpu_count() or 1) if parallel else 1

    _print_header(init, X, Y, C, J, pin, roll, load, maxiter, popsize, len(seeds), n_workers, parallel)

    seed_args = [
        (seed, X, Y, C, Sx, Sy, L_vec, W, free, pin, roll, load, J, bounds, maxiter, popsize)
        for seed in seeds
    ]

    results = []
    print()
    if parallel:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_seed, args): args[0] for args in seed_args}
            for future in as_completed(futures):
                seed, params, mr = future.result()
                results.append((mr["score"], params, mr))
                print(f"  ✓ seed={seed:<6}  score={mr['score']:.4f}  "
                      f"W_max={mr['W_max']:.2f} oz  cost=${mr['cost']:.0f}")
    else:
        for run, args in enumerate(seed_args, 1):
            seed = args[0]
            print(f"  ── Run {run}/{len(seeds)}  (seed={seed}) ──")
            _, params, mr = _run_seed_verbose(args)
            results.append((mr["score"], params, mr))
            mark = "  ← best so far" if mr["score"] >= max(r[0] for r in results) else ""
            print(f"  score={mr['score']:.4f}  W_max={mr['W_max']:.2f} oz  "
                  f"cost=${mr['cost']:.0f}{mark}")

    results.sort(key=lambda r: r[0], reverse=True)
    _, best_params, _ = results[0]

    X_opt, Y_opt = _apply_params(best_params, X, Y, free)
    m_opt = truss_metrics(X_opt, Y_opt, C, Sx, Sy, L_vec, W)
    improvement = (m_opt["score"] - init["score"]) / max(init["score"], 1e-12) * 100

    _print_results(X, Y, X_opt, Y_opt, C, m_opt, init, improvement, J, pin, roll, load)

    return X_opt, Y_opt, {
        "W_max_init":  init["W_max"],
        "cost_init":   init["cost"],
        "score_init":  init["score"],
        "W_max_opt":   m_opt["W_max"],
        "cost_opt":    m_opt["cost"],
        "score_opt":   m_opt["score"],
        "improvement": improvement,
        "w_failures":  m_opt["w_failures"],
    }


# =============================================================================
#  Helpers
# =============================================================================

def _apply_params(params, X_base, Y_base, free):
    X = X_base.copy()
    Y = Y_base.copy()
    for k, j in enumerate(free):
        X[j] = params[2*k]
        Y[j] = params[2*k + 1]
    return X, Y


def _make_bounds(X, free, pin, load):
    px = float(X[pin])
    bounds = []
    for j in free:
        if j == load:
            bounds += [(px + LOAD_X_MIN, px + LOAD_X_MAX), (0.0, LOAD_Y_MAX)]
        else:
            bounds += [(px - L_MAX, px + L_MAX * 4), (0.0, L_MAX * 4)]
    return bounds


def _constraints(X, Y, C, free, pin, roll, load, J):
    penalty = 0.0
    lens = member_lengths(X, Y, C)
    fixed = {pin, roll}

    for m, L in enumerate(lens):
        js = set(np.where(C[:, m] == 1)[0].tolist())
        if js.issubset(fixed):
            continue
        if L < L_MIN:
            penalty += BIG * (L_MIN - L) ** 2
        if L > L_MAX:
            penalty += BIG * (L - L_MAX) ** 2

    total = float(np.sum(lens))
    if total > MAT_MAX:
        penalty += BIG * (total - MAT_MAX) ** 2

    for j in free:
        if Y[j] < 0:
            penalty += BIG * Y[j] ** 2

    lx = float(X[load] - X[pin])
    ly = float(Y[load] - Y[pin])
    if lx < LOAD_X_MIN: penalty += BIG * (LOAD_X_MIN - lx) ** 2
    if lx > LOAD_X_MAX: penalty += BIG * (lx - LOAD_X_MAX) ** 2
    if ly < 0:          penalty += BIG * ly ** 2
    if ly > LOAD_Y_MAX: penalty += BIG * (ly - LOAD_Y_MAX) ** 2

    for j in range(J):
        if j != load and abs(float(X[j]) - float(X[load])) < 1e-4 and float(Y[j]) < float(Y[load]):
            penalty += BIG

    penalty += BIG * len(_find_crossings(X, Y, C))
    return penalty


def _sf_penalty(w_failures):
    if len(w_failures) < 2:
        return 0.0
    wf   = np.array(w_failures)
    mean = float(np.mean(wf))
    if mean < 1e-6:
        return 0.0
    return LAMBDA_SF * float(np.std(wf)) / mean


def _print_header(init, X, Y, C, J, pin, roll, load, maxiter, popsize, n_seeds, n_workers, parallel):
    lens      = member_lengths(X, Y, C)
    total     = float(np.sum(lens))
    wmax_str  = f"{init['W_max']:.2f}"
    cost_str  = f"{init['cost']:.0f}"
    total_str = f"{total:.2f}"
    s1 = f"  Settings: {n_seeds} seed(s), maxiter={maxiter}, popsize={popsize}"
    s2 = (f"  Parallel: {n_workers} workers (of {os.cpu_count()} cores)"
          if parallel else "  Parallel: off (sequential with live output)")
    print("╔" + "═" * 64 + "╗")
    print("║  EK301 Truss Geometric Optimizer" + " " * 31 + "║")
    print("╠" + "═" * 64 + "╣")
    print( "║  Initial geometry:                                              ║")
    print(f"║    Joints           : {J:<41}║")
    print(f"║    Total material   : {total_str} / {MAT_MAX} in{' ' * (30 - len(total_str))}║")
    print(f"║    W_max            : {wmax_str} oz{' ' * (41 - len(wmax_str))}║")
    print(f"║    Cost             : ${cost_str}{' ' * (41 - len(cost_str))}║")
    print(f"║    Score (W/cost)   : {init['score']:.4f} oz/$" + " " * 38 + "║")
    print(f"║{s1:<64}║")
    print(f"║{s2:<64}║")
    print("╚" + "═" * 64 + "╝")


def _print_results(X_i, Y_i, X_o, Y_o, C, m, init, improvement, J, pin, roll, load):
    lens  = member_lengths(X_o, Y_o, C)
    total = float(np.sum(lens))
    wf    = m["w_failures"]
    cv    = float(np.std(wf) / np.mean(wf)) if len(wf) > 1 else 0.0

    print("\n╔" + "═" * 64 + "╗")
    print("║  OPTIMIZED RESULT" + " " * 47 + "║")
    print("╠" + "═" * 64 + "╣")
    print(f"║  Total material  : {total:.2f} in" + " " * (44 - len(f"{total:.2f}")) + "║")
    print(f"║  W_max           : {m['W_max']:.2f} oz" + " " * (44 - len(f"{m['W_max']:.2f}")) + "║")
    print(f"║  Cost            : ${m['cost']:.0f}" + " " * (44 - len(f"{m['cost']:.0f}")) + "║")
    print(f"║  Score (W/cost)  : {m['score']:.4f} oz/$  ({improvement:+.1f}% improvement)" + " " * max(0, 24 - len(f"{improvement:+.1f}")) + "║")
    print(f"║  Failure CV      : {cv:.4f}  (0 = perfect simultaneous failure)" + " " * 6 + "║")
    print("╠" + "═" * 64 + "╣")
    print("║  Joint positions (optimized):                                  ║")
    for j in range(J):
        tag = "[pin]" if j == pin else ("[roller]" if j == roll else ("[load]" if j == load else ""))
        dx, dy = float(X_o[j] - X_i[j]), float(Y_o[j] - Y_i[j])
        line = f"  j{j+1} {tag:8s}  ({X_o[j]:.3f}, {Y_o[j]:.3f})  Δ({dx:+.3f}, {dy:+.3f})"
        print(f"║{line:<64}║")
    print("╠" + "═" * 64 + "╣")
    print("║  MATLAB arrays:                                                ║")
    x_str = f"  X = [{' '.join(f'{v:.4f}' for v in X_o)}]"
    y_str = f"  Y = [{' '.join(f'{v:.4f}' for v in Y_o)}]"
    print(f"║{x_str:<64}║")
    print(f"║{y_str:<64}║")
    if wf:
        print("╠" + "═" * 64 + "╣")
        print("║  Compression member failure loads:                             ║")
        for i, w in enumerate(wf):
            line = f"    Member {i+1}: {w:.2f} oz"
            print(f"║{line:<64}║")
    print("╚" + "═" * 64 + "╝")
