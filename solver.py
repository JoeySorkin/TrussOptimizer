"""
Core truss structural analysis: build A matrix, solve forces, compute metrics.
"""

import numpy as np

# ── Spec constants ────────────────────────────────────────────────────────────
L_MIN    = 6.0
L_MAX    = 14.0
MAT_MAX  = 120.0
LOAD_X_MIN = 9.0
LOAD_X_MAX = 11.0
LOAD_Y_MAX = 2.0

# ── Buckling model ────────────────────────────────────────────────────────────
C_FIT = 37.5
LO    = 10.0
ALPHA = 2.0
U_FIT = 10.0

# ── Cost model ────────────────────────────────────────────────────────────────
C_L = 1.0   # $ per inch of member
C_J = 10.0  # $ per joint


def build_A(X, Y, C, Sx, Sy):
    J = len(X)
    M = C.shape[1]
    A = np.zeros((2 * J, M + 3))
    for m in range(M):
        j1, j2 = np.where(C[:, m] == 1)[0][:2]
        L = float(np.hypot(X[j2] - X[j1], Y[j2] - Y[j1]))
        if L < 1e-10:
            continue
        cx, cy = (X[j2] - X[j1]) / L, (Y[j2] - Y[j1]) / L
        A[j1, m] = cx;  A[j2, m] = -cx
        A[J + j1, m] = cy;  A[J + j2, m] = -cy
    A[0:J,   M:M+3] = Sx
    A[J:2*J, M:M+3] = Sy
    return A


def solve_forces(X, Y, C, Sx, Sy, L_vec):
    """Return (member_forces, reactions, ok). Positive = tension."""
    A = build_A(X, Y, C, Sx, Sy)
    try:
        sol = np.linalg.solve(A, L_vec)
        T = -sol
        return T[:C.shape[1]], T[C.shape[1]:], True
    except np.linalg.LinAlgError:
        return None, None, False


def member_lengths(X, Y, C):
    M = C.shape[1]
    lengths = np.zeros(M)
    for m in range(M):
        j1, j2 = np.where(C[:, m] == 1)[0][:2]
        lengths[m] = float(np.hypot(X[j2] - X[j1], Y[j2] - Y[j1]))
    return lengths


def buckle_capacity(L):
    """Lower-bound buckling capacity (oz) for a compression member of length L."""
    return C_FIT * (LO / max(L, 0.1)) ** ALPHA - U_FIT


def truss_metrics(X, Y, C, Sx, Sy, L_vec, W):
    """
    Returns dict with: W_max, cost, score (W_max/cost), w_failures (per compression member).
    """
    J = len(X)
    forces, _, ok = solve_forces(X, Y, C, Sx, Sy, L_vec)
    lens = member_lengths(X, Y, C)
    cost = C_L * float(np.sum(lens)) + C_J * J

    if not ok or forces is None:
        return {"W_max": 0.0, "cost": cost, "score": 0.0, "w_failures": []}

    Rm = forces / W
    w_failures = []
    for m in range(len(forces)):
        if Rm[m] < 0:
            fb = buckle_capacity(lens[m])
            if fb <= 0:
                return {"W_max": 0.0, "cost": cost, "score": 0.0, "w_failures": []}
            w_failures.append(fb / abs(Rm[m]))

    W_max = float(min(w_failures)) if w_failures else float(W)
    score = W_max / cost if cost > 0 else 0.0
    return {"W_max": W_max, "cost": cost, "score": score, "w_failures": w_failures}


def make_support_matrices(J, pin_idx, roller_idx):
    """Build Sx, Sy support matrices (0-indexed joint indices)."""
    Sx = np.zeros((J, 3))
    Sy = np.zeros((J, 3))
    Sx[pin_idx, 0] = 1
    Sy[pin_idx, 1] = 1
    Sy[roller_idx, 2] = 1
    return Sx, Sy


def make_load_vector(J, load_idx, W):
    """Build load vector with downward load W at load_idx (0-indexed)."""
    L_vec = np.zeros(2 * J)
    L_vec[J + load_idx] = -W
    return L_vec
