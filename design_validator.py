"""
EK301 truss design validator — checks all spec constraints.
"""

import numpy as np
from solver import build_A, member_lengths, L_MIN, L_MAX, MAT_MAX, LOAD_X_MIN, LOAD_X_MAX, LOAD_Y_MAX


def validate(X, Y, C, Sx, Sy, L_vec, pin_joint, roller_joint, load_joint, verbose=True):
    """
    Validate truss against EK301 Spring 2026 spec.

    Parameters use 1-indexed joint numbers. Returns (valid: bool, issues: list[str]).
    """
    p  = pin_joint    - 1
    r  = roller_joint - 1
    ld = load_joint   - 1

    J = len(X)
    M = C.shape[1]

    passed = []
    failed = []

    def check(condition, pass_msg, fail_msg):
        if condition:
            passed.append(pass_msg)
        else:
            failed.append(fail_msg)
        return condition

    # Dimensions
    check(len(Y) == J,           f"X/Y length match ({J})", f"X/Y length mismatch")
    check(C.shape[0] == J,       f"C rows = J ({J})",       f"C rows ({C.shape[0]}) != J ({J})")
    check(len(L_vec) == 2 * J,   f"L_vec length = 2J ({2*J})", f"L_vec length wrong")

    # Determinacy
    check(M == 2 * J - 3, f"M = 2J-3 = {M}", f"M = {M} but 2J-3 = {2*J-3}")

    # C matrix validity
    col_sums = C.sum(axis=0)
    bad = np.where(col_sums != 2)[0] + 1
    check(len(bad) == 0, "All C columns sum to 2", f"C columns not summing to 2: {bad.tolist()}")

    pairs = [tuple(sorted(np.where(C[:, m] == 1)[0])) for m in range(M)]
    check(len(pairs) == len(set(pairs)), "No duplicate members", "Duplicate members detected")

    # Member lengths
    lens = member_lengths(X, Y, C)
    TOL = 0.005
    short = np.where(lens < L_MIN - TOL)[0] + 1
    long_ = np.where(lens > L_MAX + TOL)[0] + 1
    check(len(short) == 0, f"All members >= {L_MIN} in", f"Members too short: {short.tolist()}")
    check(len(long_) == 0, f"All members <= {L_MAX} in", f"Members too long:  {long_.tolist()}")

    total = float(np.sum(lens))
    check(total <= MAT_MAX, f"Total material = {total:.2f} in (≤ {MAT_MAX})",
          f"Total material = {total:.2f} in exceeds {MAT_MAX} in")

    # Joint positions
    below = np.where(Y < -1e-9)[0] + 1
    check(len(below) == 0, "All joints y >= 0", f"Joints below baseline: {below.tolist()}")
    check(abs(Y[p]) < 1e-9, f"Pin j{pin_joint} on baseline",    f"Pin j{pin_joint} not on baseline (y={Y[p]:.4f})")
    check(abs(Y[r]) < 1e-9, f"Roller j{roller_joint} on baseline", f"Roller j{roller_joint} not on baseline (y={Y[r]:.4f})")

    # Span
    span = abs(X[r] - X[p])
    check(26 <= span <= 30, f"Span = {span:.2f} in (26–30)", f"Span = {span:.2f} in out of 26–30 range")

    # Load joint position
    lx = float(X[ld] - X[p])
    ly = float(Y[ld] - Y[p])
    check(LOAD_X_MIN <= lx <= LOAD_X_MAX, f"Load x-offset = {lx:.2f} in ({LOAD_X_MIN}–{LOAD_X_MAX})",
          f"Load x-offset = {lx:.2f} in out of {LOAD_X_MIN}–{LOAD_X_MAX} range")
    check(0 <= ly <= LOAD_Y_MAX, f"Load y-offset = {ly:.2f} in (0–{LOAD_Y_MAX})",
          f"Load y-offset = {ly:.2f} in out of 0–{LOAD_Y_MAX} range")

    below_load = [j+1 for j in range(J) if j != ld and abs(X[j] - X[ld]) < 1e-9 and Y[j] < Y[ld] - 1e-9]
    check(len(below_load) == 0, "No joints directly below load joint",
          f"Joints below load joint: {below_load}")

    # No crossing members
    crosses = _find_crossings(X, Y, C)
    check(len(crosses) == 0, "No crossing members", f"Crossing members: {crosses}")

    # A matrix solvable
    A = build_A(X, Y, C, Sx, Sy)
    det = np.linalg.det(A)
    check(abs(det) > 1e-9, f"A matrix invertible (det={det:.2e})",
          f"A matrix singular (det={det:.2e}) — truss is not determinate")

    valid = len(failed) == 0

    if verbose:
        _print_report(passed, failed, valid)

    return valid, failed


def _find_crossings(X, Y, C):
    M = C.shape[1]
    crosses = []
    for m1 in range(M - 1):
        for m2 in range(m1 + 1, M):
            j1s = np.where(C[:, m1] == 1)[0]
            j2s = np.where(C[:, m2] == 1)[0]
            if any(j in j2s for j in j1s):
                continue
            p1 = np.array([X[j1s[0]], Y[j1s[0]]])
            p2 = np.array([X[j1s[1]], Y[j1s[1]]])
            p3 = np.array([X[j2s[0]], Y[j2s[0]]])
            p4 = np.array([X[j2s[1]], Y[j2s[1]]])
            if _segs_cross(p1, p2, p3, p4):
                crosses.append((m1 + 1, m2 + 1))
    return crosses


def _segs_cross(p1, p2, p3, p4):
    d1, d2 = p2 - p1, p4 - p3
    den = d1[0]*d2[1] - d1[1]*d2[0]
    if abs(den) < 1e-10:
        return False
    t = ((p3[0]-p1[0])*d2[1] - (p3[1]-p1[1])*d2[0]) / den
    u = ((p3[0]-p1[0])*d1[1] - (p3[1]-p1[1])*d1[0]) / den
    return (1e-6 < t < 1-1e-6) and (1e-6 < u < 1-1e-6)


def _print_report(passed, failed, valid):
    print("\n┌─ Truss Validation " + "─" * 46 + "┐")
    for msg in passed:
        print(f"│  ✓  {msg}")
    for msg in failed:
        print(f"│  ✗  {msg}")
    status = "VALID" if valid else "INVALID"
    bar = "─" * 65
    print(f"│  {bar}")
    print(f"│  Result: {status}")
    print("└" + "─" * 66 + "┘\n")
