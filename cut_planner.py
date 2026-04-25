"""
Material cut planner — assigns members to stock pieces with minimal waste.
Uses backtracking search (exact, no approximation).
"""

import numpy as np
from solver import member_lengths


def plan_cuts(X, Y, C, stock_lengths):
    """
    Assign each truss member to a stock piece.

    stock_lengths: list of available stock piece lengths (inches).
    Returns cut plan dict or None if no feasible assignment exists.
    """
    lens = member_lengths(X, Y, C)
    members = sorted(enumerate(lens.tolist()), key=lambda x: x[1], reverse=True)
    stock   = [float(s) for s in stock_lengths]
    remaining  = stock.copy()
    assignment = [[] for _ in stock]

    if sum(l for _, l in members) > sum(stock) + 1e-9:
        return None

    def backtrack(pos):
        if pos == len(members):
            return True
        idx, length = members[pos]
        seen = set()
        for i in range(len(stock)):
            if remaining[i] + 1e-9 < length:
                continue
            r = round(remaining[i], 9)
            if r in seen:
                continue
            seen.add(r)
            remaining[i] -= length
            assignment[i].append((idx, length))
            if backtrack(pos + 1):
                return True
            assignment[i].pop()
            remaining[i] += length
        return False

    if not backtrack(0):
        return None

    return {"stock": stock, "cuts": assignment, "leftover": remaining}


def print_cut_plan(X, Y, C, stock_lengths):
    """Compute and print the cut plan for a truss."""
    lens = member_lengths(X, Y, C)
    total_members = float(np.sum(lens))
    total_stock   = float(np.sum(stock_lengths))

    print("\n┌─ Material Cut Plan " + "─" * 45 + "┐")
    print("│  Member lengths:")
    for i, L in enumerate(lens):
        print(f"│    m{i+1}: {L:.3f} in")
    print(f"│  Total member length : {total_members:.3f} in")
    print(f"│  Total stock length  : {total_stock:.3f} in")
    print(f"│  Waste budget        : {total_stock - total_members:.3f} in")

    plan = plan_cuts(X, Y, C, stock_lengths)

    if plan is None:
        print("│  ✗  No feasible cut plan for the given stock.")
        print("└" + "─" * 66 + "┘\n")
        return None

    print("│")
    print("│  Feasible cut plan:")
    for i, (slen, cuts, left) in enumerate(zip(plan["stock"], plan["cuts"], plan["leftover"]), 1):
        used = slen - left
        members_txt = ", ".join(f"m{midx+1} ({ml:.3f}\")" for midx, ml in cuts) or "none"
        print(f"│    Stock {i} ({slen:.1f} in):  {members_txt}")
        print(f"│      Used: {used:.3f} in  |  Leftover: {left:.3f} in")
    print("└" + "─" * 66 + "┘\n")
    return plan
