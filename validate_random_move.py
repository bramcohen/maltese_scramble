"""Is an N-move random scramble distinguishable from a uniform random state?

xyzzy's method (speedsolving.com, Megaminx scramble proposal): simulate many
scrambles, compare per-piece location distributions and a few aggregate counts
against a reference sample of uniform states.

The reference is drawn by the exact uniform sampler (model.random_state, valid by
the group facts in groupfacts.py) and cross-checked against a 5,000-move random
walk.  All states are rotation-normalised (DBL corner home), since the puzzle has
no fixed orientation.

Usage: .venv/bin/python validate_random_move.py [--n 100000] [--lengths 100,200,300,500]
"""
import argparse
import time

import numpy as np
from scipy import stats

import model as M
from batch import Batch, tables

EDGE_SLOTS = [i for i in range(M.NR) if not M.IS_CENTER[i]]
SIB = [(M.GEAR_NAMES.index(e[0] + e[1]), M.GEAR_NAMES.index(e[1] + e[0])) for e in
       (M.RING_NAMES[i] for i in EDGE_SLOTS)]


def statistics(b):
    a = b.a
    n = b.n
    out = {}
    # per-piece slot distributions (corner pieces 0..6, ring pieces, gear pieces)
    for orbit, size in (("cp", M.NC), ("rp", M.NR), ("gp", M.NG)):
        perm = a[orbit].astype(np.intp)
        # location of each piece: inverse permutation
        loc = np.empty_like(perm)
        np.put_along_axis(loc, perm, np.arange(size)[None, :].repeat(n, 0), axis=1)
        out[orbit + "_loc"] = loc
    gcol = M.GEAR_COLOR[a["gp"]]
    gok = (gcol == M.GEAR_COLOR[None, :]) & (a["go"] == 0)
    out["intact gear pairs"] = sum(gok[:, i] & gok[:, j] for i, j in SIB).astype(float)
    ringtype = M.IS_CENTER[a["rp"]] == M.IS_CENTER[None, :]
    out["ring pieces in home slot type"] = ringtype.sum(1).astype(float)
    out["solved corners"] = ((a["cp"] == np.arange(M.NC)) & (a["co"] == 0)).sum(1).astype(float) - 1
    out["solved gear slots"] = gok.sum(1).astype(float)
    rok = (a["rp"] == np.arange(M.NR)) & ((a["ro"] == 0) | M.IS_CENTER[None, :])
    out["solved ring slots"] = rok.sum(1).astype(float)
    out["corner parity"] = np.array([M._parity(r) for r in a["cp"]], float)
    # sibling gear pairs still on a common edge (any spin): a 2-piece correlation
    loc = out["gp_loc"]
    edge_of = np.full(M.NG, -1)
    for k, (i, j) in enumerate(SIB):
        edge_of[i] = edge_of[j] = k
    out["sibling gears sharing an edge"] = sum((edge_of[loc[:, i]] == edge_of[loc[:, j]]) for i, j in SIB).astype(float)
    # adjacent corner pairs that are still adjacent (share a face) and an edge piece in a home-ring slot
    return out


def compare(s, ref):
    """Worst p-value over all tests and a table of the aggregate statistics."""
    rows, pmin, tv_max = [], 1.0, 0.0
    for orbit, size, pieces in (("cp", M.NC, range(7)), ("rp", M.NR, range(M.NR)), ("gp", M.NG, range(M.NG))):
        x, y = s[orbit + "_loc"], ref[orbit + "_loc"]
        for p in pieces:
            cx = np.bincount(x[:, p], minlength=size)
            cy = np.bincount(y[:, p], minlength=size)
            keep = (cx + cy) > 0
            _, pv, _, _ = stats.chi2_contingency(np.stack([cx[keep], cy[keep]]))
            tv = 0.5 * np.abs(cx / cx.sum() - cy / cy.sum()).sum()  # = EMD under the 0/1 metric
            pmin, tv_max = min(pmin, pv), max(tv_max, tv)
    rows.append(("per-piece location (min p, max TV/EMD)", pmin, tv_max))
    for k in ("corner parity", "intact gear pairs", "sibling gears sharing an edge", "ring pieces in home slot type",
              "solved corners", "solved gear slots", "solved ring slots"):
        x, y = s[k], ref[k]
        pm = stats.ttest_ind(x, y, equal_var=False).pvalue
        pv = stats.levene(x, y).pvalue
        rows.append((k, min(pm, pv), (x.mean(), y.mean(), x.var(), y.var())))
        pmin = min(pmin, pm, pv)
    return pmin, rows


# The scrambler's move set (scramble.random_move): faces U, R, F, each as X, X', X2, X'2; never the same
# face twice in a row, so the fanout is 8 at every step.  Half-turn metric: every move counts 1.
HTM_MOVES = [f + t for f in "URF" for t in ("", "'", "2", "'2")]
HTM_TABLES = tables([M.element(M.parse(m)) for m in HTM_MOVES])
FACE = np.repeat(np.arange(3), 4)


def draw_moves(rng, prev, n):
    """Uniform among the 8 moves on a face other than that of `prev` (prev = None: all 12)."""
    if prev is None:
        return rng.integers(0, 12, n)
    other = rng.integers(0, 8, n)                       # 0..7 -> the two other faces x four turns
    face = (FACE[prev] + 1 + other // 4) % 3
    return face * 4 + other % 4


def random_move_batch(n, length, seed, parity_fix=None):
    """`length` random half-turn-metric moves per puzzle (parity_fix is accepted for compatibility and ignored:
    with double turns in the set, corner parity is not tied to the length)."""
    rng = np.random.default_rng(seed)
    b = Batch(n)
    prev = None
    for i in range(length):
        mv = draw_moves(rng, prev, n)
        b.apply(mv, HTM_TABLES)
        prev = mv
    b.normalize()
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--lengths", default="20,40,60,80,100,150,200,300,500")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    t = time.time()
    ref = statistics(Batch.uniform(args.n, seed=args.seed))
    walk = statistics(random_move_batch(args.n, 5000, seed=args.seed + 1))
    fixed = statistics(random_move_batch(args.n, 200, seed=3, parity_fix=False))
    print("fixed-length 200-move scrambles, mean corner parity:", fixed["corner parity"].mean())
    p, rows = compare(walk, ref)
    print(f"reference check: 5000-move walk vs exact uniform sampler: min p = {p:.3g}  [{time.time() - t:.0f}s]")
    for r in rows:
        print("   ", r)
    print(f"\n{'N':>5} {'min p':>10}  " + "  ".join(f"{k[:22]:>22}" for k, *_ in rows[1:]) + "   max TV")
    for L in [int(x) for x in args.lengths.split(",")]:
        s = statistics(random_move_batch(args.n, L, seed=args.seed * 1000 + L))
        p, rows = compare(s, ref)
        cells = "  ".join(f"{r[2][0]:9.4f}/{r[2][1]:<9.4f}"[:22].rjust(22) for r in rows[1:])
        print(f"{L:5d} {p:10.3g}  {cells}   {rows[0][2]:.4f}", flush=True)


if __name__ == "__main__":
    main()
