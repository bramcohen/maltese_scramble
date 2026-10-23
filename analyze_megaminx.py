"""Is the WCA megaminx scramble (7 lines of R++/D++ ... U, 77 moves) biased in piece positions?

Metric: the sum over all 50 pieces of the graph distance from the piece's position to its home position
(orientation ignored).  Compared against exact uniform random positions (even permutations of corners and
of edges) and against long random walks.  Usage: .venv/bin/python analyze_megaminx.py [--n 1000000]
"""
import argparse
import time

import numpy as np
from scipy import stats

import megaminx as X


def describe(states):
    d = X.piece_distances(states)
    tot = d.sum(axis=1)
    return dict(total=tot, corners=d[:, :20].sum(axis=1), edges=d[:, 20:].sum(axis=1),
                home_c=(d[:, :20] == 0).sum(axis=1), home_e=(d[:, 20:] == 0).sum(axis=1), d=d)


def hist(d, lo, hi):
    return np.array([(d == k).sum() for k in range(hi + 1)], float) / d.size


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1_000_000); ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(); rng = np.random.default_rng(args.seed); n = args.n
    t = time.time()
    ref = describe(X.uniform_positions(rng, n)); print(f"uniform reference: {n} states ({time.time() - t:.0f}s)", flush=True)
    walk = describe(X.simulate_batch(rng, n, "random", length=400)); print(f"400-move random walk ({time.time() - t:.0f}s)", flush=True)
    maxd = int(X.DIST.max())
    print("\n%-22s %10s %10s %10s %10s %9s %9s   %s" % ("sample", "total", "sd", "corners", "edges", "home_c", "home_e", "per-piece distance distribution (corners | edges)"))

    def row(name, s):
        hc = hist(s["d"][:, :20], 0, maxd); he = hist(s["d"][:, 20:], 0, maxd)
        print("%-22s %10.3f %10.3f %10.3f %10.3f %9.4f %9.4f   %s | %s" % (
            name, s["total"].mean(), s["total"].std(), s["corners"].mean(), s["edges"].mean(), s["home_c"].mean(), s["home_e"].mean(),
            " ".join("%.4f" % x for x in hc), " ".join("%.4f" % x for x in he)))

    row("uniform (exact)", ref); row("random walk 400", walk)
    results = {}
    for lines in (1, 2, 3, 4, 5, 7, 10, 14):
        s = describe(X.simulate_batch(rng, n, "wca", length=11 * lines)); results[lines] = s
        row(f"WCA {lines} lines ({11 * lines})", s)
    print("\nTests against the exact uniform sample (Welch t on the mean total distance; KS on its distribution;")
    print("chi-square on the per-piece distance histogram; t on the number of home pieces):")
    print("%-22s %12s %12s %12s %12s %12s" % ("sample", "t p(total)", "KS p", "chi2 p corn", "chi2 p edge", "t p(home)"))
    for name, s in [("random walk 400", walk)] + [(f"WCA {k} lines", v) for k, v in results.items()]:
        tp = stats.ttest_ind(s["total"], ref["total"], equal_var=False).pvalue
        ks = stats.ks_2samp(s["total"], ref["total"]).pvalue
        pc = stats.chi2_contingency(np.vstack([np.bincount(s["d"][:, :20].ravel(), minlength=maxd + 1), np.bincount(ref["d"][:, :20].ravel(), minlength=maxd + 1)]))[1]
        pe = stats.chi2_contingency(np.vstack([np.bincount(s["d"][:, 20:].ravel(), minlength=maxd + 1), np.bincount(ref["d"][:, 20:].ravel(), minlength=maxd + 1)]))[1]
        th = stats.ttest_ind(s["home_c"] + s["home_e"], ref["home_c"] + ref["home_e"], equal_var=False).pvalue
        print("%-22s %12.2e %12.2e %12.2e %12.2e %12.2e" % (name, tp, ks, pc, pe, th))
    print(f"\n({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
