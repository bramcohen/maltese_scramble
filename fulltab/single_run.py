"""Base measurement per rung: one run on random inputs of the rung's own starting group (one frame, no beam, merges
only inside the rung's own sequence).  Reports mean HTM, failure rate, mean solutions found (distinct landings within
the rung's window), seconds per run, and the table's size.  argv: n inputs.  Env: SEED, RUNGS=12,3 (which rungs), RAW=1
(rungs 1 + 2 without their first-move retry: the base failure rate), RUNG7=small."""
import sys, os, time, pickle, numpy as np, model as M, coset as CS, ladder as L
sys.path.insert(0, "fulltab"); sys.path.insert(0, "."); import table as FT
n = int(sys.argv[1]) if len(sys.argv) > 1 else 24; rng = np.random.default_rng(int(os.environ.get("SEED", "7")))
R = FT.full_rungs(os.environ.get("RUNG7")); lad = L.Ladder(rung7=os.environ.get("RUNG7")); P = L.Ladder.PARAMS
from mkbsgs import random_Z_state
gens = {"4": (pickle.load(open("tables/bsgs_Y.pkl", "rb"))["gens"], 400), "5": (pickle.load(open("tables/bsgs_H1p.pkl", "rb"))["gens"], 400),
        "6": (pickle.load(open("tables/bsgs_H1.pkl", "rb"))["gens"], 400), "7": (pickle.load(open("tables/bsgs_A1.pkl", "rb"))["gens"], 400), "8": (pickle.load(open("tables/bsgs_F16.pkl", "rb"))["gens"], 400)}
def inputs(r): return [M.random_state(rng) for _ in range(n)] if r == "12" else [random_Z_state(rng) for _ in range(n)] if r == "3" else [M.apply_seq(M.State(), " ".join(gens[r][0][rng.integers(len(gens[r][0]))] for _ in range(gens[r][1]))) for _ in range(n)]
def size(r):
    if r in R:
        t = R[r].table; nb = t.tab.nbytes + sum(getattr(t, a).nbytes for a in ("wp", "wq", "wit", "rows", "DER", "bwp", "bwq") if hasattr(t, a))
        return f"{len(t.tab):,} classes to {t.tmax} HTM, {nb / 1e9:.1f} GB (+ forward {R[r].fwd.maps.nbytes / 1e9:.1f} GB)"
    return f"{lad.mI1.total_back:,} classes (complete), {lad.mI1.flat_keys.nbytes / 1e9:.2f} GB" if r == "6" else f"{len(lad.f8):,} elements (complete)"
if os.environ.get("RAW") == "1": R["12"].prefix_retry = False; R["12"].deeper = []          # rungs 1 + 2 without the first-move retry
for r in os.environ.get("RUNGS", "12,3,4,5,6,7,8").split(","):
    sts = inputs(r); costs = []; nsol = []; t = time.time(); miss = 0; w = P[r]["window"] if r in P else 0
    for st in sts:
        if r in R: cands = R[r].solve_many([st], window=w)[0] if r != "12" else R["12"].solve_many([st], window=w)[0]       # one frame
        elif r == "6": cands = sorted(lad._lookups("6", lad.mI1, st), key=lambda c: c[0])
        else: c8, path = lad._finish(st); cands = [(c8, [CS.inverse_word(x) for x in reversed(path)], [], None)]
        if not cands: miss += 1; continue
        costs.append(L.htm(" ".join(cands[0][1] + cands[0][2]))); nsol.append(len(cands))
    dt = (time.time() - t) / n
    print(f"rung {r}: {n} inputs: mean {np.mean(costs):.2f} HTM, failed {miss}/{n}, mean solutions within window {w}: {np.mean(nsol):.1f}, {dt:.2f} s per run; table {size(r)}", flush=True)
