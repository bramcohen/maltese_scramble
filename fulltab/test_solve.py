"""End to end on a table rung: random inputs of its starting group, the meet, the reconstructed moves verified.
argv: rung (12, 3, 4, 5 or 7) [n=12]."""
import sys, os, time, pickle, numpy as np, model as M, coset as CS, ladder as L
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import table as T
r = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 12; R = T.full_rungs()[r]
print(f"rung {r}: table {len(R.table.tab):,} classes to {R.table.tmax}, forward {len(R.fwd.maps):,} elements to {R.fwd.E}", flush=True)
rng = np.random.default_rng(11); tot = []
if r == "12": gens, nw = None, 0
elif r == "3": gens, nw = None, 0
else: gens = pickle.load(open({"4": "tables/bsgs_Y.pkl", "5": "tables/bsgs_H1p.pkl", "7": "tables/bsgs_A1.pkl"}[r], "rb"))["gens"]; nw = 400
for i in range(n):
    st = M.random_state(rng) if r == "12" else __import__("mkbsgs").random_Z_state(rng) if r == "3" else M.apply_seq(M.State(), " ".join(gens[rng.integers(len(gens))] for _ in range(nw)))
    t = time.time(); res = R.solve(st); dt = time.time() - t
    if res is None: print(f"  input {i}: no solution ({dt:.1f}s)", flush=True); continue
    cost, seq, _, land = res; tot.append(cost)
    print(f"  input {i}: {cost} HTM, {len(seq)} runs, verified ({dt:.1f}s)", flush=True)
print(f"rung {r}: mean {np.mean(tot):.2f} HTM on {len(tot)}/{n}", flush=True)
