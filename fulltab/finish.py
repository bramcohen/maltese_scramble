"""Rung 8's finish table, tables/finish_full.pkl.gz: for every element of F16 the shorter of its finish over F16's own
moves (ladder.finish_dijkstra) and its shortest finish over all moves up to 18 HTM, by meet in the middle on the
9-HTM ball: two ball states in one F16-coset give the element p . inverse(q) with p's moves then q's inverted.
{state key: (HTM, moves from solved)}.  About 8 minutes, 50 GB of memory, a 31 GB scratch file under data/mm."""
import os, sys, time, gzip, pickle, numpy as np, model as M, coset as CS, ladder as L
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import ftlib as F
T0 = time.time(); log = lambda s: print(f"{s} ({time.time() - T0:.0f}s)", flush=True)
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"); OUT = os.environ.get("TABLES", "tables"); MAXCOST = 18
ST = np.load(os.path.join(D, "ball9.npy"), mmap_mode="r"); C = np.load(os.path.join(D, "ball9_cost.npy")); N = len(C); W = F.Witness(9, D)
spF = CS.Spec(canon="F16"); lab = np.empty(N, np.uint64)
for c0 in range(0, N, 20_000_000): lab[c0:c0 + 20_000_000] = spF.keys(np.ascontiguousarray(ST[c0:c0 + 20_000_000]))
o = np.argsort(lab, kind="stable"); sl = lab[o]; del lab
cut = np.flatnonzero(sl[1:] != sl[:-1]) + 1; gst = np.r_[0, cut, N]; sizes = np.diff(gst); del sl, cut
keep = np.repeat(sizes > 1, sizes); o = o[keep]; gst = np.r_[0, np.cumsum(sizes[sizes > 1])].astype(np.int64); del keep, sizes
log(f"{N:,} ball states labelled by F16-coset: {len(gst) - 1:,} cosets with two or more states, {len(o):,} states in them")
S = F.gather_rows(ST, o); Cs = np.ascontiguousarray(C[o]); del C
ec, ew, st = F.elems50(S, Cs, gst, MAXCOST, 22); del S
log(f"{len(ec):,} distinct F16 elements within {MAXCOST} HTM over all moves ({st['pairs']:,} pairs, {st['seconds']:.0f}s in the join)")
out = {k: (L.htm(" ".join(p)), list(p)) for k, (c, p) in L.finish_dijkstra().items()}; better = 0
for c, w in zip(ec.tolist(), ew.tolist()):
    p, q = int(o[w >> 32]), int(o[w & 0xffffffff])
    for path in (W.element(p, q), W.element(q, p)):              # the element and its inverse (the join keeps one of the two orders)
        k = M.apply_seq(M.State(), " ".join(path)).key(); cc = L.htm(" ".join(path)); assert k in out and cc <= c, (c, cc, path)
        if cc < out[k][0]: out[k] = (cc, path); better += 1
cs = np.array([c for c, p in out.values()])
pickle.dump(out, gzip.open(os.path.join(OUT, "finish_full.pkl.gz"), "wb"))
log(f"finish table: {len(out):,} elements, {better:,} shortened by all-moves finishes; mean {cs.mean():.3f} HTM, max {cs.max()}; saved {OUT}/finish_full.pkl.gz")
for f in os.listdir(os.path.join(D, "mm")):
    if f.startswith(f"gather_{os.getpid()}_"): os.remove(os.path.join(D, "mm", f))
