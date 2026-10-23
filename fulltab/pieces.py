"""Chaining pieces for a rung: the in-group single runs plus every excursion (an in-group sequence with no in-group
point strictly inside) up to 2h HTM, as 50-point maps with costs.  Excursions come from a ball over all moves that is
not expanded through the starting group, paired within group cosets outside the group itself.
Saves fulltab/data/pieces_rung{r}_h{h}.npz (maps, cost).  argv: h, rung."""
import sys, os, time, numpy as np, coset as CS, model as M, ladder as L
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import ftlib as F
h = int(sys.argv[1]); r = sys.argv[2]; D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
T0 = time.time(); log = lambda s: print(f"{s} ({time.time() - T0:.0f}s)", flush=True)
isUD = np.zeros(18, np.int8); isUD[L.UDE] = 1; isEQ = np.zeros(24, np.int8); isEQ[L.EQG] = 1
Z = CS.Spec(pos={"r": (np.ones(18), isUD), "g": (np.ones(24), isEQ)}, fun={"c": (np.zeros(8, np.int8), np.eye(8, dtype=np.int8))})
spG = Z if r == "3" else CS.Spec(canon={"4": "Y", "5": "H1p", "7": "A1"}[r])
k0 = spG.keys(CS.pack_states([M.State()]))[0]
runs = [f + a for f in "URF" for a in ("", "2", "3", "4", "'", "'2", "'3")]
cost = [L.word_htm(x) for x in runs]; E = [CS.pack_elem(M.element(M.parse(x))) for x in runs]
full = CS.Spec(canon={"levels": []})
cap = {4: 20_000, 5: 150_000, 6: 1_200_000, 7: 10_000_000, 8: 80_000_000}[h]
ST = np.zeros((cap, 100), np.int8); CST = np.zeros(cap, np.int8); ING = np.zeros(cap, bool); PAR = np.full(cap, -1, np.int32); PRUN = np.full(cap, -1, np.int8)
st0 = CS.pack_states([M.State()]); ST[0] = st0[0]; seen = CS.KeySet(1 << 27); seen.insert(full.keys(st0))
lvl = {0: (0, 1)}; n_tot = 1
for n in range(1, h + 1):
    start = n_tot
    for ri in range(21):
        if n - cost[ri] not in lvl: continue
        a, k = lvl[n - cost[ri]]
        src = np.flatnonzero(~ING[a:a + k]) + a
        for c0 in range(0, len(src), 5_000_000):
            s, kk = full.apply(np.ascontiguousarray(ST[src[c0:c0 + 5_000_000]]), E[ri]); new = np.where(seen.insert(kk))[0]
            if len(new):
                assert n_tot + len(new) <= cap
                ST[n_tot:n_tot + len(new)] = s[new]; CST[n_tot:n_tot + len(new)] = n
                PAR[n_tot:n_tot + len(new)] = src[c0:c0 + 5_000_000][new]; PRUN[n_tot:n_tot + len(new)] = ri; n_tot += len(new)
    ING[start:n_tot] = spG.keys(ST[start:n_tot]) == k0
    lvl[n] = (start, n_tot - start)
ST = ST[:n_tot]; CST = CST[:n_tot]; ING = ING[:n_tot]; PAR = PAR[:n_tot]; PRUN = PRUN[:n_tot]
# in-group single runs (cost 1 or 2)
single = [(CS.pack_states([M.apply_seq(M.State(), w)])[0], c) for w, c in zip(runs, cost)]
single = [(s, c) for s, c in single if spG.keys(np.stack([s]))[0] == k0]
# excursions: pairs in the same coset, outside the group
lab = spG.keys(ST); keep = lab != k0
o = np.flatnonzero(keep)[np.argsort(lab[keep], kind="stable")]; sl = lab[o]; cut = np.flatnonzero(sl[1:] != sl[:-1]) + 1
gst = np.r_[0, cut, len(o)].astype(np.int64); S = np.ascontiguousarray(ST[o]); Cs = np.ascontiguousarray(CST[o])
ec, ew, st = F.elems50(S, Cs, gst, 2 * h, 28)
EM = F.pair_maps50(S, ew)
def seq(row):
    out = []
    while row > 0: out.append(runs[int(PRUN[row])]); row = int(PAR[row])
    return out[::-1]
inv = lambda w: w[0] + ("" if w[1:].startswith("'") else "'") + w[1:].lstrip("'")
pr = o[(ew >> np.uint64(32)).astype(np.int64)]; qr = o[(ew & np.uint64(0xffffffff)).astype(np.int64)]       # ball rows of the witnesses
single_names = [(w, c) for w, c in zip(runs, cost) if spG.keys(CS.pack_states([M.apply_seq(M.State(), w)]))[0] == k0]
eseqs = [" ".join(seq(int(a)) + [inv(w) for w in seq(int(b))[::-1]]) for a, b in zip(pr, qr)]
maps = np.concatenate([F.maps50(np.stack([s for s, c in single])), EM]); pc = np.r_[[c for s, c in single], ec].astype(np.int8)
allseq = np.array([w for w, c in single_names] + eseqs)
o2 = np.argsort(pc, kind="stable"); maps = np.ascontiguousarray(maps[o2]); pc = np.ascontiguousarray(pc[o2]); allseq = allseq[o2]
np.savez(os.path.join(D, f"pieces_rung{r}_h{h}.npz"), maps=maps, cost=pc, seq=allseq)
cnt = np.bincount(pc.astype(np.int64))
log(f"rung {r}: {len(single)} single runs + {len(ec):,} excursions up to {2 * h} HTM (exact to {2 * h - 1}); by HTM " +
    ", ".join(f"{c}: {int(v):,}" for c, v in enumerate(cnt) if v))
