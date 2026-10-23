"""A rung's full table to 2h HTM from the (h-1)-HTM ball plus level h generated on the fly, one label slice at a time,
so level h is never held whole.  Per slice k of ns: the ball's states whose group label hashes to k, plus the children
(parent . run) of level h-1 with label in k, deduplicated and with lower-level states removed; then the class join with
representatives.  argv: rung, h, ns [mirror=0] [hbits=30].  Saves fulltab/data/full_rung{r}_s{h}{m}.npy."""
import sys, os, time, numpy as np, coset as CS, model as M, ladder as L
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import ftlib as F
r, h, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); mir = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
hbits = int(sys.argv[5]) if len(sys.argv) > 5 else 30
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
T0 = time.time(); log = lambda s: print(f"{s} ({time.time() - T0:.0f}s)", flush=True)
isUD = np.zeros(18, np.int8); isUD[L.UDE] = 1; isEQ = np.zeros(24, np.int8); isEQ[L.EQG] = 1
Z = CS.Spec(pos={"r": (np.ones(18), isUD), "g": (np.ones(24), isEQ)}, fun={"c": (np.zeros(8, np.int8), np.eye(8, dtype=np.int8))})
spG, tgt = {"3": (Z, "Y"), "4": (CS.Spec(canon="Y"), "H1p"), "5": (CS.Spec(canon="H1p"), "H1"), "7": (CS.Spec(canon="A1"), "F16")}[r]
spT = CS.Spec(canon=tgt)
if r == "3": raise SystemExit("rung 3's group label is not a canonical-form spec; use fulltab.py")
ST = np.load(os.path.join(D, f"ball{h - 1}.npy"), mmap_mode="r"); C = np.load(os.path.join(D, f"ball{h - 1}_cost.npy"))
BM = np.empty((len(C), 64), np.uint8)
for c0 in range(0, len(C), 20_000_000): BM[c0:c0 + 20_000_000] = F.maps50(np.ascontiguousarray(ST[c0:c0 + 20_000_000]))
del ST
glab = F.labels_of_maps(BM, spG); BH = np.sort(F.maps_hash(BM))
runs = [f + a for f in "URF" for a in ("", "2", "3", "4", "'", "'2", "'3")]
RM = F.maps50(CS.pack_states([M.apply_seq(M.State(), w) for w in runs])); RC = np.array([L.word_htm(w) for w in runs], np.int8)
slice_of = F.slice_of(glab, ns)
lvl1 = np.flatnonzero(C == h - 1); lvl2 = np.flatnonzero(C == h - 2); r1 = np.flatnonzero(RC == 1); r2 = np.flatnonzero(RC == 2)
log(f"ball {h - 1}: {len(C):,} states, level {h - 1}: {len(lvl1):,}, level {h - 2}: {len(lvl2):,}; slices {ns}")
ht = np.zeros(1 << hbits, np.uint64); EWP = np.zeros(1 << hbits, np.uint64); EWQ = np.zeros(1 << hbits, np.uint64); tot = np.zeros(3, np.int64); ngen = 0
GEN = np.uint64(1) << np.uint64(40)                       # ids of generated states: GEN + parent row * 32 + run
assert lvl1[-1] - lvl1[0] + 1 == len(lvl1) and lvl2[-1] - lvl2[0] + 1 == len(lvl2)       # levels are contiguous: use views, not copies
par1 = BM[lvl1[0]:lvl1[-1] + 1]; par2 = BM[lvl2[0]:lvl2[-1] + 1]
RM1 = np.ascontiguousarray(RM[r1]); RM2 = np.ascontiguousarray(RM[r2])
for k in range(ns):
    cap = int((len(par1) * len(r1) + len(par2) * len(r2)) / ns * 1.6) + 1_000_000
    a = F.gen_slice(par1, RM1, spG, k, ns, cap); b = F.gen_slice(par2, RM2, spG, k, ns, cap)
    gm = np.concatenate([a[0], b[0]]); gl = np.concatenate([a[1], b[1]]); gh = np.concatenate([a[2], b[2]])
    gid = np.concatenate([GEN + lvl1[a[3]].astype(np.uint64) * np.uint64(32) + r1[a[4]].astype(np.uint64),
                          GEN + lvl2[b[3]].astype(np.uint64) * np.uint64(32) + r2[b[4]].astype(np.uint64)]); del a, b
    # dedupe the children, then drop any that are already in the ball (lower cost)
    o = np.argsort(gh, kind="stable"); gh = gh[o]; first = np.r_[True, gh[1:] != gh[:-1]]; o = o[first]; gh = gh[first]
    pos = np.searchsorted(BH, gh).clip(0, len(BH) - 1); keep = BH[pos] != gh
    o = o[keep]; gm = gm[o]; gl = gl[o]; gid = gid[o]; gcost = np.full(len(o), h, np.int8); ngen += len(o)
    sel = np.flatnonzero(slice_of == k)
    maps = np.concatenate([BM[sel], gm]); cost = np.concatenate([C[sel], gcost]); lab = np.concatenate([glab[sel], gl])
    ids = np.concatenate([sel.astype(np.uint64), gid]); del gm, gl, gid
    tl = F.labels_of_maps(maps, spT) & ~F.CM
    S, Cs, gst, nrep, ids = F.order_maps_reps(maps, cost, lab, tl, ids); del maps, cost, lab, tl
    st = F.join50maps(S, Cs, gst, nrep, spT, 2 * h, hbits, ht, EWP, EWQ, ids, mir=mir); tot += [st["pairs"], st["lost"], 0]
    log(f"  slice {k + 1}/{ns}: {len(sel):,} stored + {len(o):,} new level-{h} states, {len(gst) - 1:,} groups; {st['pairs']:,} pairs ({st['seconds']:.0f}s), lost {st['lost']}")
    del S, Cs, gst, nrep
used = np.flatnonzero(ht != 0); tab = ht[used]; wp = EWP[used]; wq = EWQ[used]; del ht, EWP, EWQ, used
o = np.argsort(tab, kind="stable"); tab = tab[o]; wp = wp[o]; wq = wq[o]
name = f"full_rung{r}_s{h}{'m' if mir else ''}"; np.save(os.path.join(D, name + ".npy"), tab); np.savez(os.path.join(D, name + "_wit.npz"), p=wp, q=wq)
cnt = np.bincount((tab & F.CM).astype(np.int64))
log(f"rung {r}: level {h} has {ngen:,} states; {tot[0]:,} pairs, lost {tot[1]}; {len(tab):,} classes to {2 * h}{' (mirror)' if mir else ''}; per HTM "
    + ", ".join(f"{c}: {int(v):,}" for c, v in enumerate(cnt) if v))
