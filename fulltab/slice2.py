"""Full table to 2(b+2)-1 HTM from the b-HTM ball with TWO generated levels, never holding either whole:
  A: level b+1 per label slice (from levels b and b-1), saved as (parent row, run) with hashes;
  B: level b+2 candidates from every level-(b+1) state, binned by label slice to disk as (level-(b+1) global index, run);
  C: per slice: stored states + level b+1 + deduplicated level b+2, class join (mirror optional), entries to disk;
  D: merge the slice outputs by label keeping the cheapest.
argv: rung, b, ns [mirror=0] [hbits=30].  Resumable: every stage skips the slices whose files exist (rerun the same command).
Output: data/full_rung{r}_t{b+2}{m}.npy and _wit.npz (ids: ball row;
GEN1 + global level-(b+1) index; GEN2 + (global level-(b+1) index << 5 | run))."""
import sys, os, time, glob, numpy as np, coset as CS, model as M, ladder as L
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import ftlib as F
r, b, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); mir = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False; hbits = int(sys.argv[5]) if len(sys.argv) > 5 else 30
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"); W = os.path.join(D, f"t{b + 2}_{r}"); os.makedirs(W, exist_ok=True)
T0 = time.time(); log = lambda s: print(f"{s} ({time.time() - T0:.0f}s)", flush=True)
spG, tgt = {"7": (CS.Spec(canon="A1"), "F16"), "5": (CS.Spec(canon="H1p"), "H1"), "4": (CS.Spec(canon="Y"), "H1p")}[r]; spT = CS.Spec(canon=tgt)
g1, g2, maxcost = b + 1, b + 2, 2 * (b + 2) - 1
GEN1, GEN2 = np.uint64(1) << np.uint64(40), np.uint64(1) << np.uint64(41)
ST = np.load(os.path.join(D, f"ball{b}.npy"), mmap_mode="r"); C = np.load(os.path.join(D, f"ball{b}_cost.npy")); N = len(C)
import subprocess
def avail():
    v = {l.split(":")[0]: int(l.split(":")[1].strip(" .")) for l in subprocess.run(["vm_stat"], capture_output=True, text=True).stdout.splitlines()[1:] if ":" in l}
    return os.sysconf("SC_PAGESIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9 - (v["Pages wired down"] + v["Anonymous pages"] + v["Pages occupied by compressor"]) * 16384 / 1e9
if not os.path.exists(os.path.join(W, "BM.npy")):
    BM = np.lib.format.open_memmap(os.path.join(W, "BM_build.npy"), mode="w+", dtype=np.uint8, shape=(N, 64))
    for c0 in range(0, N, 20_000_000): BM[c0:c0 + 20_000_000] = F.maps50(np.ascontiguousarray(ST[c0:c0 + 20_000_000]))
    BM.flush(); del BM; os.replace(os.path.join(W, "BM_build.npy"), os.path.join(W, "BM.npy"))
BM = np.load(os.path.join(W, "BM.npy"), mmap_mode="r"); del ST
glab = F.labels_of_maps(BM, spG); BH = np.sort(F.maps_hash(BM)); slice_of = F.slice_of(glab, ns)
runs = [f + a for f in "URF" for a in ("", "2", "3", "4", "'", "'2", "'3")]
RM = F.maps50(CS.pack_states([M.apply_seq(M.State(), w) for w in runs])); RC = np.array([L.word_htm(w) for w in runs], np.int8)
lvl1 = np.flatnonzero(C == b); lvl2 = np.flatnonzero(C == b - 1); r1 = np.flatnonzero(RC == 1); r2 = np.flatnonzero(RC == 2)
par1 = BM[lvl1[0]:lvl1[-1] + 1]; par2 = BM[lvl2[0]:lvl2[-1] + 1]; RM1 = np.ascontiguousarray(RM[r1]); RM2 = np.ascontiguousarray(RM[r2])
log(f"ball {b}: {N:,} states; level {b}: {len(lvl1):,}; slices {ns}")
# ---- A: level g1 per slice
n1 = np.zeros(ns, np.int64); bm1 = F.Bitmap(37 if b >= 9 else 30); resumed = 0
def savez(path, **kw): np.savez(path + ".tmp.npz", **kw); os.replace(path + ".tmp.npz", path)     # never leave a truncated file
for k in range(ns):
    fk = os.path.join(W, f"lvl{g1}_s{k}.npz")
    if os.path.exists(fk):
        gh = np.load(fk)["h"]; n1[k] = len(gh); bm1.add(gh); resumed += 1; continue
    cap = int((len(par1) * len(r1) + len(par2) * len(r2)) / ns * 1.6) + 1_000_000
    a = F.gen_slice(par1, RM1, spG, k, ns, cap); c = F.gen_slice(par2, RM2, spG, k, ns, cap)
    gh = np.concatenate([a[2], c[2]]); gp = np.concatenate([lvl1[a[3]], lvl2[c[3]]]).astype(np.int32); gr = np.concatenate([r1[a[4]], r2[c[4]]]).astype(np.int8); del a, c
    o = np.argsort(gh, kind="stable"); gh = gh[o]; first = np.r_[True, gh[1:] != gh[:-1]]; o = o[first]; gh = gh[first]
    keep = BH[np.searchsorted(BH, gh).clip(0, len(BH) - 1)] != gh; o = o[keep]; gh = gh[keep]
    savez(fk, p=gp[o], r=gr[o], h=gh); n1[k] = len(o); bm1.add(gh)
    if k % 8 == 0 or k == ns - 1: log(f"  A slice {k + 1}/{ns}: {len(o):,} level-{g1} states")
off1 = np.r_[0, np.cumsum(n1)]; log(f"level {g1}: {int(n1.sum()):,} states ({resumed} of {ns} slices resumed from disk)")
# ---- B: level g2 candidates binned by slice; a source slice is redone unless all its ns bin files exist
resumed = 0
for k in range(ns):
    if all(os.path.exists(os.path.join(W, f"bin{g2}_s{j}_{k}.npz")) for j in range(ns)): resumed += 1; continue
    z = np.load(os.path.join(W, f"lvl{g1}_s{k}.npz")); maps = F.child_maps(BM, z["p"], z["r"], RM)
    cap = int(len(maps) * len(r1) / ns * 1.8) + 1_000_000
    bins = F.gen_bins(maps, int(off1[k]), RM1, spG, ns, cap); del maps                       # level g2 = cost-1 runs from level g1 ...
    for j, (bp_, br_) in enumerate(bins): savez(os.path.join(W, f"bin{g2}_s{j}_{k}.npz"), p=bp_, r=r1[br_].astype(np.int8))      # run indices into the full list
    if k % 8 == 0 or k == ns - 1: log(f"  B from slice {k + 1}/{ns}: {sum(len(x[0]) for x in bins):,} candidates binned")
if all(os.path.exists(os.path.join(W, f"bin{g2}_s{j}_stored.npz")) for j in range(ns)): resumed += 1
else:   # cost-2 runs from level b (stored) also reach level g2: bin those too
    cap = int(len(par1) * len(r2) / ns * 1.8) + 1_000_000; bins = F.gen_bins(par1, 0, RM2, spG, ns, cap)
    for j, (bp_, br_) in enumerate(bins): savez(os.path.join(W, f"bin{g2}_s{j}_stored.npz"), p=(lvl1[0] + bp_).astype(np.int32), r=r2[br_].astype(np.int8))
log(f"  B: cost-2 children of the stored level binned ({resumed} of {ns + 1} source slices resumed from disk)")
# ---- C: per slice join
if not os.path.exists(os.path.join(W, "R1.npy")):
    np.save(os.path.join(W, "P1.npy"), np.concatenate([np.load(os.path.join(W, f"lvl{g1}_s{k}.npz"))["p"] for k in range(ns)]))
    np.save(os.path.join(W, "R1_build.npy"), np.concatenate([np.load(os.path.join(W, f"lvl{g1}_s{k}.npz"))["r"] for k in range(ns)])); os.replace(os.path.join(W, "R1_build.npy"), os.path.join(W, "R1.npy"))
P1 = np.load(os.path.join(W, "P1.npy"), mmap_mode="r"); R1 = np.load(os.path.join(W, "R1.npy"), mmap_mode="r"); CH = 40_000_000
resumed = 0
tot = np.zeros(2, np.int64); n2tot = 0
for j in range(ns):
    if os.path.exists(os.path.join(W, f"out_s{j}.npz")): resumed += 1; continue
    if j == 0 or resumed: log(f"  C: {resumed} slices resumed from disk"); resumed = 0
    sel = np.flatnonzero(slice_of == j); z = np.load(os.path.join(W, f"lvl{g1}_s{j}.npz"))
    m1 = F.child_maps(BM, z["p"], z["r"], RM); id1 = GEN1 + (off1[j] + np.arange(len(m1))).astype(np.uint64); del z
    files = sorted(glob.glob(os.path.join(W, f"bin{g2}_s{j}_*.npz"))); parts = [np.load(f) for f in files]
    bp_ = np.concatenate([q["p"] for q in parts]); br_ = np.concatenate([q["r"] for q in parts])
    from_stored = np.concatenate([np.full(len(q["p"]), f.endswith("_stored.npz")) for q, f in zip(parts, files)]); del parts
    # two kinds of candidates: from level g1 (index into P1/R1) and from the stored level b (ball rows, file 'stored')
    def maps_of(idx):
        out = np.empty((len(idx), 64), np.uint8); fs = from_stored[idx]; a = np.flatnonzero(~fs); s_ = np.flatnonzero(fs)
        if len(a): ia = idx[a]; pa = bp_[ia]; out[a] = F.child_maps2(BM, P1[pa], R1[pa], br_[ia], RM)
        if len(s_): is_ = idx[s_]; out[s_] = F.child_maps(BM, bp_[is_], br_[is_], RM)
        return out
    tC = [time.time()]; tick = lambda: tC.append(time.time())
    h2 = np.empty(len(bp_), np.uint64)                                       # pass 1: hashes only
    for c0 in range(0, len(bp_), CH): h2[c0:c0 + CH] = F.maps_hash(maps_of(np.arange(c0, min(c0 + CH, len(bp_)))))
    o = np.argsort(h2, kind="quicksort"); hs = h2[o]; first = np.r_[True, hs[1:] != hs[:-1]]; o = o[first]; hs = hs[first]   # any witness of a state will do
    keep = (BH[np.searchsorted(BH, hs).clip(0, len(BH) - 1)] != hs) & ~bm1.test(hs); o = np.sort(o[keep]); del h2, hs, first, keep; tick()
    n2 = len(o); m2 = np.empty((n2, 64), np.uint8)                           # pass 2: maps of the kept candidates
    for c0 in range(0, n2, CH): m2[c0:c0 + CH] = maps_of(o[c0:c0 + CH])
    bo = bp_[o].astype(np.uint64); ro = br_[o].astype(np.uint64)
    id2 = np.where(from_stored[o], GEN2 + (bo << np.uint64(5)) | ro | (np.uint64(1) << np.uint64(39)), GEN2 + (bo << np.uint64(5)) | ro)
    n2tot += n2; del bp_, br_, from_stored, o, bo, ro; tick()
    maps = np.concatenate([BM[sel], m1, m2]); cost = np.concatenate([C[sel], np.full(len(m1), g1, np.int8), np.full(n2, g2, np.int8)])
    lab = F.labels_of_maps(maps, spG); ids = np.concatenate([sel.astype(np.uint64), id1, id2]); del m1, m2
    tl = F.labels_of_maps(maps, spT) & ~F.CM
    tick(); S, Cs, gst, nrep, ids = F.order_maps_reps(maps, cost, lab, tl, ids); del maps, cost, lab, tl; tick()
    ht = np.zeros(1 << hbits, np.uint64); EWP = np.zeros(1 << hbits, np.uint64); EWQ = np.zeros(1 << hbits, np.uint64)
    st = F.join50maps(S, Cs, gst, nrep, spT, maxcost, hbits, ht, EWP, EWQ, ids, mir=mir); tot += [st["pairs"], st["lost"]]
    tick()
    used = np.flatnonzero(ht != 0); savez(os.path.join(W, f"out_s{j}.npz"), t=ht[used], p=EWP[used], q=EWQ[used]); del ht, EWP, EWQ, S, Cs
    log(f"  C slice {j + 1}/{ns}: {len(sel):,} stored + {len(id1):,} level-{g1} + {n2:,} level-{g2}; {st['pairs']:,} pairs, lost {st['lost']}; {len(used):,} entries; {avail():.0f} GB available; "
        f"steps hash/dedupe {tC[1] - tC[0]:.0f}, maps {tC[2] - tC[1]:.0f}, labels {tC[3] - tC[2]:.0f}, order {tC[4] - tC[3]:.0f}, join {tC[5] - tC[4]:.0f} s")
log(f"level {g2}: {n2tot:,} states; {tot[0]:,} pairs")
# ---- D: merge by label (16 label ranges to bound memory)
outs = sorted(glob.glob(os.path.join(W, "out_s*.npz"))); parts_t, parts_p, parts_q = [], [], []
for rg in range(16):
    lo, hi = np.uint64(rg) << np.uint64(60), np.uint64(rg + 1) << np.uint64(60) if rg < 15 else np.uint64(2**64 - 1)
    ts, ps, qs = [], [], []
    for f in outs:
        z = np.load(f); t = z["t"]; m = (t >= lo) & (t < hi) if rg < 15 else (t >= lo); ts.append(t[m]); ps.append(z["p"][m]); qs.append(z["q"][m])
    t = np.concatenate(ts); pq = np.concatenate(ps); qq = np.concatenate(qs); o = np.argsort(t, kind="stable"); t = t[o]; pq = pq[o]; qq = qq[o]
    hi_ = t & ~F.CM; first = np.r_[True, hi_[1:] != hi_[:-1]]               # sorted by label then cost: the first of each label is the cheapest
    parts_t.append(t[first]); parts_p.append(pq[first]); parts_q.append(qq[first])
t = np.concatenate(parts_t); wp = np.concatenate(parts_p); wq = np.concatenate(parts_q)
name = f"full_rung{r}_t{g2}{'m' if mir else ''}"; np.save(os.path.join(D, name + ".npy"), t); np.savez(os.path.join(D, name + "_wit.npz"), p=wp, q=wq)
cnt = np.bincount((t & F.CM).astype(np.int64))
log(f"rung {r}: {len(t):,} classes to {maxcost} (exact) {'mirrored' if mir else ''}; per HTM " + ", ".join(f"{c}: {int(v):,}" for c, v in enumerate(cnt) if v))
