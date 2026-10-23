"""Stage D of slice2.py, streaming: the per-slice join outputs (out_s*.npz: label | cost, witness p, q) hold every class
once per slice that produced it; keep the cheapest per label.  Pass 1 partitions each file by label range into part
files; pass 2 sorts and dedupes one range at a time and appends to the final arrays.  Resumable.  argv: rung, b [mirror=1] [ranges=64]."""
import sys, os, glob, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import ftlib as F
r, b = sys.argv[1], int(sys.argv[2]); mir = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True; NR = int(sys.argv[4]) if len(sys.argv) > 4 else 64
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"); W = os.path.join(D, f"t{b + 2}_{r}"); P = os.path.join(W, "parts"); os.makedirs(P, exist_ok=True)
T0 = time.time(); log = lambda s: print(f"{s} ({time.time() - T0:.0f}s)", flush=True)
outs = sorted(glob.glob(os.path.join(W, "out_s*.npz"))); shift = np.uint64(64 - NR.bit_length() + 1) if NR & (NR - 1) == 0 else None; assert shift is not None, "ranges must be a power of two"
def rng_of(t): return (t >> shift).astype(np.int64)                     # top bits of the label
for f in outs:                                                           # pass 1
    j = os.path.basename(f)[5:-4]
    if os.path.exists(os.path.join(P, f"done_{j}")): continue
    z = np.load(f); t, p, q = z["t"], z["p"], z["q"]; g = rng_of(t); o = np.argsort(g, kind="stable"); t, p, q, g = t[o], p[o], q[o], g[o]
    cuts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1, len(g)]
    for a, b_ in zip(cuts[:-1], cuts[1:]):
        np.savez(os.path.join(P, f"r{int(g[a])}_{j}.tmp.npz"), t=t[a:b_], p=p[a:b_], q=q[a:b_]); os.replace(os.path.join(P, f"r{int(g[a])}_{j}.tmp.npz"), os.path.join(P, f"r{int(g[a])}_{j}.npz"))
    open(os.path.join(P, f"done_{j}"), "w").close()
    if int(j) % 16 == 0: log(f"  pass 1: {f.split('/')[-1]} ({len(t):,} entries) partitioned")
log("pass 1 done")
name = f"full_rung{r}_t{b + 2}{'m' if mir else ''}"; parts_t, parts_p, parts_q = [], [], []
for rg in range(NR):                                                     # pass 2
    fr = os.path.join(P, f"range{rg}.npz")
    if not os.path.exists(fr):
        fs = sorted(glob.glob(os.path.join(P, f"r{rg}_*.npz")))
        if fs:
            zs = [np.load(x) for x in fs]; t = np.concatenate([z["t"] for z in zs]); p = np.concatenate([z["p"] for z in zs]); q = np.concatenate([z["q"] for z in zs]); del zs
            o = np.argsort(t, kind="quicksort"); t = t[o]; p = p[o]; q = q[o]; hi = t & ~F.CM; first = np.r_[True, hi[1:] != hi[:-1]]   # label then cost: first = cheapest
            t, p, q = t[first], p[first], q[first]
        else: t = np.zeros(0, np.uint64); p = np.zeros(0, np.uint64); q = np.zeros(0, np.uint64)
        np.savez(fr + ".tmp.npz", t=t, p=p, q=q); os.replace(fr + ".tmp.npz", fr)
        if rg % 8 == 0: log(f"  pass 2: range {rg}: {len(t):,} classes")
    parts_t.append(fr)
# final assembly straight into memory-mapped files (the table is ~2e9 classes: 19 GB of labels, 38 GB of witnesses)
sizes = [len(np.load(f, mmap_mode="r")["t"]) if False else int(np.load(f)["t"].shape[0]) for f in parts_t]; N = sum(sizes); log(f"assembling {N:,} classes")
mmd = os.path.join(D, "mm"); os.makedirs(mmd, exist_ok=True)
T_ = np.lib.format.open_memmap(os.path.join(mmd, name + "_tab.tmp.npy"), mode="w+", dtype=np.uint64, shape=(N,))
P_ = np.lib.format.open_memmap(os.path.join(mmd, name + "_wp.tmp.npy"), mode="w+", dtype=np.uint64, shape=(N,))
Q_ = np.lib.format.open_memmap(os.path.join(mmd, name + "_wq.tmp.npy"), mode="w+", dtype=np.uint64, shape=(N,))
cnt = np.zeros(32, np.int64); a = 0
for f, n in zip(parts_t, sizes):
    z = np.load(f); T_[a:a + n] = z["t"]; P_[a:a + n] = z["p"]; Q_[a:a + n] = z["q"]; cnt += np.bincount((z["t"] & F.CM).astype(np.int64), minlength=32); a += n
T_.flush(); P_.flush(); Q_.flush(); del T_, P_, Q_
for k in ("tab", "wp", "wq"): os.replace(os.path.join(mmd, f"{name}_{k}.tmp.npy"), os.path.join(mmd, f"{name}_{k}.npy"))
open(os.path.join(D, name + ".npy.mm"), "w").write("table assembled directly under data/mm/ (memmapped); no flat copy")
log(f"rung {r}: {N:,} classes to {2 * (b + 2) - 1} (exact) {'mirrored' if mir else ''}; per HTM " + ", ".join(f"{c}: {int(v):,}" for c, v in enumerate(cnt) if v))
