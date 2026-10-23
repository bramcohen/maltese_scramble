"""Rungs 1+2 as a flat table: Z-coset labels (50-point) of every ball state up to h HTM with the cheapest cost and a witness
ball row.  argv: h.  Saves fulltab/data/full_rung12_b{h}.npy (label | cost, sorted) and full_rung12_b{h}_wit.npy (ball rows)."""
import sys, os, time, numpy as np, coset as CS, model as M
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import ftlib as F, table as T
h = int(sys.argv[1]); D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
T0 = time.time(); log = lambda s: print(f"{s} ({time.time() - T0:.0f}s)", flush=True)
ST = np.load(os.path.join(D, f"ball{h}.npy"), mmap_mode="r"); C = np.load(os.path.join(D, f"ball{h}_cost.npy")); spZ = CS.Spec(canon="Z")      # a stabiliser-chain spec: the kernel's label is then a right-coset label
lab = np.concatenate([F.labels50(np.ascontiguousarray(ST[c:c + 5_000_000]), spZ) & ~F.CM for c in range(0, len(ST), 5_000_000)])
log(f"ball {h}: {len(lab):,} states labelled")
key = lab | C.astype(np.uint64); o = np.argsort(key, kind="quicksort"); key = key[o]; first = np.r_[True, (key[1:] & ~F.CM) != (key[:-1] & ~F.CM)]
tab = key[first]; wit = o[first].astype(np.int64)                        # sorted by label; the first of each label is its cheapest (cost in the low bits)
np.save(os.path.join(D, f"full_rung12_b{h}.npy"), tab); np.save(os.path.join(D, f"full_rung12_b{h}_wit.npy"), wit)
cnt = np.bincount((tab & F.CM).astype(np.int64)); log(f"rung 12: {len(tab):,} Z-classes to {h}; per HTM " + ", ".join(f"{c}: {int(v):,}" for c, v in enumerate(cnt) if v))
