"""A rung's table beyond the join: base = the full table from the h-HTM ball (join with witnesses, so every class has
a representative), extended by chaining the pieces (single in-group runs and excursions up to L HTM) to maxcost.
argv: rung, h, L, maxcost [hbits].  Saves fulltab/data/full_rung{r}_chain{maxcost}.npy and the representatives."""
import sys, os, time, numpy as np, coset as CS, model as M, ladder as L_
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import ftlib as F
r, h, L, maxcost = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); hbits = int(sys.argv[5]) if len(sys.argv) > 5 else 28
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
T0 = time.time(); log = lambda s: print(f"{s} ({time.time() - T0:.0f}s)", flush=True)
isUD = np.zeros(18, np.int8); isUD[L_.UDE] = 1; isEQ = np.zeros(24, np.int8); isEQ[L_.EQG] = 1
Z = CS.Spec(pos={"r": (np.ones(18), isUD), "g": (np.ones(24), isEQ)}, fun={"c": (np.zeros(8, np.int8), np.eye(8, dtype=np.int8))})
spG, tgt = {"3": (Z, "Y"), "4": (CS.Spec(canon="Y"), "H1p"), "5": (CS.Spec(canon="H1p"), "H1"), "7": (CS.Spec(canon="A1"), "F16")}[r]
tgt = os.environ.get("CHAIN_TARGET", tgt); spT = CS.Spec(canon=tgt); suf = "" if tgt in ("Y", "H1p", "H1", "F16") else tgt.lower()    # e.g. CHAIN_TARGET=Y2 for rung 3
ST = np.load(os.path.join(D, f"ball{h}.npy"), mmap_mode="r"); C = np.load(os.path.join(D, f"ball{h}_cost.npy"))
glab = np.concatenate([spG.keys(np.ascontiguousarray(ST[c:c + 5_000_000])) for c in range(0, len(ST), 5_000_000)])
S, Cs, gst, nrep, order = F.group_order_reps(np.asarray(ST), C, glab, spT, return_order=True); del glab, ST
tab, ew, st = F.join50w(S, Cs, gst, nrep, spT, 2 * h, 30 if h >= 9 else 28)
base = F.pair_maps50(S, ew); bc = (tab & F.CM).astype(np.int8); del S, Cs
bwp = order[(ew >> np.uint64(32)).astype(np.int64)].astype(np.int64); bwq = order[(ew & np.uint64(0xffffffff)).astype(np.int64)].astype(np.int64); del ew, order
kid = F.labels50(CS.pack_states([M.State()]), spT)[0] & ~F.CM
idm = np.flatnonzero((tab & ~F.CM) == kid); bc[idm] = 0; base[idm] = F.maps50(CS.pack_states([M.State()]))[0]; bwp[idm] = 0; bwq[idm] = 0   # the target: cost 0, empty sequence
log(f"rung {r}: base {len(tab):,} classes to {2 * h} HTM ({st['seconds']:.0f}s join)")
pz = np.load(os.path.join(D, f"pieces_rung{r}_h{(L + 1) // 2}.npz")); keep = pz["cost"] <= L
pieces, pc = pz["maps"][keep], pz["cost"][keep]
log(f"pieces: {len(pieces):,} up to {L} HTM")
t, rows, R, RC, DER, (bwp, bwq) = F.chain(base, bc, pieces, pc, spT, maxcost, 2 * h - L, hbits=hbits, log=log, base_wit=(bwp, bwq))
np.save(os.path.join(D, f"full_rung{r}_chain{maxcost}{suf}.npy"), t)
np.savez(os.path.join(D, f"reps_rung{r}_chain{maxcost}{suf}.npz"), rows=rows, RC=RC, DER=DER, bwp=bwp, bwq=bwq, ball=h, pieces=f"pieces_rung{r}_h{(L + 1) // 2}.npz")
cnt = np.bincount((t & F.CM).astype(np.int64))
log(f"rung {r}: {len(t):,} classes to {maxcost}; per HTM " + ", ".join(f"{c}: {int(v):,}" for c, v in enumerate(cnt) if v))
