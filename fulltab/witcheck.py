"""Witness check of a sliced table: random entries decoded to move sequences; each must land in the entry's class at the
entry's cost.  argv: rung, table name, h [n=2000]."""
import sys, os, numpy as np, model as M, coset as CS, ladder as L
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import ftlib as F, table as T
r, name, h = sys.argv[1], sys.argv[2], int(sys.argv[3]); n = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
tab = T.SlicedTable(r, name, h, {"7": "F16"}[r]); rng = np.random.default_rng(7); idx = rng.integers(0, len(tab.tab), n); bad = 0; kinds = {}
for ti in idx:
    e = int(tab.tab[ti]); seq = tab.moves(int(ti)); st = M.apply_seq(M.State(), " ".join(seq))
    P = CS.pack_states([st]); lab = int((F.labels50m(F.maps50(P), tab.spT) if tab.mir else F.labels50(P, tab.spT))[0]) & ~F.CM; c = L.htm(" ".join(seq))   # mirrored tables store the mirror-canonical label
    w = int(tab.wp[ti]); kind = "ball" if w < (1 << 40) else "level b+1" if w < (1 << 41) else ("stored+run" if (w - (1 << 41)) >> 39 & 1 else "level b+2")
    kinds[kind] = kinds.get(kind, 0) + 1
    if lab != (e & ~F.CM) or c != (e & F.CM): bad += 1
print(f"{name}: {n} entries decoded, {bad} wrong (label or cost); witness kinds (p side): {kinds}", flush=True)
