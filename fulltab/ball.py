"""Every state within h HTM of solved over all moves (every single-face run), deduplicated, in cost order.
Saves fulltab/data/ball{h}.npy (states, 100 bytes each) and ball{h}_cost.npy.  argv: h."""
import sys, os, time, numpy as np, model as M, coset as CS, ladder as L
h = int(sys.argv[1]); D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
runs = [f + a for f in "URF" for a in ("", "2", "3", "4", "'", "'2", "'3")]
cost = [L.word_htm(r) for r in runs]; E = [CS.pack_elem(M.element(M.parse(r))) for r in runs]
full = CS.Spec(canon={"levels": []})
T = time.time(); log = lambda s: print(f"{s} ({time.time() - T:.0f}s)", flush=True)
cap = {6: 700_000, 7: 6_000_000, 8: 50_000_000, 9: 470_000_000}[h]
ST = np.lib.format.open_memmap(os.path.join(D, f"ball{h}_build.npy"), mode="w+", dtype=np.int8, shape=(cap, 100))
CST = np.zeros(cap, np.int8); PAR = np.full(cap, -1, np.int32); PRUN = np.full(cap, -1, np.int8)
st0 = CS.pack_states([M.State()]); ST[0] = st0[0]; seen = CS.KeySet(1 << (27 if h < 9 else 30)); seen.insert(full.keys(st0))
lvl = {0: (0, 1)}; n_tot = 1
for n in range(1, h + 1):
    start = n_tot
    for ri in range(21):
        if n - cost[ri] not in lvl:
            continue
        a, k = lvl[n - cost[ri]]
        for c0 in range(a, a + k, 5_000_000):
            c1 = min(c0 + 5_000_000, a + k)
            s, kk = full.apply(np.ascontiguousarray(ST[c0:c1]), E[ri]); new = np.where(seen.insert(kk))[0]
            if len(new):
                assert n_tot + len(new) <= cap, "ball capacity"
                ST[n_tot:n_tot + len(new)] = s[new]; CST[n_tot:n_tot + len(new)] = n
                PAR[n_tot:n_tot + len(new)] = c0 + new; PRUN[n_tot:n_tot + len(new)] = ri; n_tot += len(new)
    lvl[n] = (start, n_tot - start)
    log(f"level {n}: {n_tot - start:,} states")
del seen
ST.flush(); del ST
# trim to size
src = np.load(os.path.join(D, f"ball{h}_build.npy"), mmap_mode="r")
out = np.lib.format.open_memmap(os.path.join(D, f"ball{h}_t.npy"), mode="w+", dtype=np.int8, shape=(n_tot, 100))
for c0 in range(0, n_tot, 20_000_000): out[c0:c0 + 20_000_000] = src[c0:min(c0 + 20_000_000, n_tot)]
out.flush(); del out, src
os.replace(os.path.join(D, f"ball{h}_t.npy"), os.path.join(D, f"ball{h}.npy")); os.remove(os.path.join(D, f"ball{h}_build.npy"))
np.save(os.path.join(D, f"ball{h}_cost.npy"), CST[:n_tot]); np.savez(os.path.join(D, f"ball{h}_parents.npz"), par=PAR[:n_tot], run=PRUN[:n_tot])
log(f"ball {h}: {n_tot:,} states")
