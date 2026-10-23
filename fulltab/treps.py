"""Relabelling sets for every rung: left transversal of S ∩ T in S ∩ G (G starting group, T target), inverted, as packed
states -> fulltab/data/T_reps_{rung}.npy.  Rungs: 12 (G = everything, T = Z) and 6 (H1 -> A1); rungs 7 and 8: treps2.py, rung 4: treps_y2.py.  Prints the counts (2^invisible bits)."""
import sys, time, pickle, numpy as np, model as M, coset as CS, ladder as L
sys.path.insert(0, "."); import groupfacts as GF
from sympy.combinatorics import Permutation, PermutationGroup
T0 = time.time(); log = lambda s: print(f"{s} ({time.time() - T0:.0f}s)", flush=True)
S = PermutationGroup([Permutation(p) for p in pickle.load(open("tables/invisible_gens.pkl", "rb"))])
grp = lambda n: PermutationGroup([GF.perm_of(M.element(M.parse(w))) for w in pickle.load(open(f"tables/bsgs_{n}.pkl", "rb"))["gens"]])
isUD = np.zeros(18, bool); isUD[L.UDE] = True; isEQ = np.zeros(24, bool); isEQ[L.EQG] = True
def inZ(g):     # Z: UD edges in UD slots, equatorial gears in equatorial slots, corners in place (orientation free)
    a = g.array_form
    return all(isEQ[(a[GF.G0 + i * 4] - GF.G0) // 4] == isEQ[i] for i in range(24)) and all(isUD[(a[GF.R0 + i * 4] - GF.R0) // 4] == isUD[i] for i in range(18)) \
        and all((a[GF.C0 + i * 3] - GF.C0) // 3 == i for i in range(8))
def state_of(perm):
    a = perm.array_form; st = M.State()
    for base, k, src, d in ((GF.C0, 3, st.cp, st.co), (GF.R0, 4, st.rp, st.ro), (GF.G0, 4, st.gp, st.go)):
        for i in range(len(src)):
            v = a[base + i * k] - base; src[v // k] = i; d[v // k] = v % k
    return st
tgt = {"12": (None, inZ), "6": ("H1", "A1")}          # rungs 7 and 8: fulltab/treps2.py (intersecting downwards from S ∩ H1 is far faster)
for r, (gname, tname) in tgt.items():
    K = S if gname is None else S.subgroup_search(grp(gname).contains)
    if tname is None: H = PermutationGroup([Permutation(list(range(GF.NPTS)))])
    elif callable(tname): H = K.subgroup_search(tname)
    else: H = K.subgroup_search(grp(tname).contains)
    reps = [g**-1 for g in K.coset_transversal(H)]; sts = [state_of(p) for p in reps]
    assert all(GF.perm_of(s.as_element()) == p for s, p in zip(sts[:20], reps[:20]))
    P = CS.pack_states(sts); np.save(f"fulltab/data/T_reps_{r}.npy", P)
    log(f"rung {r}: |S ∩ G| = {K.order():.4e}, |S ∩ T| = {H.order():.4e}, relabellings {len(reps):,} ({np.log2(len(reps)):.1f} bits); legal {sum(1 for s in sts if not M.check(s))}; saved T_reps_{r}.npy")
