"""Rung 4's renaming set when rung 3 targets Y2: right transversal of S ∩ H1' in S ∩ Y2 (9,216 expected), as
packed states -> fulltab/data/T_reps_4_y2.npy."""
import sys, time, pickle, numpy as np, model as M, coset as CS
sys.path.insert(0, "."); import groupfacts as GF
from sympy.combinatorics import Permutation, PermutationGroup
T0 = time.time(); log = lambda s: print(f"{s} ({time.time() - T0:.0f}s)", flush=True)
S = PermutationGroup([Permutation(p) for p in pickle.load(open("tables/invisible_gens.pkl", "rb"))])
grp = lambda n: PermutationGroup([GF.perm_of(M.element(M.parse(w))) for w in pickle.load(open(f"tables/bsgs_{n}.pkl", "rb"))["gens"]])
Y2 = PermutationGroup([Permutation(p) for p in pickle.load(open("tables/bsgs_Y2.pkl", "rb"))["perm_gens"]]); H = grp("H1p")
K = S.subgroup_search(Y2.contains); KH = K.subgroup_search(H.contains); log(f"|S ∩ Y2| = {K.order():.4e}, |S ∩ H1'| = {KH.order():.4e}, cosets {K.order() // KH.order():,}")
def state_of(perm):
    a = perm.array_form; st = M.State()
    for base, k, src, d in ((GF.C0, 3, st.cp, st.co), (GF.R0, 4, st.rp, st.ro), (GF.G0, 4, st.gp, st.go)):
        for i in range(len(src)):
            v = a[base + i * k] - base; src[v // k] = i; d[v // k] = v % k
    return st
reps = K.coset_transversal(KH); sts = [state_of(p) for p in reps]
assert all(GF.perm_of(s.as_element()) == p for s, p in zip(sts[:30], reps[:30]))
rows = CS.pack_states(sts)
assert len(np.unique(CS.Spec(canon="H1p").keys(rows))) == len(rows), "right-coset transversal coverage"
np.save("fulltab/data/T_reps_4_y2.npy", rows); log(f"saved {len(sts):,} renamings (legal {sum(1 for s in sts if not M.check(s))})")
