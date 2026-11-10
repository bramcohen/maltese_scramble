"""Relabelling sets for rungs 7 and 8 by intersecting downwards from S ∩ H1 (16,384 elements; S ∩ A1 and S ∩ F16 lie inside it),
instead of searching the 2e11-element S for each.  Saves T_reps_7.npy and T_reps_8.npy."""
import sys, time, pickle, numpy as np, model as M, coset as CS
sys.path.insert(0, "."); import groupfacts as GF
from sympy.combinatorics import Permutation, PermutationGroup
T0 = time.time(); log = lambda s: print(f"{s} ({time.time() - T0:.0f}s)", flush=True)
S = PermutationGroup([Permutation(p) for p in pickle.load(open("tables/invisible_gens.pkl", "rb"))])
grp = lambda n: PermutationGroup([GF.perm_of(M.element(M.parse(w))) for w in pickle.load(open(f"tables/bsgs_{n}.pkl", "rb"))["gens"]])
def state_of(perm):
    a = perm.array_form; st = M.State()
    for base, k, src, d in ((GF.C0, 3, st.cp, st.co), (GF.R0, 4, st.rp, st.ro), (GF.G0, 4, st.gp, st.go)):
        for i in range(len(src)):
            v = a[base + i * k] - base; src[v // k] = i; d[v // k] = v % k
    return st
K6 = S.subgroup_search(grp("H1").contains); log(f"|S ∩ H1| = {K6.order()}")
K7 = K6.subgroup_search(grp("A1").contains); log(f"|S ∩ A1| = {K7.order()}")
K8 = K7.subgroup_search(grp("F16").contains); log(f"|S ∩ F16| = {K8.order()}")
E = PermutationGroup([Permutation(list(range(GF.NPTS)))])
for r, (K, H) in {"7": (K7, K8), "8": (K8, E)}.items():
    reps = K.coset_transversal(H); sts = [state_of(p) for p in reps]
    assert all(GF.perm_of(s.as_element()) == p for s, p in zip(sts, reps))
    rows = CS.pack_states(sts)
    spec = CS.Spec(canon="F16" if r == "7" else {"levels": []})
    assert len(np.unique(spec.keys(rows))) == len(rows), "right-coset transversal coverage"
    np.save(f"fulltab/data/T_reps_{r}.npy", rows)
    log(f"rung {r}: relabellings {len(reps)} ({np.log2(len(reps)):.1f} bits); legal {sum(1 for s in sts if not M.check(s))}; saved")
