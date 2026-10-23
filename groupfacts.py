"""Group facts: orders, invariants, subgroup structure.

Run:  .venv/bin/python groupfacts.py            (prints everything the report cites)

Permutation representation: points (slot, ori) with
    corners  0..23   (slot*3 + ori)
    ring    24..95   (24 + slot*4 + ori)
    gears   96..191  (96 + slot*4 + ori)
This is the full model (center spin and gear identity included).  Visible-state
counts are derived from it by dividing by the stabilizer of the visible solved
state, computed in visible_stabilizer_order().
"""
import math
import sys
import time

import numpy as np
from sympy.combinatorics import Permutation, PermutationGroup

import model as M

NPTS = 24 + 72 + 96
C0, R0, G0 = 0, 24, 96


def perm_of(e):
    """Permutation of the 192 points induced by group element e (a model.Move)."""
    img = [0] * NPTS
    for orbit, base, k, src, d in ((0, C0, 3, e.src.cp, e.dco), (1, R0, 4, e.src.rp, e.dro),
                                   (2, G0, 4, e.src.gp, e.dgo)):
        for j in range(len(src)):
            i = int(src[j])
            for o in range(k):
                img[base + i * k + o] = base + j * k + (o + int(d[j])) % k
    return Permutation(img)


def group(names_or_elems):
    gens = [perm_of(M.element(x) if isinstance(x, str) else x) for x in names_or_elems]
    return PermutationGroup(gens)


def restrict(G, lo, hi):
    """Image of G acting on points lo..hi-1."""
    gens = []
    for g in G.generators:
        a = g.array_form
        gens.append(Permutation([a[p] - lo for p in range(lo, hi)]))
    return PermutationGroup(gens)


def fmt(n):
    return f"{n} (~{n:.3e})" if n > 10 ** 6 else str(n)


# ---------------------------------------------------------------- invariants


def parity(p):
    p = list(p)
    seen, par = [False] * len(p), 0
    for i in range(len(p)):
        if not seen[i]:
            j, ln = i, 0
            while not seen[j]:
                seen[j] = True
                j = p[j]
                ln += 1
            par ^= (ln - 1) & 1
    return par


def invariants(st):
    """The invariant functionals found below, evaluated on a state (full model)."""
    edge_slot = np.array([not M.IS_CENTER[i] for i in range(M.NR)])
    edge_piece_ori = [int(st.ro[i]) for i in range(M.NR) if not M.IS_CENTER[st.rp[i]]]
    return {
        "corner twist sum mod 3": int(st.co.sum()) % 3,
        "ring permutation parity": parity(st.rp),
        "corner parity + gear parity": (parity(st.cp) + parity(st.gp)) % 2,
        "gear spin sum mod 4": int(st.go.sum()) % 4,
        "ring spin sum (all 18) mod 4": int(st.ro.sum()) % 4,
        "edge-piece spin sum mod 2": sum(edge_piece_ori) % 2,
    }


def main():
    t0 = time.time()
    out = []

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        out.append(s)
        sys.stdout.flush()

    G12 = group(M.GENERATORS)
    o12 = G12.order()
    say("|<12 quarter turns>| (fixed frame, full model) =", fmt(o12))
    G6 = group(["U", "U'", "R", "R'", "F", "F'"])
    o6 = G6.order()
    say("|<U,R,F quarter turns>| (DBL corner fixed) =", fmt(o6), " ratio", o12 // o6, o12 % o6)
    ROT = group(["x", "y", "z"])
    say("|whole-cube rotations| =", ROT.order(), " contained in G12:", all(G12.contains(g) for g in ROT.generators))
    # projections
    for nm, lo, hi in (("corners", C0, R0), ("ring", R0, G0), ("gears", G0, NPTS)):
        say(f"  projection of <U,R,F> on {nm}: ", fmt(restrict(G6, lo, hi).order()))
    # half-turn groups
    H12 = group([f + "2" for f in M.FACES])
    H6 = group(["U2", "R2", "F2"])
    say("|<R2,L2,U2,D2,F2,B2>| =", fmt(H12.order()))
    say("|<U2,R2,F2>| =", fmt(H6.order()))
    say("  H12 contains x2,y2,z2:", [H12.contains(perm_of(M.element(r + " " + r))) for r in "xyz"],
        " contains x:", H12.contains(perm_of(M.element("x"))))
    for nm, lo, hi in (("corners", C0, R0), ("ring", R0, G0), ("gears", G0, NPTS)):
        say(f"  projection of <U2,R2,F2> on {nm}: ", fmt(restrict(H6, lo, hi).order()))
    # visible counts (checked against sympy below)
    corners = 3674160
    ring_full = math.factorial(18) // 2 * 4 ** 18
    gear_full = math.factorial(24) * 4 ** 23
    say("closed form corners*ring*gears/4 == |<U,R,F>|:", corners * ring_full * gear_full // 4 == o6)
    stab = (4 ** 6 // 2) * (24 ** 6 // 2)
    vis = o6 // stab
    say("visible states =", fmt(vis), " = corners", corners, "x ring", fmt(math.factorial(18) // 2 * 4 ** 12),
        "x gears", fmt(math.factorial(24) // 24 ** 6 * 4 ** 23), ":",
        vis == corners * (math.factorial(18) // 2 * 4 ** 12) * (math.factorial(24) // 24 ** 6 * 4 ** 23))
    for nm, gens in (("<U,R2,F2>", ["U", "R2", "F2"]), ("<U,R2>", ["U", "R2"]), ("<U2,R2,F2>", ["U2", "R2", "F2"])):
        n, k = visible_order(gens)
        say(f"visible states of {nm}: {fmt(n // k)}  (|H|={fmt(n)}, |H ∩ visible stabilizer|={k})")
    # invariants hold on random states
    import random
    rng = random.Random(1)
    st = M.State()
    inv0 = invariants(st)
    for _ in range(20000):
        st = st.apply(M.MOVES[rng.choice(M.GENERATORS)])
        inv = invariants(st)
        for k in ("corner twist sum mod 3", "ring permutation parity", "corner parity + gear parity",
                  "gear spin sum mod 4"):
            assert inv[k] == inv0[k], k
        assert (inv["ring spin sum (all 18) mod 4"] + parity(st.cp)) % 2 == 0
    say("invariants verified on a 20000-move random walk")
    # gear siblings under half turns
    gUF, gFU = M.GEAR_NAMES.index("UF"), M.GEAR_NAMES.index("FU")
    orb = H12.orbit(G0 + gUF * 4)
    say("half-turn orbit of gear UF reaches sibling slot FU:", any((p - G0) // 4 == gFU for p in orb))
    say(f"[{time.time() - t0:.1f}s]")
    return out


def visible_order(gens):
    """(|H|, |H ∩ S|) where S is the stabilizer of the visible solved state."""
    G = group(gens)
    n = G.order()
    fix = list(range(0, 24)) + [R0 + s * 4 + o for s in range(6, 18) for o in range(4)]
    P = G.pointwise_stabilizer(fix)
    cen, gr = restrict(P, R0, R0 + 24), restrict(P, G0, NPTS)
    assert cen.order() * gr.order() == P.order()  # direct product (checked)
    ck = sum(1 for g in cen.generate() if all(g.array_form[s * 4] // 4 == s for s in range(6)))
    gk = sum(1 for g in gr.generate()
             if all(g.array_form[s * 4] % 4 == 0 and M.GEAR_COLOR[g.array_form[s * 4] // 4] == M.GEAR_COLOR[s]
                    for s in range(24)))
    return n, ck * gk


if __name__ == "__main__":
    main()
