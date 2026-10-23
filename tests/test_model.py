"""Discrete model vs float oracle, generator identities, published algorithms."""
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import oracle as O
import model as M


def test_oracle_agreement(n=10000, maxlen=30, seed=12345):
    rng = random.Random(seed)
    for _ in range(n):
        seq = [rng.choice(M.GENERATORS) for _ in range(rng.randint(1, maxlen))]
        ost = O.apply(O.solved(), M.to_oracle_moves(seq))
        want = M.State(*M.from_oracle(ost))
        got = M.apply_seq(M.State(), seq)
        assert got == want, seq


def test_generator_identities():
    for f in M.FACES:
        assert M.element([f] * 8).state() == M.SOLVED
        assert not (M.element([f] * 4).state() == M.SOLVED)
        assert M.element([f + "'"]) == M.element([f] * 7)
        assert not (M.element([f + "'"]) == M.element([f] * 3))
        assert M.element([f, f + "'"]).state() == M.SOLVED
    # Contrary to the plan, R and L commute: L' = R^-1 x (a whole-cube rotation
    # about the same axis), so L is R times something that commutes with R.
    for a, b, rot in (("R", "L", "x"), ("U", "D", "y"), ("F", "B", "z")):
        assert M.element(f"{a} {b}") == M.element(f"{b} {a}")
        assert M.element(f"{a} {b}'") == M.element(rot)
        assert not (M.element(f"{a} {b}").state() == M.SOLVED)


def test_render_roundtrip(n=200, seed=7):
    rng = random.Random(seed)
    for _ in range(n):
        st = M.apply_seq(M.State(), [rng.choice(M.GENERATORS) for _ in range(40)])
        assert M.State(*M.from_oracle(M.to_oracle_state(st))) == st



def _where(e, orbit, name):
    """(slot now holding the piece that started at `name`, its spin change)."""
    names = {"g": M.GEAR_NAMES, "r": M.RING_NAMES, "c": M.CORNER_NAMES}[orbit]
    perm, ori = {"g": (e.gp, e.go), "r": (e.rp, e.ro), "c": (e.cp, e.co)}[orbit]
    i = list(perm).index(names.index(name))
    return names[i], int(ori[i])


def _check_gear_cycle(alg, cycle, flipped, qt=None):
    """zera's '3 cycles A -> B -> C while rotating X, Y ... 180 degrees'."""
    e = M.element(alg).state()
    if qt is not None:
        assert len(M.parse(alg)) == qt
    moved = set()
    for a, b in zip(cycle, cycle[1:] + cycle[:1]):
        slot, spin = _where(e, "g", a)
        assert slot == b, (alg, a, slot, b)
        assert spin == (2 if a in flipped else 0), (alg, a, spin)
        moved.add(a)
    for g in M.GEAR_NAMES:
        if g in moved:
            continue
        slot, spin = _where(e, "g", g)
        assert slot == g and spin == (2 if g in flipped else 0), (alg, g, slot, spin)


def test_published_algorithms():
    # XL = R4 D R4 D' R4 is 14 quarter turns; W = (R4 U R4 U')3 is 30.
    assert len(M.parse("XL")) == 14 and len(M.parse("W")) == 30
    # W swaps the F/B centers and the FR/BL edges and turns the U and D centers 90 degrees.
    w = M.element("W").state()
    assert _where(w, "r", "F")[0] == "B" and _where(w, "r", "B")[0] == "F"
    assert _where(w, "r", "FR")[0] == "BL" and _where(w, "r", "BL")[0] == "FR"
    assert _where(w, "r", "U") in (("U", 1), ("U", 3)) and _where(w, "r", "D") in (("D", 1), ("D", 3))
    assert all(_where(w, "c", c) == (c, 0) for c in M.CORNER_NAMES)
    assert all(_where(w, "g", g) == (g, 0) for g in M.GEAR_NAMES)
    others = [n for n in M.RING_NAMES if n not in ("F", "B", "FR", "BL", "U", "D")]
    assert all(_where(w, "r", n) == (n, 0) for n in others)
    # (U L'2 W L2 U' B4)2 (80 qt) cycles the centers L -> F -> R.
    assert len(M.parse("(U L'2 W L2 U' B4)2")) == 80
    c = M.element("(U L'2 W L2 U' B4)2").state()
    assert [_where(c, "r", n)[0] for n in "LFR"] == ["F", "R", "L"]
    # zera's manual v3, algorithms 9-12 (18 tokens counting XL as 5)
    _check_gear_cycle("B4 R3 B2 R' XL R B'2 R'3 B4 XL", ["BU", "DF", "DL"], {"FD", "DL", "LD", "BU"})
    _check_gear_cycle("D L' D2 R' XL R D'2 L D' XL", ["BU", "DL", "DF"], {"FD", "DF", "LD", "BU"})
    _check_gear_cycle("D R' U2 R' XL R U'2 R D' XL", ["FU", "DR", "DB"], {"DB", "BD", "RD", "FU"})
    _check_gear_cycle("D' R' U2 R' XL R U'2 R D XL", ["FU", "DL", "DF"], {"DF", "FD", "LD", "FU"})
    # 13-16 and the 26/30-move ones
    _check_gear_cycle("A XL A' XL", ["RF", "DF", "DL"], {"RF", "DL"})
    _check_gear_cycle("A XR A' XR", ["RF", "DF", "DR"], {"RF", "DR"})
    _check_gear_cycle("A' XL A XL", ["RF", "DF", "DL"], set())
    _check_gear_cycle("C' XB C XB", ["LF", "DF", "DB"], set())
    for alg, flips in (("A' XL A A XL A'", {"RF", "DL"}), ("A' XR A A XR A'", {"RF", "DR"}),
                       ("C' XL C C XL C'", {"LF", "DL"}), ("C' XR C C XR C'", {"LF", "DR"})):
        e = M.element(alg).state()
        assert all(_where(e, "g", g) == (g, 2 if g in flips else 0) for g in M.GEAR_NAMES), alg
    # gear-parity algorithms 1-8 turn two gears by 90 degrees
    for alg, pair in (("R' U3 F' XL F U'3 R XL", ("BD", "RD")), ("R3 D3 L XL L' D'3 R'3 XL", ("DB", "DR")),
                      ("L' U3 B' XL B U'3 L XL", ("FD", "LD")), ("F D L XL L' D' F' XL", ("DF", "DL")),
                      ("L U'3 F XR F' U3 L' XR", ("BD", "LD")), ("L'3 D'3 R' XR R D3 L3 XR", ("DB", "DL")),
                      ("R U'3 B XR B' U3 R' XR", ("FD", "RD")), ("R D F XR F' D' R' XR", ("DF", "DR"))):
        e = M.element(alg).state()
        odd = {g for g in M.GEAR_NAMES if _where(e, "g", g)[1] % 2}
        assert odd == set(pair), (alg, odd)
        assert all(_where(e, "g", g)[0] == g for g in pair)
    # edge orientation algorithms (step 4 and 8)
    e = M.element("U' R F' W W F R' U").state()
    assert _where(e, "r", "DL") == ("DL", 2)
    e = M.element("R W W R'").state()
    assert _where(e, "r", "UF") == ("UF", 2) and _where(e, "r", "DB") == ("DB", 2)
    e = M.element("R' U F' W F U' R").state()
    assert _where(e, "r", "DL")[0] == "DL" and _where(e, "r", "DL")[1] % 2 == 1



def test_sampler_and_check():
    """Uniform samples satisfy every invariant; random walks normalise into <U,R,F>;
    the visible check accepts reachable states and names a broken invariant."""
    import solve as SV
    rng = np.random.default_rng(0)
    for _ in range(500):
        st = M.random_state(rng)
        assert M.check(st) == []
        assert SV.check_visible(*SV.parse_state(SV.show(st))) == []
    r = random.Random(1)
    st = M.State()
    for _ in range(500):
        st = st.apply(M.MOVES[r.choice(M.GENERATORS)])
        assert M.check(M.normalize_rotation(st)) == []
    cp, co, rp, ro, gc, go = SV.parse_state(SV.show(st))
    go[0] = (go[0] + 1) % 4
    assert any("gear" in b for b in SV.check_visible(cp, co, rp, ro, gc, go))


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time(); fn(); print(f"{name}: ok ({time.time() - t:.1f}s)")
