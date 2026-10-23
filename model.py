"""Discrete model of the Maltese Gear Cube.

Three orbits, each stored slot-indexed:

    corners : perm[8]  (piece id in each slot), ori mod 3
    ring    : perm[18] (6 centers then 12 edges), ori mod 4 (spin about the radial axis)
    gears   : perm[24], ori mod 4

A move is (src, dori) per orbit: new.perm[i] = old.perm[src[i]],
new.ori[i] = old.ori[src[i]] + dori[i].  The tables are derived from the float
oracle (oracle.py, a port of kingdeim/MalteseGearCube) and checked against it in
tests/test_model.py.

What a person can see (VISIBLE state, see groupfacts.py):
  * corners: piece and twist;
  * ring pieces: piece, and spin for edge pieces only (a center's sticker is a
    plain square, so its spin is invisible);
  * gears: colour (= home face) and spin; the four gears of one face look alike.
"""
import math
import re
import numpy as np

import oracle as O

FACES = "UDFBRL"
CORNER_NAMES = [s["name"] for s in O.CORNER_SLOTS]   # UFR UFL UBR UBL DFR DFL DBR DBL
RING_NAMES = [s["name"] for s in O.EC_SLOTS]         # U D F B R L UF UB UR UL DF DB DR DL FR FL BR BL
GEAR_NAMES = [s["name"] for s in O.GEAR_SLOTS]       # UF UB UR UL DF ... (face, toward)
NC, NR, NG = 8, 18, 24
NCENTER = 6
IS_CENTER = np.array([i < NCENTER for i in range(NR)])
GEAR_COLOR = np.array([FACES.index(n[0]) for n in GEAR_NAMES], dtype=np.int8)

# -------------------------------------------------------------- orientation frames


def _ring_frame(slot_name):
    """(radial, tangent) unit vectors for a ring slot."""
    s = next(s for s in O.EC_SLOTS if s["name"] == slot_name)
    r = s["dir"]
    if s["type"] == "center":
        t = {"U": [1, 0, 0], "D": [1, 0, 0], "F": [1, 0, 0], "B": [1, 0, 0],
             "R": [0, 0, -1], "L": [0, 0, 1]}[slot_name]
    else:
        a, b = O.FACES[slot_name[0]], O.FACES[slot_name[1]]
        t = O.cross(a, b)  # the axis of the ring this edge slot lies on
    return r, t


RING_FRAMES = {n: _ring_frame(n) for n in RING_NAMES}


def _quarter_angle(r, t_from, t_to):
    """k such that t_from = rot(r, 90k) t_to (counterclockwise about r)."""
    a = math.atan2(O.dot(O.cross(t_to, t_from), r), O.dot(t_to, t_from))
    k = a / (math.pi / 2)
    assert abs(k - round(k)) < 1e-6, (r, t_from, t_to, k)
    return round(k) % 4


def ring_ori(piece, slot_name):
    """Spin of an oracle ring piece sitting in slot_name."""
    _, tp = RING_FRAMES[piece["id"]]
    r, ts = RING_FRAMES[slot_name]
    assert O.near(O.mulMV(piece["R"], RING_FRAMES[piece["id"]][0]), r)
    return _quarter_angle(r, O.mulMV(piece["R"], tp), ts)


def from_oracle(st):
    """Convert an oracle state to (cp, co, rp, ro, gp, go) numpy arrays."""
    corner, ec, gear = O.index(st)
    cp = np.array([CORNER_NAMES.index(corner[n]["id"]) for n in CORNER_NAMES], dtype=np.int8)
    co = np.array([O.corner_twist(corner[n]) for n in CORNER_NAMES], dtype=np.int8)
    rp = np.array([RING_NAMES.index(ec[n]["id"]) for n in RING_NAMES], dtype=np.int8)
    ro = np.array([ring_ori(ec[n], n) for n in RING_NAMES], dtype=np.int8)
    gp = np.array([GEAR_NAMES.index(gear[n]["id"]) for n in GEAR_NAMES], dtype=np.int8)
    go = np.array([O.gear_quarter(gear[n]) for n in GEAR_NAMES], dtype=np.int8)
    return cp, co, rp, ro, gp, go

# -------------------------------------------------------------- state


class State:
    __slots__ = ("cp", "co", "rp", "ro", "gp", "go")

    def __init__(self, cp=None, co=None, rp=None, ro=None, gp=None, go=None):
        self.cp = np.arange(NC, dtype=np.int8) if cp is None else np.asarray(cp, dtype=np.int8)
        self.co = np.zeros(NC, np.int8) if co is None else np.asarray(co, dtype=np.int8)
        self.rp = np.arange(NR, dtype=np.int8) if rp is None else np.asarray(rp, dtype=np.int8)
        self.ro = np.zeros(NR, np.int8) if ro is None else np.asarray(ro, dtype=np.int8)
        self.gp = np.arange(NG, dtype=np.int8) if gp is None else np.asarray(gp, dtype=np.int8)
        self.go = np.zeros(NG, np.int8) if go is None else np.asarray(go, dtype=np.int8)

    def copy(self):
        return State(self.cp.copy(), self.co.copy(), self.rp.copy(), self.ro.copy(),
                     self.gp.copy(), self.go.copy())

    def arrays(self):
        return self.cp, self.co, self.rp, self.ro, self.gp, self.go

    def __eq__(self, other):
        return all(np.array_equal(a, b) for a, b in zip(self.arrays(), other.arrays()))

    def __hash__(self):
        return hash(b"".join(a.tobytes() for a in self.arrays()))

    def key(self):
        return b"".join(a.tobytes() for a in self.arrays())

    # composition: (self * m) = apply move/element m after self
    def apply(self, m):
        s = m.src
        return State(self.cp[s.cp], (self.co[s.cp] + m.dco) % 3,
                     self.rp[s.rp], (self.ro[s.rp] + m.dro) % 4,
                     self.gp[s.gp], (self.go[s.gp] + m.dgo) % 4)

    def visible(self):
        """Canonical key of what a person sees."""
        ro = np.where(IS_CENTER[self.rp], 0, self.ro).astype(np.int8)
        return b"".join(a.tobytes() for a in (self.cp, self.co, self.rp, ro,
                                              GEAR_COLOR[self.gp], self.go))

    def is_visibly_solved(self):
        return self.visible() == SOLVED_VISIBLE

    def as_element(self):
        """The group element taking solved to this state (as a Move)."""
        return Move(Src(self.cp, self.rp, self.gp), self.co, self.ro, self.go)


class Src:
    __slots__ = ("cp", "rp", "gp")

    def __init__(self, cp, rp, gp):
        self.cp, self.rp, self.gp = (np.asarray(x, dtype=np.int8) for x in (cp, rp, gp))


class Move:
    """A group element in (src, delta) form; composing Moves is the group law."""
    __slots__ = ("src", "dco", "dro", "dgo", "name")

    def __init__(self, src, dco, dro, dgo, name=""):
        self.src = src
        self.dco, self.dro, self.dgo = (np.asarray(x, dtype=np.int8) for x in (dco, dro, dgo))
        self.name = name

    def __mul__(self, other):
        """self then other."""
        st = State(self.src.cp, self.dco, self.src.rp, self.dro, self.src.gp, self.dgo).apply(other)
        return st.as_element()

    def inverse(self):
        inv = lambda p: np.argsort(p).astype(np.int8)
        cp, rp, gp = inv(self.src.cp), inv(self.src.rp), inv(self.src.gp)
        return Move(Src(cp, rp, gp), (-self.dco[cp]) % 3, (-self.dro[rp]) % 4, (-self.dgo[gp]) % 4)

    def state(self):
        return State().apply(self)

    def __eq__(self, other):
        return self.state() == other.state()


SOLVED = State()
SOLVED_VISIBLE = SOLVED.visible()


def _move_from_oracle(seq):
    st = O.apply(O.solved(), seq)
    cp, co, rp, ro, gp, go = from_oracle(st)
    return Move(Src(cp, rp, gp), co, ro, go)


# The 12 quarter-turn generators and the 3 whole-cube rotations.
MOVES = {}
for f in FACES:
    MOVES[f] = _move_from_oracle([(f, False)]); MOVES[f].name = f
    MOVES[f + "'"] = _move_from_oracle([(f, True)]); MOVES[f + "'"].name = f + "'"
ROTATIONS = {}
for r in "xyz":
    ROTATIONS[r] = _move_from_oracle([(r, False)])
    ROTATIONS[r + "'"] = _move_from_oracle([(r, True)])
GENERATORS = [f + p for f in FACES for p in ("", "'")]
IDENTITY = State().as_element()

# -------------------------------------------------------------- notation

DEFAULT_MACROS = {
    "A": "R U R' U' R' F R F'",
    "A'": "F R' F' R U R U' R'",
    "C": "L' U' L U L F' L' F",
    "C'": "F' L F L' U' L' U L",
    "XL": "R4 D R4 D' R4",
    "XR": "R4 D' R4 D R4",
    "XB": "R4 U2 R4 U'2 D R4 U2 R4 U'2 D'",
    "W": "(R4 U R4 U')3",
    "J": "R U R U R U' R' U' R' U'",
    "J'": "U R U R U R' U' R' U' R'",
    "K": "L' U' L' U' L' U L U L U",
    "K'": "U' L' U' L' U' L U L U L",
}


def parse(text, macros=DEFAULT_MACROS):
    """Parse zera/kingdeim notation into a list of quarter turns ('R' or "R'").

    Supports R R' R3 R'3 R3' (modifiers in any order), groups '(...)n' and "(...)'",
    macros (longest name first; X' uses an explicit X' definition if present),
    and whole-cube rotations x y z.
    """
    s = re.sub(r"[’‘`′´]", "'", text).replace("*", "")
    names = sorted({n.rstrip("'") for n in macros}, key=len, reverse=True)
    i = 0

    def mods():
        nonlocal i
        prime, num = False, ""
        while i < len(s) and (s[i] == "'" or s[i].isdigit()):
            if s[i] == "'":
                prime = not prime
            else:
                num += s[i]
            i += 1
        return prime, int(num) if num else 1

    def invert(qs):
        return [q[:-1] if q.endswith("'") else q + "'" for q in reversed(qs)]

    def seq(depth, stack):
        nonlocal i
        out = []
        while i < len(s):
            ch = s[i]
            if ch in " \t\n,;.":
                i += 1
                continue
            if ch == "(":
                i += 1
                inner = seq(depth + 1, stack)
                if i >= len(s) or s[i] != ")":
                    raise ValueError("unbalanced parenthesis")
                i += 1
                prime, count = mods()
                body = invert(inner) if prime else inner
                out += body * count
                continue
            if ch == ")":
                if depth == 0:
                    raise ValueError(f"unexpected ')' at {i + 1}")
                return out
            name = next((n for n in names if s.startswith(n, i)), None)
            if name:
                i += len(name)
                prime, count = mods()
                if name in stack:
                    raise ValueError(f"recursive macro {name}")
                if prime and name + "'" in macros:
                    base = _expand(name + "'", stack)
                else:
                    base = _expand(name, stack)
                    if prime:
                        base = invert(base)
                out += base * count
                continue
            if ch in "UDFBRLxyz":
                i += 1
                prime, count = mods()
                out += [ch + ("'" if prime else "")] * count
                continue
            raise ValueError(f"unexpected character {ch!r} at {i + 1}")
        if depth > 0:
            raise ValueError("unbalanced parenthesis")
        return out

    def _expand(name, stack):
        nonlocal s, i
        saved = s, i
        s, i = re.sub(r"[’‘`′´]", "'", macros[name]).replace("*", ""), 0
        try:
            return seq(0, stack + [name])
        finally:
            s, i = saved

    return seq(0, [])


def element(moves):
    """Group element of a sequence of quarter-turn tokens (or a notation string)."""
    if isinstance(moves, str):
        moves = parse(moves)
    st = State()
    for q in moves:
        st = st.apply(MOVES[q] if q in MOVES else ROTATIONS[q])
    return st.as_element()


def apply_seq(state, moves):
    if isinstance(moves, str):
        moves = parse(moves)
    for q in moves:
        state = state.apply(MOVES[q] if q in MOVES else ROTATIONS[q])
    return state


def compress(moves):
    """Group consecutive identical quarter turns: R R R -> R3, R' R' -> R'2 (zera notation)."""
    out = []
    for q in moves:
        if out and out[-1][0] == q:
            out[-1][1] += 1
        else:
            out.append([q, 1])
    res = []
    for q, n in out:
        if n == 1:
            res.append(q)
        elif q.endswith("'"):
            res.append(f"{q[0]}'{n}")
        else:
            res.append(f"{q}{n}")
    return " ".join(res)


def invert_moves(moves):
    return [q[:-1] if q.endswith("'") else q + "'" for q in reversed(moves)]


def to_oracle_moves(moves):
    return [(q[0], q.endswith("'")) for q in moves]

# -------------------------------------------------------------- description


def describe_diff(a, b=None):
    """Human-readable cycles of what changed from state a to state b (default: solved -> a)."""
    if b is None:
        a, b = SOLVED, a
    lines = []
    for title, names, pa, oa, pb, ob, mod in (
            ("corners", CORNER_NAMES, a.cp, a.co, b.cp, b.co, 3),
            ("ring", RING_NAMES, a.rp, a.ro, b.rp, b.ro, 4),
            ("gears", GEAR_NAMES, a.gp, a.go, b.gp, b.go, 4)):
        where = {int(p): i for i, p in enumerate(pb)}
        seen, items = set(), []
        for i in range(len(names)):
            if i in seen:
                continue
            cyc, cur = [], i
            while cur not in seen:
                seen.add(cur)
                cyc.append(cur)
                cur = where[int(pa[cur])]
            tw = [(int(ob[where[int(pa[c])]]) - int(oa[c])) % mod for c in cyc]
            if title == "ring":
                tw = [0 if IS_CENTER[pa[c]] else t for c, t in zip(cyc, tw)]
            if len(cyc) == 1 and tw[0] == 0:
                continue
            unit = 120 if mod == 3 else 90
            lab = [names[c] + (f"(+{t * unit})" if t else "") for c, t in zip(cyc, tw)]
            items.append(" -> ".join(lab) + (" -> " + names[cyc[0]] if len(cyc) > 1 else ""))
        if items:
            lines.append(f"{title}: " + "; ".join(items))
    return "\n".join(lines) if lines else "no change"


# -------------------------------------------------------------- net rendering (text)

COLOR_LETTER = {"U": "W", "D": "Y", "F": "G", "B": "B", "L": "R", "R": "P"}


def net(state):
    """A text net: for each face, the corner stickers, edge/center stickers and gears.

    Each face is printed as a 5x5 block in the kingdeim net orientation
    (U on top; L F R in the middle row; D below F; B below D, upside down):

        c g e g c      c = corner sticker colour
        g . . . g      g = gear colour + spin digit (0 = pointing at its edge)
        e . m . e      e = edge-slot sticker colour on this face, m = center slot
        g . . . g
        c g e g c

    Colours: W white (U), Y yellow (D), G green (F), B blue (B), R red (L), P purple (R).
    """
    ost = to_oracle_state(state)
    corner, ec, gear = O.index(ost)
    FR = {"U": [[1, 0, 0], [0, 0, 1]], "D": [[1, 0, 0], [0, 0, -1]], "F": [[1, 0, 0], [0, -1, 0]],
          "B": [[1, 0, 0], [0, 1, 0]], "R": [[0, 0, -1], [0, -1, 0]], "L": [[0, 0, 1], [0, -1, 0]]}

    def home_face(R, p):
        h = O.mulMV(O.tr(R), p)
        k = max(range(3), key=lambda i: abs(h[i]))
        v = [0, 0, 0]
        v[k] = 1 if h[k] > 0 else -1
        return O.letter(v)

    def P3(face, u, v):
        r, d = FR[face]
        return O.add(O.FACES[face], O.add(O.scl(r, u), O.scl(d, v)))

    blocks = {}
    for face in FACES:
        g = [["  " for _ in range(5)] for _ in range(5)]
        for su in (-1, 1):
            for sv in (-1, 1):
                p3 = P3(face, su, sv)
                slot = next(c for c in O.CORNER_SLOTS if O.near(c["pos"], [math.copysign(1, x) for x in p3]))
                c = corner[slot["name"]]
                g[2 + 2 * sv][2 + 2 * su] = COLOR_LETTER[home_face(c["R"], P3(face, su * .5, sv * .5))] + " "
        g[2][2] = COLOR_LETTER[home_face(ec[face]["R"], P3(face, 0, 0))] + " "
        for nb in FACES:
            nv = O.FACES[nb]
            if abs(O.dot(nv, O.FACES[face])) > 1e-6:
                continue
            r, d = FR[face]
            du, dv = round(O.dot(nv, r)), round(O.dot(nv, d))
            en = O.edge_name(face, nb)
            # sample the edge stub at its centre point on this face
            g[2 + 2 * dv][2 + 2 * du] = COLOR_LETTER[home_face(ec[en]["R"], P3(face, du * .93, dv * .93))] + " "
            gp = gear[face + nb]
            # gear cells sit between the centre and the edge stub
            pos = (2 + dv, 2 + du)
            g[pos[0]][pos[1]] = COLOR_LETTER[gp["id"][0]] + str(O.gear_quarter(gp))
        blocks[face] = ["".join(row).rstrip() for row in g]
    pad = " " * 12
    out = []
    for row in blocks["U"]:
        out.append(pad + row)
    out.append("")
    for i in range(5):
        out.append("  ".join(blocks[f][i].ljust(10) for f in "LFR").rstrip())
    out.append("")
    for f in "DB":
        for row in blocks[f]:
            out.append(pad + row)
        out.append("")
    return "\n".join(out).rstrip()


def to_oracle_state(state):
    """Rebuild an oracle (float) state from a discrete state (for rendering)."""
    st = O.solved()
    # corners: find a rotation for each piece consistent with slot+twist by search over
    # the 24 cube rotations
    rots = _cube_rotations()
    corners = []
    for i, n in enumerate(CORNER_NAMES):
        pid = CORNER_NAMES[state.cp[i]]
        home = next(s for s in O.CORNER_SLOTS if s["name"] == pid)["pos"]
        tgt = next(s for s in O.CORNER_SLOTS if s["name"] == n)["pos"]
        for R in rots:
            if O.near([round(x) for x in O.mulMV(R, home)], tgt):
                c = {"id": pid, "pos": list(tgt), "R": R}
                if O.corner_twist(c) == state.co[i]:
                    corners.append(c)
                    break
        else:
            raise AssertionError("corner")
    ecs = []
    for i, n in enumerate(RING_NAMES):
        pid = RING_NAMES[state.rp[i]]
        rh, th = RING_FRAMES[pid]
        rs, ts = RING_FRAMES[n]
        # rotation taking (rh, th) to (rs, rot(rs, 90*o) ts)
        t2 = O.mulMV(O.rot(rs, 90 * int(state.ro[i])), ts)
        R = _frame_rotation(rh, th, rs, t2)
        ecs.append({"id": pid, "type": "center" if pid in O.FACES else "edge",
                    "dir": list(next(s for s in O.EC_SLOTS if s["name"] == n)["dir"]), "R": R})
    gears = []
    for i, n in enumerate(GEAR_NAMES):
        pid = GEAR_NAMES[state.gp[i]]
        f, d = O.FACES[n[0]], O.FACES[n[1]]
        P = O.mulMV(O.rot(f, 90 * int(state.go[i])), d)
        gears.append({"id": pid, "f": list(f), "d": list(d), "P": P})
    st["corners"], st["ecs"], st["gears"] = corners, ecs, gears
    return st


def _frame_rotation(a1, b1, a2, b2):
    """Rotation matrix R with R a1 = a2, R b1 = b2 (orthonormal pairs)."""
    c1, c2 = O.cross(a1, b1), O.cross(a2, b2)
    M1 = [a1, b1, c1]  # rows
    M2 = [a2, b2, c2]
    # R = M2^T M1
    return [[sum(M2[k][i] * M1[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


_ROTS = None


def _cube_rotations():
    global _ROTS
    if _ROTS is None:
        rs = [O.I3]
        frontier = [O.I3]
        gens = [O.rot([1, 0, 0], 90), O.rot([0, 1, 0], 90), O.rot([0, 0, 1], 90)]
        key = lambda R: tuple(round(x) for r in R for x in r)
        seen = {key(O.I3)}
        while frontier:
            nf = []
            for R in frontier:
                for G in gens:
                    M = O.mulMM(G, R)
                    if key(M) not in seen:
                        seen.add(key(M))
                        rs.append(M)
                        nf.append(M)
            frontier = nf
        _ROTS = rs
    return _ROTS


# -------------------------------------------------------------- group membership / sampling
#
# Facts used here are established in groupfacts.py: the group generated
# by U, R, F (DBL corner fixed) is exactly the set of states with
#   DBL corner home with twist 0, corner twist sum = 0 mod 3,
#   ring permutation even, ring spin sum = corner parity (mod 2),
#   gear permutation parity = corner parity, gear spin sum = 0 mod 4.

DBL = CORNER_NAMES.index("DBL")


def _parity(p):
    p = [int(x) for x in p]
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


def check(state, fixed_dbl=True):
    """List of violated invariants (empty = reachable)."""
    bad = []
    cpar = _parity(state.cp)
    if sorted(state.cp.tolist()) != list(range(NC)) or sorted(state.rp.tolist()) != list(range(NR)) \
            or sorted(state.gp.tolist()) != list(range(NG)):
        return ["not a permutation (a piece is missing or duplicated)"]
    if fixed_dbl and (state.cp[DBL] != DBL or state.co[DBL] != 0):
        bad.append("DBL corner is not home (normalize the rotation first)")
    if int(state.co.sum()) % 3:
        bad.append(f"corner twist sum is {int(state.co.sum()) % 3} mod 3 (a corner is twisted in place)")
    if _parity(state.rp):
        bad.append("ring permutation is odd (two ring pieces are swapped)")
    if (int(state.ro.sum()) + cpar) % 2:
        bad.append("ring spin sum does not match corner parity (a ring piece is turned 90 degrees)")
    if _parity(state.gp) != cpar:
        bad.append("gear permutation parity does not match corner parity")
    if int(state.go.sum()) % 4:
        bad.append(f"gear spin sum is {int(state.go.sum()) % 4} quarter turns mod 4 (a gear is turned)")
    return bad


def random_state(rng):
    """Uniformly random element of <U,R,F> (np.random.Generator)."""
    cp = np.append(rng.permutation(7), 7).astype(np.int8)
    co = np.append(rng.integers(0, 3, 6), 0)
    co = np.append(co[:6], [(-co[:6].sum()) % 3, 0]).astype(np.int8)
    cpar = _parity(cp)
    rp = rng.permutation(NR).astype(np.int8)
    if _parity(rp):
        rp[[0, 1]] = rp[[1, 0]]
    ro = rng.integers(0, 4, NR).astype(np.int8)
    if (int(ro.sum()) + cpar) % 2:
        ro[0] = (ro[0] + 1) % 4
    gp = rng.permutation(NG).astype(np.int8)
    if _parity(gp) != cpar:
        gp[[0, 1]] = gp[[1, 0]]
    go = rng.integers(0, 4, NG).astype(np.int8)
    go[0] = (go[0] - int(go.sum())) % 4
    return State(cp, co, rp, ro, gp, go)


def _all_rotations():
    out, frontier = {IDENTITY.state().key(): IDENTITY}, [IDENTITY]
    while frontier:
        nf = []
        for e in frontier:
            for r in ("x", "y", "z"):
                f = e * ROTATIONS[r]
                k = f.state().key()
                if k not in out:
                    out[k] = f
                    nf.append(f)
        frontier = nf
    assert len(out) == 24
    return list(out.values())


ALL_ROTATIONS = _all_rotations()


def normalize_rotation(state):
    """The same physical puzzle viewed so that the DBL corner is home (unique)."""
    for r in ALL_ROTATIONS:
        s = state.apply(r)
        if s.cp[DBL] == DBL and s.co[DBL] == 0:
            return s
    raise AssertionError("no rotation fixes DBL")
