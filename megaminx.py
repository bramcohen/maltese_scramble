"""Position-only megaminx model, derived from dodecahedron geometry, and the WCA (Pochmann) scrambler.

Pieces: 20 corners (dodecahedron vertices) and 30 edges (edge midpoints); orientation is ignored.  A face turn
is a 72-degree rotation about the face normal applied to the 5 corners and 5 edges of that face.  Face
permutations are found numerically by rotating coordinates and matching, so they are correct by construction
(checked: every face turn has order 5, opposite faces commute, the position group is A20 x A30).

The WCA scramble (TNoodle) is 7 lines of "R++ D++ R-- D++ ... U" : R++ turns everything except the left face
by 144 degrees about the left face's axis, D++ everything except the top face about the top axis, and each line
ends with U (if the last D was ++) or U' (if --).  Relative to the core (the centres), R++ is the face
currently on the left turned by two clicks the other way, plus a change of the physical orientation; the
simulation tracks the orientation as an element of the 60-element rotation group so every move acts on the
right core face.

Distance metric (the user's): sum over all 50 pieces of the graph distance between a piece's position and its
home position, where two positions are adjacent if one face turn (one click) maps one to the other.
"""
import itertools

import numpy as np

PHI = (1 + 5 ** 0.5) / 2


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# ---- geometry: 20 corners (dodecahedron vertices), 12 face normals (the directions with five maximal vertices)
CORNERS = np.array([(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)] +
                   [(0, b / PHI, c * PHI) for b in (1, -1) for c in (1, -1)] +
                   [(b / PHI, c * PHI, 0) for b in (1, -1) for c in (1, -1)] +
                   [(c * PHI, 0, b / PHI) for b in (1, -1) for c in (1, -1)], float)
_cands = {tuple(np.array(p) * np.array(sg)) for p in itertools.permutations((0, 1, PHI)) for sg in itertools.product((1, -1), repeat=3)}
FACES = []
for v in sorted(_cands):
    n = _unit(v); d = CORNERS @ n
    if (d > d.max() - 1e-6).sum() == 5 and not any(np.allclose(n, m) for m in FACES):
        FACES.append(n)
FACES = np.array(FACES)
_edge_len = min(np.linalg.norm(CORNERS[i] - CORNERS[j]) for i in range(20) for j in range(i + 1, 20))
EDGES = np.array([(CORNERS[i] + CORNERS[j]) / 2 for i in range(20) for j in range(i + 1, 20)
                  if abs(np.linalg.norm(CORNERS[i] - CORNERS[j]) - _edge_len) < 1e-6])
assert len(FACES) == 12 and len(CORNERS) == 20 and len(EDGES) == 30
PIECES = np.vstack([CORNERS, EDGES])          # 50 piece positions; a state is "which piece sits at each position"
NP = 50


def _rot(axis, angle):
    a = _unit(axis); x, y, z = a; c, s = np.cos(angle), np.sin(angle); C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _match(points, ref):
    """Index in ref of each point (exact geometric match)."""
    d = np.linalg.norm(points[:, None, :] - ref[None, :, :], axis=2)
    idx = d.argmin(axis=1)
    assert (d[np.arange(len(points)), idx] < 1e-6).all()
    return idx


def _face_turn(f, clicks):
    """Permutation p of positions: position i receives the piece from position p[i] when face f turns `clicks`
    clicks (72 degrees each, clockwise seen from outside for clicks > 0)."""
    n = FACES[f]; on = PIECES @ n > 1.0          # the 5 corners and 5 edges of the face lie in its plane (dot = inradius 1.11)
    R = _rot(n, -2 * np.pi / 5 * clicks)       # clockwise from outside = negative rotation about the outward normal
    moved = PIECES.copy(); moved[on] = PIECES[on] @ R.T
    dest = _match(moved, PIECES)                # piece at position i goes to position dest[i]
    p = np.empty(NP, int); p[dest] = np.arange(NP)
    return p


TURN = {(f, k): _face_turn(f, k) for f in range(12) for k in (1, 2, 3, 4)}


def apply(state, perm):
    """state[i] = piece at position i; returns the new state."""
    return state[..., perm]


# ---- the 60 rotations of the dodecahedron, as permutations of faces, for orientation tracking
def _rotation_group():
    gens = [_rot(FACES[f], 2 * np.pi / 5) for f in range(12)]
    mats = [np.eye(3)]; keys = {tuple(np.round(np.eye(3), 6).ravel())}
    frontier = [np.eye(3)]
    while frontier:
        new = []
        for m in frontier:
            for g in gens:
                r = g @ m; k = tuple(np.round(r, 6).ravel())
                if k not in keys:
                    keys.add(k); mats.append(r); new.append(r)
        frontier = new
    assert len(mats) == 60
    return mats


ROTS = _rotation_group()
ROT_KEY = {tuple(np.round(m, 6).ravel()): i for i, m in enumerate(ROTS)}


def _rot_index(m):
    return ROT_KEY[tuple(np.round(m, 6).ravel())]


# spatial frame: U face normal up (+z); L face: an adjacent face, on the -x side
U_FACE = int(np.argmax(FACES @ np.array([0, 0, 1.0])))
_adj = [f for f in range(12) if abs(FACES[f] @ FACES[U_FACE] - 1 / 5 ** 0.5) < 1e-6]   # 5 faces adjacent to U
L_FACE = min(_adj, key=lambda f: FACES[f][0])                                          # the one furthest to -x
# orientation o: the puzzle is rotated by ROTS[o] from its home placement; the core face at the spatial U
# position is the f with ROTS[o] @ FACES[f] = FACES[U_FACE]
FACE_AT = np.array([[int(np.argmax(FACES @ (ROTS[o].T @ FACES[g]))) for g in range(12)] for o in range(60)])
# FACE_AT[o, g] = core face currently at spatial position g
COMPOSE = {}
for name, axis_face, ang in (("D++", U_FACE, +1), ("D--", U_FACE, -1), ("R++", L_FACE, +1), ("R--", L_FACE, -1)):
    R = _rot(FACES[axis_face], -2 * np.pi / 5 * 2 * ang)      # whole-puzzle rotation by two clicks about the spatial axis
    COMPOSE[name] = np.array([_rot_index(R @ ROTS[o]) for o in range(60)])


def wca_scramble(rng, lines=7):
    """WCA megaminx scramble as a list of (core face, clicks) turns, simulated from a random orientation-free
    start (the orientation only matters relative to the moves, so start at o = 0)."""
    turns = []; o = 0
    for _ in range(lines):
        last_d = None
        for i in range(10):
            sign = 1 if rng.random() < 0.5 else -1
            if i % 2 == 0:                          # R++ / R--: everything except the left face
                f = FACE_AT[o, L_FACE]; turns.append((f, (-2 * sign) % 5)); o = COMPOSE["R++" if sign > 0 else "R--"][o]
            else:                                   # D++ / D--: everything except the top face
                f = FACE_AT[o, U_FACE]; turns.append((f, (-2 * sign) % 5)); o = COMPOSE["D++" if sign > 0 else "D--"][o]
                last_d = sign
        turns.append((FACE_AT[o, U_FACE], 1 if last_d > 0 else 4))    # U after D++, U' after D--
    return turns


def random_turns(rng, n):
    return [(int(rng.integers(12)), int(rng.integers(1, 5))) for _ in range(n)]


def simulate(turn_lists):
    """Batch: list of turn lists -> states array (n, 50)."""
    states = np.tile(np.arange(NP), (len(turn_lists), 1))
    for i, turns in enumerate(turn_lists):
        s = states[i]
        for f, k in turns:
            s = s[TURN[(f, k)]]
        states[i] = s
    return states


def simulate_batch(rng, n, kind="wca", length=77):
    """Vectorised simulation of n scrambles.  kind: 'wca' (7 lines) or 'random' (uniform random face turns)."""
    states = np.tile(np.arange(NP), (n, 1))
    if kind == "wca":
        o = np.zeros(n, int)
        for line in range(length // 11):
            last_d = np.ones(n, int)
            for i in range(10):
                sign = np.where(rng.random(n) < 0.5, 1, -1)
                g = L_FACE if i % 2 == 0 else U_FACE
                f = FACE_AT[o, g]; k = (-2 * sign) % 5
                for ff in range(12):
                    for kk in (3, 2):
                        sel = (f == ff) & (k == kk)
                        if sel.any():
                            states[sel] = states[sel][:, TURN[(ff, kk)]]
                tab = COMPOSE[("R++" if i % 2 == 0 else "D++")] if True else None
                o = np.where(sign > 0, COMPOSE["R++" if i % 2 == 0 else "D++"][o], COMPOSE["R--" if i % 2 == 0 else "D--"][o])
                if i % 2 == 1:
                    last_d = sign
            f = FACE_AT[o, U_FACE]; k = np.where(last_d > 0, 1, 4)
            for ff in range(12):
                for kk in (1, 4):
                    sel = (f == ff) & (k == kk)
                    if sel.any():
                        states[sel] = states[sel][:, TURN[(ff, kk)]]
    else:
        for _ in range(length):
            f = rng.integers(12, size=n); k = rng.integers(1, 5, size=n)
            for ff in range(12):
                for kk in (1, 2, 3, 4):
                    sel = (f == ff) & (k == kk)
                    if sel.any():
                        states[sel] = states[sel][:, TURN[(ff, kk)]]
    return states


def uniform_positions(rng, n):
    """Uniform random position states of the megaminx: an even permutation of the corners and an even permutation
    of the edges (every face turn is a 5-cycle on each, and A20 x A30 is reached)."""
    out = np.empty((n, NP), int)
    for i in range(n):
        for lo, hi in ((0, 20), (20, 50)):
            p = rng.permutation(hi - lo)
            if _parity(p):
                p[0], p[1] = p[1], p[0]
            out[i, lo:hi] = p + lo
    return out


def _parity(p):
    p = list(p); n = 0
    for i in range(len(p)):
        while p[i] != i:
            j = p[i]; p[i], p[j] = p[j], p[i]; n += 1
    return n & 1


# ---- distance tables: graph distance between positions (one click of any face = one step)
def _distance_table():
    adj = [set() for _ in range(NP)]
    for (f, k), p in TURN.items():
        if k == 1:
            for i in range(NP):
                if p[i] != i:
                    adj[i].add(p[i]); adj[p[i]].add(i)
    D = np.full((NP, NP), -1, int)
    for s in range(NP):
        D[s, s] = 0; frontier = [s]; d = 0
        while frontier:
            d += 1; new = []
            for a in frontier:
                for b in adj[a]:
                    if D[s, b] < 0:
                        D[s, b] = d; new.append(b)
            frontier = new
    return D


DIST = _distance_table()


def total_distance(states):
    """Sum over pieces of the distance from the piece's current position to its home position."""
    pos = np.argsort(states, axis=1)                  # pos[n, piece] = position of that piece
    return DIST[np.arange(NP)[None, :], pos].sum(axis=1)


def piece_distances(states):
    pos = np.argsort(states, axis=1)
    return DIST[np.arange(NP)[None, :], pos]


if __name__ == "__main__":
    # sanity checks
    for f in range(12):
        p = TURN[(f, 1)]; q = np.arange(NP)
        for _ in range(5):
            q = q[p]
        assert (q == np.arange(NP)).all(), "face turn must have order 5"
        assert (TURN[(f, 2)] == p[p]).all()
    opp = {f: int(np.argmin(FACES @ FACES[f])) for f in range(12)}
    for f in range(12):
        a, b = TURN[(f, 1)], TURN[(opp[f], 1)]
        assert (a[b] == b[a]).all(), "opposite faces commute"
    print("corner distances: max", DIST[:20, :20].max(), "edge distances: max", DIST[20:, 20:].max())
    print("U face", U_FACE, "L face", L_FACE, "faces adjacent to U:", _adj)
    rng = np.random.default_rng(0)
    print("example scramble (core face, clicks):", wca_scramble(rng)[:12])
