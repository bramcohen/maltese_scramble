"""Float-matrix reference oracle: a line-by-line port of the simulation in
kingdeim/MalteseGearCube (index.html: SLOTS, solved(), quarter(), wholeRot(),
index(), gearQuarter(), cornerTwist()).

This is kept only as a test fixture.  The working model is the discrete one in
model.py, which is checked against this file (tests/test_model.py).
"""
import math

EPS = 1e-6


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def scl(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def norm(a):
    return scl(a, 1 / math.hypot(*a))


I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def mulMV(M, v):
    return [dot(M[0], v), dot(M[1], v), dot(M[2], v)]


def mulMM(A, B):
    return [[r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j] for j in range(3)] for r in A]


def tr(M):
    return [[M[j][i] for j in range(3)] for i in range(3)]


def rot(axis, deg):
    """Rodrigues; positive = counterclockwise seen from the axis tip."""
    t = deg * math.pi / 180
    c, s = math.cos(t), math.sin(t)
    x, y, z = axis
    C = 1 - c
    return [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]


def near(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) < 1e-4


FACES = {"U": [0, 1, 0], "D": [0, -1, 0], "F": [0, 0, 1], "B": [0, 0, -1], "R": [1, 0, 0], "L": [-1, 0, 0]}
ORDER = "UDFBRL"
GEAR_R = 0.525
GEAR_SPIN = -90


def letter(v):
    return next(k for k in ORDER if dot(FACES[k], v) > 0.9)


def edge_name(a, b):
    return "".join(sorted([a, b], key=ORDER.index))


CORNER_SLOTS = []
for sy in (1, -1):
    for sz in (1, -1):
        for sx in (1, -1):
            CORNER_SLOTS.append({"name": letter([0, sy, 0]) + letter([0, 0, sz]) + letter([sx, 0, 0]),
                                 "pos": [sx, sy, sz]})
EC_SLOTS = [{"name": k, "dir": FACES[k], "type": "center"} for k in ORDER]
for i in range(6):
    for j in range(i + 1, 6):
        a, b = ORDER[i], ORDER[j]
        if abs(dot(FACES[a], FACES[b])) < EPS:
            EC_SLOTS.append({"name": edge_name(a, b), "dir": norm(add(FACES[a], FACES[b])), "type": "edge"})
GEAR_SLOTS = [{"name": f + d, "f": FACES[f], "d": FACES[d]}
              for f in ORDER for d in ORDER if abs(dot(FACES[f], FACES[d])) < EPS]


def ec_slot_by_dir(d):
    best = EC_SLOTS[0]
    for s in EC_SLOTS:
        if dot(s["dir"], d) > dot(best["dir"], d):
            best = s
    return best


def solved():
    return {
        "corners": [{"id": s["name"], "pos": list(s["pos"]), "R": I3} for s in CORNER_SLOTS],
        "ecs": [{"id": s["name"], "type": s["type"], "dir": list(s["dir"]), "R": I3} for s in EC_SLOTS],
        "gears": [{"id": s["name"], "f": list(s["f"]), "d": list(s["d"]), "P": list(s["d"])} for s in GEAR_SLOTS],
    }


def quarter(st, face, sense):
    """sense +1 = clockwise.  Mutates st."""
    n = FACES[face]
    Q = rot(n, -90 * sense)
    M = rot(n, -45 * sense)
    for c in st["corners"]:
        if dot(c["pos"], n) > EPS:
            c["pos"] = [round(x) for x in mulMV(Q, c["pos"])]
            c["R"] = mulMM(Q, c["R"])
    for e in st["ecs"]:
        d = dot(e["dir"], n)
        if d > EPS:
            e["dir"] = list(ec_slot_by_dir(mulMV(Q, e["dir"]))["dir"])
            e["R"] = mulMM(Q, e["R"])
        elif abs(d) < EPS:
            e["dir"] = list(ec_slot_by_dir(mulMV(M, e["dir"]))["dir"])
            e["R"] = mulMM(M, e["R"])
    for g in st["gears"]:
        p = add(g["f"], scl(g["d"], GEAR_R))
        if dot(p, n) > EPS:
            g["f"] = [round(x) for x in mulMV(Q, g["f"])]
            g["d"] = [round(x) for x in mulMV(Q, g["d"])]
            g["P"] = mulMV(Q, g["P"])
        elif abs(dot(g["f"], n)) < EPS and abs(dot(g["d"], n)) < EPS:
            rf = [round(x) for x in mulMV(Q, g["f"])]
            if near(g["d"], rf):
                nf, nd = g["d"], g["f"]
            else:
                nf, nd = g["f"], rf
            a = dot(g["P"], cross(n, g["f"]))
            b = dot(g["P"], n)
            P = add(scl(cross(n, nf), a), scl(n, b))
            P = mulMV(rot(nf, GEAR_SPIN * sense), P)
            g["f"], g["d"], g["P"] = list(nf), list(nd), P
    return st


def whole_rot(st, axis_face, sense):
    Q = rot(FACES[axis_face], -90 * sense)
    for c in st["corners"]:
        c["pos"] = [round(x) for x in mulMV(Q, c["pos"])]
        c["R"] = mulMM(Q, c["R"])
    for e in st["ecs"]:
        e["dir"] = list(ec_slot_by_dir(mulMV(Q, e["dir"]))["dir"])
        e["R"] = mulMM(Q, e["R"])
    for g in st["gears"]:
        g["f"] = [round(x) for x in mulMV(Q, g["f"])]
        g["d"] = [round(x) for x in mulMV(Q, g["d"])]
        g["P"] = mulMV(Q, g["P"])
    return st


def apply(st, moves):
    """moves: iterable of (face, prime) quarter turns, face in UDFBRLxyz."""
    for face, prime in moves:
        if face in "xyz":
            whole_rot(st, {"x": "R", "y": "U", "z": "F"}[face], -1 if prime else 1)
        else:
            quarter(st, face, -1 if prime else 1)
    return st


def index(st):
    corner, ec, gear = {}, {}, {}
    for c in st["corners"]:
        corner[next(s["name"] for s in CORNER_SLOTS if near(s["pos"], c["pos"]))] = c
    for e in st["ecs"]:
        ec[ec_slot_by_dir(e["dir"])["name"]] = e
    for g in st["gears"]:
        gear[letter(g["f"]) + letter(g["d"])] = g
    return corner, ec, gear


def gear_quarter(g):
    a = math.atan2(dot(g["P"], cross(g["f"], g["d"])), dot(g["P"], g["d"]))
    return round(a / (math.pi / 2)) % 4


def corner_twist(c):
    ud = [0, math.copysign(1, c["pos"][1]), 0]
    h = mulMV(tr(c["R"]), ud)
    hs = next(s for s in CORNER_SLOTS if s["name"] == c["id"])["pos"]
    n0 = [0, hs[1], 0]
    o = [[hs[0], 0, 0], [0, 0, hs[2]]]
    n1, n2 = o if dot(cross(n0, o[0]), o[1]) > 0 else (o[1], o[0])
    return next(i for i, v in enumerate([n0, n1, n2]) if dot(v, h) > 0.9)
