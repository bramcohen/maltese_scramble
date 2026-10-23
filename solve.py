"""Solve, check and describe Maltese Gear Cube states.

    .venv/bin/python solve.py --solve state.txt          # solve a state (- for stdin): prints the moves
    .venv/bin/python solve.py --solve-seq "R U R' U'"    # solve the state reached by a sequence
    .venv/bin/python solve.py --show "R U R' U'"        # print the state reached by a sequence
    .venv/bin/python solve.py --check state.txt          # is this (visible) state reachable?
    .venv/bin/python solve.py --check-seq "R U R'"       # same, for a sequence (always reachable)

Solving needs the solver's tables (LADDER.md) and about 20 GB of memory; a solve takes about half a minute and
averages 135 moves. The solution is verified before it is printed.

State format (what --show prints), slots in this fixed order:

    corners: UFR UFL UBR UBL DFR DFL DBR DBL
             each entry = the corner piece in that slot + its twist digit, e.g. UBR0 DFL2
    ring:    U D F B R L UF UB UR UL DF DB DR DL FR FL BR BL
             each entry = the ring piece in that slot; edge pieces carry a spin digit
             (quarter turns about the slot's radial axis), centers carry none
    gears:   UF UB UR UL DF DB DR DL FU FD FR FL BU BD BR BL RU RD RF RB LU LD LF LB
             each entry = colour letter of the gear in that slot + spin digit,
             W white (U) Y yellow (D) G green (F) B blue (B) P purple (R) R red (L)

Only what a person can see is used: the four gears of a face are identical and a
center's spin is invisible.  The check tests every visible invariant and says which
one fails; the solver picks any invisible labelling that makes the state reachable
(the solution is the same for all of them).
"""
import argparse
import sys

import numpy as np

import model as M

LETTER = {"U": "W", "D": "Y", "F": "G", "B": "B", "R": "P", "L": "R"}
FACE_OF = {v: k for k, v in LETTER.items()}


def show(st):
    st = M.normalize_rotation(st)
    c = " ".join(f"{M.CORNER_NAMES[p]}{o}" for p, o in zip(st.cp, st.co))
    r = " ".join(M.RING_NAMES[p] + ("" if M.IS_CENTER[p] else str(o)) for p, o in zip(st.rp, st.ro))
    g = " ".join(f"{LETTER[M.FACES[M.GEAR_COLOR[p]]]}{o}" for p, o in zip(st.gp, st.go))
    return f"corners: {c}\nring: {r}\ngears: {g}"


def parse_state(text):
    """Parse the --show format into (cp, co, rp, ro, gear colours, go) arrays."""
    rows = {}
    for line in text.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            rows[k.strip().lower()] = v.split()
    try:
        cs, rs, gs = rows["corners"], rows["ring"], rows["gears"]
    except KeyError as e:
        raise ValueError(f"missing line {e}")
    if (len(cs), len(rs), len(gs)) != (8, 18, 24):
        raise ValueError(f"expected 8 corners, 18 ring entries, 24 gears; got {len(cs)}, {len(rs)}, {len(gs)}")
    cp = [M.CORNER_NAMES.index(x[:3]) for x in cs]
    co = [int(x[3]) for x in cs]
    rp, ro = [], []
    for x in rs:
        name = x.rstrip("0123")
        rp.append(M.RING_NAMES.index(name))
        ro.append(int(x[len(name):] or 0))
    gc = [M.FACES.index(FACE_OF[x[0]]) for x in gs]
    go = [int(x[1]) for x in gs]
    return cp, co, rp, ro, gc, go


def check_visible(cp, co, rp, ro, gc, go):
    """Visible invariants (groupfacts.py).  Returns a list of problems."""
    bad = []
    if sorted(cp) != list(range(8)):
        bad.append("corners: a corner piece is missing or appears twice")
    if sorted(rp) != list(range(18)):
        bad.append("ring: a ring piece is missing or appears twice")
    if sorted(gc) != sorted(int(c) for c in M.GEAR_COLOR):
        bad.append("gears: there must be exactly four gears of each colour")
    if any(not 0 <= o < 3 for o in co) or any(not 0 <= o < 4 for o in list(ro) + list(go)):
        bad.append("an orientation digit is out of range")
    if bad:
        return bad
    if sum(co) % 3:
        bad.append(f"corner twists sum to {sum(co) % 3} mod 3: a corner is twisted in place")
    if M._parity(rp):
        bad.append("ring permutation is odd: two ring pieces are swapped")
    if sum(go) % 4:
        bad.append(f"gear spins sum to {sum(go) % 4} (mod 4, in quarter turns): a gear is turned in place")
    # corner parity and ring-spin parity couple only to invisible data (gear
    # identities, center spins), so they impose no visible constraint.
    return bad


def lift(cp, co, rp, ro, gc, go):
    """A reachable full state showing the given visible state: same-coloured gears are numbered in slot order and the
    centre spins are 0, then the two invisible invariants are met by a centre spin and a swap of two gears of one
    colour.  Raises ValueError if the visible state itself is unreachable."""
    bad = check_visible(cp, co, rp, ro, gc, go)
    if bad:
        raise ValueError("; ".join(bad))
    pieces = {c: [p for p in range(24) if int(M.GEAR_COLOR[p]) == c] for c in range(6)}
    gp = [pieces[c].pop(0) for c in gc]
    st = M.State(np.array(cp, np.int8), np.array(co, np.int8), np.array(rp, np.int8), np.array(ro, np.int8),
                 np.array(gp, np.int8), np.array(go, np.int8))
    st = M.normalize_rotation(st)
    cpar = M._parity(st.cp)
    if (int(st.ro.sum()) + cpar) % 2:
        centre = next(i for i in range(18) if M.IS_CENTER[st.rp[i]])
        st.ro[centre] = (st.ro[centre] + 1) % 4
    if M._parity(st.gp) != cpar:
        a = next(i for i in range(24) if int(M.GEAR_COLOR[st.gp[i]]) == int(M.GEAR_COLOR[st.gp[0]]) and i != 0)
        st.gp[[0, a]] = st.gp[[a, 0]]
    assert not M.check(st), M.check(st)
    return st


def solve_state(st, frames=3):
    """(HTM, run tokens) solving a full state with the ladder; the solution is checked to leave the puzzle visibly
    solved."""
    import ladder
    from scramble import normalise_runs, runs_to_quarters, htm
    print("loading the solver's tables (about 10 s; they must have been built, see LADDER.md)...", file=sys.stderr)
    cost, plan = ladder.Ladder().solve(st, frames=frames)
    sol = normalise_runs(M.parse(" ".join(w for name, ws in plan for w in ws)))
    got = M.apply_seq(st, runs_to_quarters(sol))
    assert got.visible() == M.State().visible(), "the solution does not solve the state"
    return htm(sol), sol


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solve", metavar="FILE", help="solve a state file (- for stdin)")
    ap.add_argument("--solve-seq", metavar="SEQ", help="solve the state reached by a sequence")
    ap.add_argument("--frames", type=int, default=3, help="solve in 1-3 frames of the fixed corner, keep the cheapest (default 3)")
    ap.add_argument("--show", metavar="SEQ", help="print the state reached by a move sequence")
    ap.add_argument("--check", metavar="FILE", help="check a state file (- for stdin)")
    ap.add_argument("--check-seq", metavar="SEQ", help="check the state reached by a sequence")
    ap.add_argument("--net", action="store_true", help="also print a text net")
    args = ap.parse_args(argv)
    if args.solve or args.solve_seq:
        if args.solve_seq:
            st = M.normalize_rotation(M.apply_seq(M.State(), args.solve_seq))
        else:
            text = sys.stdin.read() if args.solve == "-" else open(args.solve).read()
            try:
                st = lift(*parse_state(text))
            except (ValueError, IndexError) as e:
                print(f"cannot solve: {e}")
                return 2
        if args.net:
            print(M.net(st))
        cost, sol = solve_state(st, frames=args.frames)
        print(f"{cost} moves (HTM): {' '.join(sol)}")
        return 0
    if args.show or args.check_seq:
        st = M.apply_seq(M.State(), args.show or args.check_seq)
        print(show(st))
        if args.net:
            print(M.net(M.normalize_rotation(st)))
        if args.check_seq:
            print("reachable" if not M.check(M.normalize_rotation(st)) else M.check(M.normalize_rotation(st)))
        return 0
    if args.check:
        text = sys.stdin.read() if args.check == "-" else open(args.check).read()
        try:
            bad = check_visible(*parse_state(text))
        except (ValueError, IndexError) as e:
            print(f"cannot parse state: {e}")
            return 2
        if bad:
            print("NOT reachable (the puzzle is assembled wrong):")
            for b in bad:
                print("  -", b)
            return 1
        print("reachable")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
