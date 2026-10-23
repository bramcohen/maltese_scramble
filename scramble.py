"""Scrambles for the Maltese Gear Cube.

    .venv/bin/python scramble.py                    # random-move scramble, 100 moves (half-turn metric)
    .venv/bin/python scramble.py --random-move 150  # longer random-move scramble
    .venv/bin/python scramble.py -n 5 --seed 1      # five scrambles, reproducible
    .venv/bin/python scramble.py --random-state     # provably uniform: solve a uniform random state (~36 min; --frames 1: ~12 min)

The random-move length 100 is the smallest clean length in the uniformity tests (75)
plus a margin; 50 is borderline and 25 is detectably non-uniform.

Notation (zera / kingdeim): R = right half a quarter turn clockwise, R' counter-
clockwise, R3 = three quarter turns, R'2 = two counter-clockwise.  Hold the cube
white (U) on top, green (F) in front.  After the moves the cube should match the
printed net (see model.net for the legend).

Random-state: a uniformly random reachable state is drawn (model.random_state, valid
by groupfacts.py), solved with the subgroup ladder (ladder.py), and the solution
is inverted; the result is uniform by construction.  Random-move: N moves, each a
single or double turn of U, R or F (never the face just turned), fanout 8 per move.
"""
import argparse
import random
import sys
import time

import numpy as np

import model as M


RANDOM_FACES = "URF"                      # D, B, L are fixed; R and L' differ only by a whole-cube rotation
RANDOM_TURNS = ("", "'", "2", "'2")       # single and double turns in both directions (X2 and X'2 differ here)


def random_move(n, rng):
    """n random half-turn-metric moves: each move turns one of U, R, F (never the face just turned) by a
    single or double turn in either direction, so the fanout is 8 at every step.  Returned as run
    tokens (e.g. "R'2"); use runs_to_quarters for quarter turns."""
    seq, prev = [], None
    for _ in range(n):
        f = rng.choice([x for x in RANDOM_FACES if x != prev])
        seq.append(f + rng.choice(RANDOM_TURNS)); prev = f
    return seq


def invert_runs(runs):
    out = []
    for m in reversed(runs):
        f, rest = m[0], m[1:]
        if rest.startswith("'"):
            out.append(f + rest[1:])
        else:
            out.append(f + "'" + rest)
    return [r.replace("'1", "'") if r.endswith("'1") else r for r in out]


def runs_to_quarters(runs):
    return M.parse(" ".join(runs))


def normalise_runs(quarters):
    """Merge consecutive turns of one face into a single run, counted modulo 8 (a face has order 8 on this
    puzzle), written as the shortest run: X, X2, X3, X4, X', X'2, X'3; cancelled runs are dropped."""
    runs = []
    for q in quarters:
        f, k = q[0], -1 if q.endswith("'") else 1
        if runs and runs[-1][0] == f:
            runs[-1][1] = (runs[-1][1] + k) % 8
            if runs[-1][1] == 0:
                runs.pop()
        else:
            runs.append([f, k % 8])
    out = []
    for f, k in runs:
        k = k - 8 if k > 4 else k
        out.append(f + ("'" if k < 0 else "") + ("" if abs(k) == 1 else str(abs(k))))
    return out


def htm(runs):
    """Half-turn-metric length of run tokens: X and X2 count 1, X3 and X4 count 2 (after normalising runs)."""
    return sum(1 if abs(int(t.lstrip("UDRLFB'") or 1)) <= 2 else 2 for t in normalise_runs(M.parse(" ".join(runs))))


def random_state_scramble(solver, rng_np, frames=3):
    """Returns (scramble runs, target state).  solver.solve(state, frames) -> (HTM, plan) with plan a list of
    (rung name, words), the cheapest of `frames` solves in the three frames of the fixed corner; the scramble is
    the inverse of the concatenated solution."""
    st = M.random_state(rng_np)
    cost, plan = solver.solve(st, frames=frames)
    sol = normalise_runs(M.parse(" ".join(w for name, ws in plan for w in ws)))
    scr = invert_runs(sol)
    # verify: applying the scramble to solved gives the target (visibly)
    got = M.apply_seq(M.State(), runs_to_quarters(scr))
    assert got.visible() == st.visible(), "scramble does not reach the target state"
    return scr, st


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--random-move", type=int, metavar="N", default=100,
                    help="random-move scramble of N moves in the half-turn metric (default 100)")
    ap.add_argument("--random-state", action="store_true", help="random-state scramble: a uniformly random state solved by the subgroup ladder (ladder.py; about 30 s and 30 GB, see LADDER.md)")
    ap.add_argument("--frames", type=int, default=3, help="random-state: solve in 1-3 frames and keep the cheapest (default 3; 1 is three times faster, ~10 moves longer)")
    ap.add_argument("-n", type=int, default=1, help="number of scrambles")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-net", action="store_true", help="do not print the target net")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)
    solver = None
    for i in range(args.n):
        t = time.time()
        if not args.random_state:
            runs = random_move(args.random_move, rng)
            text = " ".join(runs)
            seq = runs_to_quarters(runs)
            target = M.apply_seq(M.State(), seq)
            qt, ht = len(seq), len(runs)
        else:
            if solver is None:
                import ladder
                print("loading the ladder's tables (about 10 s; they must have been built, see LADDER.md)...", file=sys.stderr)
                solver = ladder.Ladder()
            try:
                scr, target = random_state_scramble(solver, rng_np, frames=args.frames)
            except RuntimeError as e:
                sys.exit(f"random-state solver did not finish ({e})")
            text = " ".join(scr)
            qt = len(runs_to_quarters(scr)); ht = htm(scr)
        print(f"{i + 1}. [{ht} moves (HTM), {qt} quarter turns, {time.time() - t:.1f}s]  {text}")
        if not args.no_net:
            print(M.net(M.normalize_rotation(target)) if args.random_state else M.net(target))
            print()


if __name__ == "__main__":
    main()
