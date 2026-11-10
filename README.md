# Maltese Gear Cube solver

A solver for Meffert's Maltese Gear Cube (Oskar van Deventer's geared Mixup cube). It is a Kociemba-style ladder of
eight nested subgroups with pattern tables for each rung, solved up to the relabellings the eye cannot see. A solve
averages 135 moves in the half-turn metric and takes about 35 s; every solution is verified before it is reported.
[LADDER.md](LADDER.md) describes the ladder, the measurements and the tables.

The simulation model is a discrete port of [kingdeim/MalteseGearCube](https://github.com/kingdeim/MalteseGearCube),
verified against it move by move.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install numpy sympy scipy
```

A C compiler is needed (`csrc/coset.c`, `fulltab/ft.c`; built automatically on first use).

The solver's tables are built once:

```bash
PYTHONPATH=. .venv/bin/python fulltab/build.py
```

This builds everything from the puzzle model, with nothing generated checked in. It takes a few hours and leaves
about 50 GB of tables plus a 42 GB ball of states that is only used while building; the big steps need 60 GB of
memory (one at a time on a 64 GB machine). `--rung7 large` builds a larger
table for the last big rung instead (13 h, 1.1 TB of disk while building) and brings solves to 134 moves and 28 s;
the solver uses it whenever it is present. Details and times per step are in LADDER.md.

## Solving

```bash
.venv/bin/python solve.py --solve-seq "R U R' U'"   # solve the state a sequence reaches
.venv/bin/python solve.py --solve mystate.txt       # solve a state written down from the puzzle
.venv/bin/python solve.py --show "R U R' U'"        # print a state in that format
```

The solver loads its tables in about 10 s and holds about 20 GB of memory. The state format is documented at the top
of `solve.py`: the corner, ring and gear pieces slot by slot with their twist or spin digits, using only what a
person can see (the four gears of a face are identical and a centre's spin is invisible). From Python,
`ladder.Ladder().solve(state)` returns the move count and the plan rung by rung.

- Hold the cube white (U) on top and green (F) in front.
- `R` turns the right half a quarter turn clockwise (looking at the right face). `R'` is counterclockwise, `R3` is
  three quarter turns, `R'2` two counterclockwise. On this puzzle `R4` is **not** the identity: the ring has only
  turned 180°, so `R'` is not `R3`.

## Search options

MITM candidate windows and caps use HTM after merging the forward path, backward path, and preceding solution.
The C probe handles cascading cancellations before pruning; returned witnesses are checked against the puzzle
model. This uses the existing tables and does not require rebuilding them.

On 24 paired random states (seed 20261111), with the **small rung-7 table** and otherwise default settings,
merged-cost selection reduced mean complete solution length from 135.75 to 134.33 HTM. Ten solutions shortened,
nine tied, and five lengthened. Mean solve time was 33.8 s versus 37.7 s; the slowest paired trial was 47.2 s
versus 57.0 s. A fresh-process repeat of the slowest trial took 51.6 s including compilation and table loading.
These are sample measurements, not a worst-case runtime guarantee. Reproduce the current solver run with
`RUNG7=small PYTHONPATH=. .venv/bin/python fulltab/beam_full.py 24 20261111`.

`--endgame-width 4` tries three additional rung-6 landings through rungs 7 and 8, keeping the successful original
continuation as a candidate. This trades more search time for shorter complete solutions. Finishes are now compared
after merging with the preceding moves, including all invisible finish alternatives. From Python:
`ladder.Ladder(endgame_width=4)`. The default width remains 1.

Existing first-stage relabelling files are checked on load. The old generator inverted SymPy's right transversal;
on the existing build, its 1,296 representatives covered only 500 distinct target cosets. The loader repairs that
direction in memory, and the generators now write and verify the correct representatives. Large tables do not need
rebuilding.

`--first-stage direct` is an experimental IDA* search for rungs 1+2 using visible piece coordinates and approximately
97 MB of pattern tables, built in seconds. It uses the same physical goal, returning a legal relabelled state for
rung 3. It has a per-input search budget; exhaustion can produce fewer candidates. It is primarily a time/storage
improvement, not a way to beat an optimal MITM entry. `python firststage.py --count 100` benchmarks it alone.

For full benchmarks, `fulltab/beam_full.py` accepts `ENDGAME_WIDTH` and `FIRST_STAGE` in the
environment. The original benchmark numbers above describe the original implementation, not these options.

## Checking a puzzle ("is mine assembled wrong?")

```bash
.venv/bin/python solve.py --check mystate.txt    # reachable? if not, which invariant fails
```

A reachable puzzle satisfies
- corner twists sum to 0 mod 3;
- the ring pieces (centres and edges) form an even permutation;
- gear spins sum to 0 mod 4.

Anything else (a single twisted gear, two swapped ring pieces, a twisted corner) means it was reassembled wrong.

## Scrambles

```bash
.venv/bin/python scramble.py --random-state  # random-state scramble: a uniformly random state, solved and inverted
.venv/bin/python scramble.py                 # random-move scramble, 100 moves
```

Both print the scramble and a text net of the resulting state. Options: `-n 5` for five scrambles, `--seed 1` for
reproducibility, `--random-move 150` for a longer random-move scramble.

**Random-state** scrambles are exact: a uniformly random state is drawn, solved, and the inverted solution is the
scramble. **Random-move** scrambles turn U, R or F (never the face just turned) by a single or double turn in either
direction, 8 choices at every step; they need no tables. 100 moves is indistinguishable from uniform by the
statistical test in `validate_random_move.py` (uniform reference states from the exact sampler in `model.py`), 50 is
borderline, 25 is detectably non-uniform.

### Reading the net

Each face is a 5×5 block, arranged as in kingdeim's app: U on top; L F R in the middle row; D below F; B below D,
upside down.

```
c g e g c      c = corner sticker colour     e = edge-slot sticker
g . . . g      g = gear: colour + spin digit (0 = pointing at its edge, 1-3 = quarter turns)
e . m . e      m = centre-slot sticker
```

Colours: W white (U), Y yellow (D), G green (F), B blue (B), R red (L), P purple (R).

## The megaminx aside

`megaminx.py` and `analyze_megaminx.py` measure whether the WCA megaminx scramble (7 lines of `R++`/`D++` moves and
`U`, 77 moves) is biased in piece positions, against exact uniform positions; the result (`analyze_megaminx.out`)
is that it is slightly off.

## What's here

| file | |
| --- | --- |
| `ladder.py` | the solver: the beam over the rungs |
| `fulltab/` | the tables: building them (`build.py` and the scripts it runs), the table rungs (`table.py`), the kernels (`ft.c`, `ftlib.py`), benchmarks |
| `coset.py`, `csrc/coset.c` | coset labels (stabiliser chains) and rung 6's complete table |
| `model.py` | discrete model: moves, notation parser, uniform random states, rotation normalisation, invariant check, text net |
| `solve.py` | solving, state display and the reachability check |
| `groupfacts.py` | group orders, invariants and subgroup facts (sympy) |
| `oracle.py` | float-matrix port of kingdeim's simulator, used only by the tests |
| `tests/test_model.py` | model vs oracle (10,000 sequences), generator identities, all published algorithms |
| `scramble.py`, `validate_random_move.py` | the scramblers and the statistical test of random-move scrambles |
| `megaminx.py`, `analyze_megaminx.py` | the megaminx scramble measurement |
| `maltese.tws`, `maltese*.kpuzzle.json` | the puzzle for twsearch / twips |

Run the tests with `.venv/bin/python tests/test_model.py`.
