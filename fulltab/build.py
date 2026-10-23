"""Build everything the solver needs from the puzzle model alone, in order, skipping the steps whose outputs exist
(LADDER.md, "Building the tables"): the subgroup chains, the ball, the finish table, the pieces, the renamings, the rung tables.
Usage, from the repository root:  PYTHONPATH=. .venv/bin/python fulltab/build.py [--rung7 small|large|both]
small (the default): the 20-HTM rung-7 table (about 1.9 h, 50 GB of memory); large: the 21-HTM one (about 13 h, 1.1 TB
of disk while building, 60 GB of memory), 1.3 HTM and 7 s per solve better; the solver uses it whenever it is present.  Run under fulltab/guard.sh for memory safety; the long steps resume from their files on disk."""
import argparse, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); D = os.path.join(HERE, "data"); T = os.path.join(ROOT, "tables")
PY = sys.executable; ENV = {**os.environ, "PYTHONPATH": ROOT}
def run(label, cmd, outputs, env=None):
    if all(os.path.exists(o) for o in outputs):
        print(f"{label}: present", flush=True); return
    t = time.time(); print(f"{label}: {' '.join(cmd)}", flush=True)
    subprocess.check_call([PY] + cmd, cwd=ROOT, env={**ENV, **(env or {})})
    print(f"{label}: done in {(time.time() - t) / 60:.0f} min", flush=True)
ap = argparse.ArgumentParser(); ap.add_argument("--rung7", default="small", choices=["small", "large", "both"]); a = ap.parse_args()
d = lambda *p: os.path.join(D, *p); t = lambda p: os.path.join(T, p)
run("subgroup chains", ["fulltab/mkbsgs.py"], [t(f"bsgs_{n}.pkl") for n in ("Y", "H1p", "H1", "A1", "F16", "Z", "Y2")] + [t("invisible_gens.pkl")])
run("ball 7", ["fulltab/ball.py", "7"], [d("ball7.npy"), d("ball7_parents.npz")])
run("ball 9", ["fulltab/ball.py", "9"], [d("ball9.npy"), d("ball9_parents.npz")])
for h, r in (("5", "3"), ("6", "4"), ("7", "5")): run(f"pieces of rung {r}", ["fulltab/pieces.py", h, r], [d(f"pieces_rung{r}_h{h}.npz")])   # excursions to 2h HTM: the chains use L = 10, 12, 14
run("rung 8 finish table", ["fulltab/finish.py"], [t("finish_full.pkl.gz")])
run("renamings of rungs 1+2 and 6", ["fulltab/treps.py"], [d("T_reps_12.npy"), d("T_reps_6.npy")])
run("renamings of rungs 7 and 8", ["fulltab/treps2.py"], [d("T_reps_7.npy"), d("T_reps_8.npy")])
run("renamings of rung 4", ["fulltab/treps_y2.py"], [d("T_reps_4_y2.npy")])
run("rungs 1+2 table", ["fulltab/ztable.py", "9"], [d("full_rung12_b9.npy"), d("full_rung12_b9_wit.npy")])
run("rung 3 table", ["fulltab/chain.py", "3", "9", "10", "20"], [d("full_rung3_chain20y2.npy"), d("reps_rung3_chain20y2.npz")], env={"CHAIN_TARGET": "Y2"})
run("rung 4 table", ["fulltab/chain.py", "4", "9", "12", "28", "30"], [d("full_rung4_chain28.npy"), d("reps_rung4_chain28.npz")])
run("rung 5 table", ["fulltab/chain.py", "5", "9", "14", "28", "30"], [d("full_rung5_chain28.npy"), d("reps_rung5_chain28.npz")])
if a.rung7 in ("large", "both"):
    run("rung 7 table, joins (resumable)", ["fulltab/slice2.py", "7", "9", "128", "1", "29"], [d("t11_7", "out_s127.npz")])
    run("rung 7 table, merge (resumable)", ["fulltab/merge2.py", "7", "9", "1", "64"], [d("mm", "full_rung7_t11m_tab.npy")])
if a.rung7 in ("small", "both"):
    run("rung 7 small table", ["fulltab/slicebuild.py", "7", "10", "64", "1", "30"], [d("full_rung7_s10m.npy"), d("full_rung7_s10m_wit.npz")])
print("forward lists and rung 6's table are built on the solver's first use (minutes); then: PYTHONPATH=. .venv/bin/python fulltab/beam_full.py 3", flush=True)
