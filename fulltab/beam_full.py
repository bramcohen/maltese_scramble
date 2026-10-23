"""Whole solves on random states: mean HTM, seconds, per-rung costs and seconds, misses and retries.
argv: n states [seed=6].  Env: PARAMS_JSON per-rung overrides, e.g. {"3": {"keep": 4}}; RAW=1 for the raw run
(one orientation, keep 1 everywhere, no retry of any kind: the base of the beam measurements)."""
import sys, os, time, json, resource, numpy as np, model as M, ladder as L
n = int(sys.argv[1]) if len(sys.argv) > 1 else 24; seed = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("SEED", "6"))
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
PV = json.loads(os.environ.get("PARAMS_JSON", "{}")); RAW = os.environ.get("RAW") == "1"
if RAW: PV = {**{r: {"keep": 1} for r in ("12", "3", "4", "5")}, **PV}
t0 = time.time(); lad = L.Ladder(params=PV, retries=not RAW, rung7=os.environ.get("RUNG7"))
if RAW: lad.m12.prefix_retry = False; lad.m12.deeper = []
print(f"ladder ready ({time.time() - t0:.0f}s, peak RSS {rss():.1f} GB); overrides {PV}{' (raw run)' if RAW else ''}", flush=True)
rng = np.random.default_rng(seed); states = [M.random_state(rng) for _ in range(n)]
tots, per, times, TIME, FAILS, SPARE, WIDEN = [], {}, [], {}, [], 0, 0
for i, st in enumerate(states):
    t = time.time()
    try: tot, plan = lad.solve(st, frames=1 if RAW else 3, max_restarts=0 if RAW else 3)
    except RuntimeError as e: print(f"  state {i}: FAILED ({e}) in {time.time() - t:.0f}s", flush=True); FAILS.append(i); times.append(time.time() - t); continue
    dt = time.time() - t; tots.append(tot); times.append(dt)
    for r, ws in plan: per.setdefault(r, []).append(L.htm(" ".join(ws)))
    print(f"  state {i}: {tot} HTM in {dt:.0f}s; per rung " + " ".join(f"{r}:{L.htm(' '.join(ws))}" for r, ws in plan), flush=True)
    sec = {}
    for e in lad.misslog:
        if e[0] in ("spare", "widen"): SPARE += e[0] == "spare"; WIDEN += e[0] == "widen"
        elif e[0] != "rung6": sec[e[0]] = sec.get(e[0], 0.0) + e[5]
    for r, v in sec.items(): TIME[r] = TIME.get(r, 0.0) + v
    print("     seconds per rung: " + " ".join(f"{r}:{v:.1f}" for r, v in sorted(sec.items())), flush=True)
print(f"failed {len(FAILS)} of {n}; spare landings used {SPARE} times, widenings {WIDEN}", flush=True)
print("mean seconds per rung: " + ", ".join(f"{r}: {v / max(len(tots), 1):.1f}" for r, v in sorted(TIME.items())), flush=True)
print(f"mean {np.mean(tots):.2f} HTM, mean {np.mean(times):.0f}s per scramble (max {max(times):.0f}s); per rung: " + ", ".join(f"{r}: {np.mean(v):.2f}" for r, v in sorted(per.items())), flush=True)
