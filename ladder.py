"""The random-state solver: a ladder of subgroups, each rung a meet in the middle over a precomputed table.

    everything --1+2--> Z --3--> Y2 --4--> H1' --5--> H1 --6--> A1 --7--> F16 --8--> solved (up to invisibility)

Z: U/D edges on U/D slots, equatorial gears on equatorial slots, corners untwisted.  Y2: Y = <U, R2, F2, c, the 24
six-HTM words> together with its parity-swapped coset (the eye cannot tell them apart).  H1' = <H1, 24 words>,
H1 = <U, R2, F2>, A1 = <U, R4, F4, R2 U2 R2>, F16 = the 393,216-element group of the finish.

Every rung is solved *up to relabelling*: the invisible subgroup (face-centre spins, permutations of same-coloured
gears) meets each rung's starting group in a set of renamings, the rung renames its input every allowed way and
probes all copies at once, and the whole solve ends in an invisible state rather than the identity.  Tables and
forward lists live in fulltab/ (fulltab/table.py); rung 6 is a complete lookup table (coset.CostMITM); the finish is
a table over F16.  The solve is a beam over the rungs (Ladder.solve)."""
import itertools, os, pickle, time
import numpy as np
import coset as CS
import model as M

UDE = [M.RING_NAMES.index(n) for n in ("UF", "UB", "UR", "UL", "DF", "DB", "DR", "DL")]
EQG = [M.GEAR_NAMES.index(n) for n in ("FR", "FL", "BR", "BL", "RF", "RB", "LF", "LB")]
RUNS = [f + a for f in "URF" for a in ("", "2", "3", "4", "'", "'2", "'3")]                 # the 21 single-face runs
H1_RUNS = [(n, M.element(M.parse(n)), len(M.parse(n)), f) for f, ns in
           enumerate((("U", "U2", "U3", "U'", "U'2", "U'3", "U4"), ("R2", "R4", "R'2"), ("F2", "F4", "F'2"))) for n in ns]
HERE = os.path.dirname(os.path.abspath(__file__))
MAXSAV6 = 4          # the most merging can save at a rung boundary (R4 then R4): rung-6 renamings costlier than best + this cannot win
TOP6 = 12            # rung 6: path reconstructions per beam entry (the cheapest by table cost over frames x renamings)


class Ladder:
    """Per-rung parameters (PARAMS; override with Ladder(params={"3": {"keep": 4}})):
      window   a rung keeps every solution within this many HTM of its best as candidates
      keep     how many of the rung's landings (cheapest by total HTM so far, distinct by the next rung's class) go on
    Beam policy: rungs 3 and 7 run once per solve and are retried only when they find nothing (the previous rung's
    spare landings, then its wider window, then a restart of rungs 1 + 2); rungs 1 + 2 try the three top/bottom
    orientations and, when all miss, shoved-in first moves; the cheap rungs carry many landings."""

    PARAMS = {
        "12": dict(window=1, keep=1),
        "3": dict(window=2, keep=8),
        "4": dict(window=2, keep=16),
        "5": dict(window=6, keep=100),
        "6": dict(window=0, keep=1, lookup=True),
        "7": dict(window=6, keep=50),
    }

    def __init__(self, threads=16, verbose=False, params=None, retries=True, rung7=None, first_stage="table", endgame_width=1):
        if first_stage not in ("table", "direct"):
            raise ValueError('first_stage must be "table" or "direct"')
        if endgame_width < 1:
            raise ValueError('endgame_width must be positive')
        self.endgame_width = endgame_width
        self.verbose = verbose; self.threads = threads
        self.retries = retries                     # False: a rung that finds nothing fails the solve (the raw benchmark)
        self.params = {r: dict(v) for r, v in self.PARAMS.items()}
        for r, v in (params or {}).items():
            self.params[str(r)].update(v)
        t = time.time()
        # labels of the right cosets each rung's landings are pruned by (the class the next rung has to solve)
        self.spec2b = CS.Spec(canon="Y2"); self.spec3 = CS.Spec(canon="H1p"); self.spec4 = CS.Spec(canon="H1")
        self.specI1 = CS.Spec(canon="A1"); self.specI2 = CS.Spec(canon="F16")
        self.k4 = self.spec4.keys(CS.pack_states([M.State()]))[0]; self.kI1 = self.specI1.keys(CS.pack_states([M.State()]))[0]
        # rung 6: H1 -> A1 with the rung-6 runs; the table holds every class (1.3M), so a solve is a lookup
        self.mI1 = CS.CostMITM([(n, e, word_htm(n)) for n, e, c, f in H1_RUNS], self.specI1,
                               lambda st: self.specI1.keys(CS.pack_states([st]))[0] == self.kI1, back_cost=16, fwd_store=4,
                               threads=threads, verbose=verbose, cache=os.path.join(HERE, "tables", "cache", "rung6"),
                               max_cosets=10 ** 12, max_bytes=int(5e9), max_fwd=20_000_000, lean=True, compact=True)
        # rung 8: the cheapest finish from every element of F16
        self.f8 = _finish_table()
        # the table rungs (fulltab/table.py): 1+2, 3, 4, 5, 7
        import sys; sys.path.insert(0, os.path.join(HERE, "fulltab")); import table as FT
        full = FT.full_rungs(rung7, first_stage=first_stage)
        self.m12 = full["12"]
        self.rungs = {"3": (full["3"], self.spec2b), "4": (full["4"], self.spec3), "5": (full["5"], self.spec4),
                      "6": (self.mI1, self.specI1), "7": (full["7"], self.specI2)}
        self._init_inner_frames()
        # invisible renamings of the lookup rungs (fulltab/treps.py): every acceptable coset is tried, the cheapest kept
        self.reps = {r: [CS.unpack_state(x) for x in np.load(os.path.join(HERE, "fulltab", "data", f"T_reps_{r}.npy"))] for r in ("6", "8")}
        self.misslog = []
        if verbose:
            print(f"ladder ready ({time.time() - t:.0f}s)", flush=True)

    # ------------------------------------------------------------------ the beam

    def solve(self, state, frames=3, max_restarts=3):
        """(HTM, plan) for a state: plan = [(rung, words)], the words in the original frame, verified to leave the
        puzzle visibly solved.  A restart re-solves rungs 1 + 2 with a wider window, skipping landings already tried."""
        self.misslog = []; self._tried12 = set()
        for attempt in range(max_restarts + 1):
            r = self._beam(state, frames, slack=attempt)
            if r is not None:
                return r
            if self.verbose:
                print(f"   restart {attempt + 1}: rungs 1+2 with window +{attempt + 1}, skipping the {len(self._tried12)} landings already tried", flush=True)
        raise RuntimeError("ladder: no solution after restarts")

    solve_beam = solve

    def _extend(self, r, beam, window, cap=None):
        """Every kept partial solution in `beam` extended by rung r's solutions within `window` (children)."""
        m, spec = self.rungs[r]; p = dict(self.params[r]); p["window"] = window; children = []
        misses = 0; t_ext = time.time()
        if not beam: return []
        many = None
        if not p.get("lookup"):            # table rungs: all inputs in one pass over the forward list
            prefixes=[self._prefix_in_frame(e[4],e[5],e[6]) if len(e)>5 and e[5] else e[4] for e in beam]
            many = m.solve_many([e[3] for e in beam], window=p["window"], max_candidates=cap or m.max_candidates,prefixes=prefixes)
        for bi, (h0, k, plan, st, flat, *inner) in enumerate(beam):
            fi, st_std = inner if inner else (0, st); renamed_already = False
            if p.get("lookup"):
                if fi == 0:
                    # rung 6 in every inner frame and renaming: all (frame, renaming) pairs keyed in one call, paths
                    # reconstructed for the cheapest few, the winner by the merged length of the whole sequence so far
                    best = None; pairs = []
                    for j, rot in enumerate(self.inner_rots):
                        sj = st if j == 0 else self._conj(rot, st); pairs.append((j, sj))
                        ej = sj.as_element(); pairs.extend((j, kk.apply(ej)) for kk in self.reps["6"])
                    tk = time.time(); cost, _, hit = m._table_cost(m.spec.keys(CS.pack_states([sj for j, sj in pairs]), threads=m.threads)); tk = time.time() - tk
                    nlk = 0; tl = time.time()
                    for i in [i for i in np.argsort(np.where(hit, cost, 99), kind="stable") if hit[i]][:TOP6]:
                        (j, sj), c = pairs[i], cost[i]
                        rj = m.lookup(sj); nlk += 1
                        if not rj:
                            continue
                        wj = rj[1] + rj[2]
                        if j:
                            wj = self.rename_inner(wj, j, st)
                        tot = htm(" ".join(flat + wj))
                        if best is None or tot < best[0]:
                            best = (tot, j, (rj[0], wj, [], rj[3]))
                    self.misslog.append(("rung6", len(pairs), round(tk, 2), nlk, round(time.time() - tl, 2)))
                    if best is None:
                        cands = []
                    else:
                        _, fi, res = best; cands = [res]; renamed_already = True
                else:
                    cands = sorted(self._lookups(r, m, st), key=lambda c: c[0])[:1]
            else:
                cands = many[bi] if p["window"] else many[bi][:1]
            if not cands: misses += 1
            for c in cands:
                w = c[1] + c[2]
                if fi and not renamed_already:   # words found in an inner frame: rename to the standard frame as we go
                    w = self.rename_inner(w, fi, st_std)
                st2_std = M.apply_seq(st_std, " ".join(w)) if fi else c[3]
                children.append((htm(" ".join(flat + w)), k, plan + [(r, w)], c[3], flat + w, fi, st2_std))
        self.misslog.append((r, len(beam), misses, window, cap or 0, round(time.time() - t_ext, 2)))      # inputs, misses, window, cap, seconds
        return children

    def _lookups(self, r, m, st):
        """Table lookups of st and of its invisible renamings; the results carry the renamed landing state."""
        e = st.as_element(); sts = [st] + [k.apply(e) for k in self.reps[r]]
        cost, _, hit = m._table_cost(m.spec.keys(CS.pack_states(sts), threads=m.threads))
        if not hit.any():
            return []
        best = int(cost[hit].min()); out = []
        for s_, c, h in zip(sts, cost, hit):
            if h and c <= best + MAXSAV6:
                res = m.lookup(s_)
                if res:
                    out.append(res)
        return out

    def _finish(self, st):
        """(cost, path from solved) of the cheapest finish over st and its invisible renamings."""
        return min(self._finishes(st), key=lambda c: c[0])

    def _finishes(self, st):
        """All stored invisible finishes; the shortest alone need not merge best with a prefix."""
        seen = {st.key()}; yield self.f8[st.key()]; e = st.as_element()
        for k in self.reps["8"]:
            key = k.apply(e).key()
            if key in seen:
                continue
            seen.add(key); c = self.f8.get(key)
            if c is not None:
                yield c

    def _best_finish(self, st, flat, fi=0, st_std=None):
        """Choose by the merged whole path, including each invisible finish and inner frame."""
        best = None
        for _, path in self._finishes(st):
            fin = [CS.inverse_word(w) for w in reversed(path)]
            if fi:
                fin = self.rename_inner(fin, fi, st_std)
            total = htm(" ".join(flat + fin))
            if best is None or total < best[0]:
                best = (total, fin)
        return best

    def _beam(self, state, frames, slack):
        P = self.params; t = time.time()
        nxt = {"2": self.spec2b, "3": self.spec3, "4": self.spec4, "5": self.specI1, "6": self.specI2}
        beam = []                                   # (HTM so far, frame, plan [(rung, words)], state in that frame, words flat)
        # rungs 1 + 2: the three top/bottom orientations at once; a restart widens the window by one HTM
        t12 = time.time(); rss = [rotate_state(state, k) for k in range(frames)]
        many12 = self.m12.solve_many(rss, window=P["12"]["window"] + slack)
        for k in range(frames):
            for c in many12[k]:
                w = c[1] + c[2]
                beam.append((htm(" ".join(w)), k, [("12", w)], c[3], w))
        k12 = lambda e: (e[1], int(nxt["2"].keys(CS.pack_states([e[3]]))[0]))         # a restart must not hand rung 3 the same inputs again
        beam = [e for e in beam if k12(e) not in self._tried12]
        beam, spare = self._prune(beam, nxt["2"], P["12"]["keep"]); self._tried12.update(k12(e) for e in beam)
        self.misslog.append(("12", frames, 0, 0, 0, round(time.time() - t12, 2)))
        if self.verbose:
            print(f"   rungs 1+2: {len(beam)} kept, best {beam[0][0] if beam else None} HTM ({time.time() - t:.0f}s)", flush=True)
        prev_r, prev_inputs = None, None
        for r in ("3", "4", "5", "6", "7"):
            p = P[r]; inputs = beam; widened = 0
            while True:
                children = self._extend(r, beam, p["window"])
                if children:
                    break
                # no solution from any kept landing: the previous rung's spare landings; when those run out, the
                # previous rung re-solved with a wider window and many more candidates; then a restart
                if not self.retries: return None
                beam = list(itertools.islice(spare, max(1, len(beam)))); self.misslog.append(("spare", r, len(beam)))
                if r == "3": self._tried12.update(k12(e) for e in beam)
                if self.verbose:
                    print(f"   rung {r}: no solution; trying the next {len(beam)} partial solutions of the previous rung", flush=True)
                if not beam and prev_r is not None:
                    while not beam and widened < 4:
                        widened += 1; w = P[prev_r]["window"] + 4 * widened
                        self.misslog.append(("widen", r, widened))
                        kids = self._extend(prev_r, prev_inputs, w, cap=200 if widened == 1 else 5000)
                        seen = {e[3].key() for e in inputs}
                        kids = [e for e in kids if e[3].key() not in seen]
                        beam, spare = self._prune(kids, nxt[prev_r], 10 ** 6)
                        beam = beam[:max(1, p["keep"]) * 4]; spare = iter(beam[len(beam):])
                        if self.verbose:
                            print(f"   rung {r}: rung {prev_r} re-solved (window {w}): {len(kids)} new landings", flush=True)
                        inputs = inputs + beam
                    if children:
                        break
                if not beam:
                    return None
            if r == "7":
                # Preserve the successful incumbent, then spend extra search on other
                # rung-6 landings. A cheap prefix can have an expensive continuation.
                extra = list(itertools.islice(spare, max(0, self.endgame_width - len(beam))))
                if extra:
                    children += self._extend(r, extra, p["window"])
                best = None; t8 = time.time()
                for h, k, plan, st, flat, *inner in children:
                    fi, st_std = inner if inner else (0, st)
                    tot, fin = self._best_finish(st, flat, fi, st_std)
                    if best is None or tot < best[0]:
                        best = (tot, k, plan + [("8", fin)])
                tot, k, plan = best; self.misslog.append(("8", len(children), 0, 0, 0, round(time.time() - t8, 2)))
                plan = rename_plan(plan, k)
                seq = " ".join(w for name, ws in plan for w in ws)
                if not M.apply_seq(state, seq).is_visibly_solved():       # the end state is invisible, not the identity
                    raise RuntimeError("ladder: verification failed")
                if self.verbose:
                    print(f"   finish: {tot} HTM, frame {k} ({time.time() - t:.0f}s)", flush=True)
                return htm(seq), plan
            beam, spare = self._prune(children, nxt[r], p["keep"]); prev_r, prev_inputs = r, inputs
            if self.verbose:
                print(f"   rung {r}: {len(children)} extensions, kept {len(beam)}, best {beam[0][0]} HTM so far ({time.time() - t:.0f}s)", flush=True)

    def _prune(self, entries, spec_next, keep):
        """The `keep` cheapest entries by HTM so far, distinct by (frame, class of the next rung), and an iterator
        over the rest in the same order (spares, used only if the next rung finds nothing from the kept ones)."""
        seen = set()
        def distinct():
            for e in sorted(entries, key=lambda e: e[0]):
                key = (e[1], int(spec_next.keys(CS.pack_states([e[3]]))[0]))
                if key not in seen:
                    seen.add(key)
                    yield e
        it = distinct()
        return list(itertools.islice(it, keep)), it

    # ------------------------------------------------------------------ inner frames (rungs 6-8)

    def _prefix_in_frame(self, words, fi, end):
        """Walk a prefix backwards to express it in the next rung's frame."""
        rot=self.inner_rots[fi]; after=self._conj(rot,end); reverse=[]
        for word in reversed(simplify(' '.join(words))):
            before=M.apply_seq(end,CS.inverse_word(word)); rotated=self._conj(rot,before)
            for run in RUNS:
                if M.apply_seq(rotated,run)==after:
                    if word_htm(run)!=word_htm(word): raise RuntimeError('frame changes run cost')
                    reverse.append(run);break
            else: raise RuntimeError('cannot express prefix in inner frame')
            end,after=before,rotated
        return reverse[::-1]

    def _init_inner_frames(self):
        """The cube rotations that map H1 = <U, R2, F2> onto itself (8: the turns about the U/D axis and the flips
        that swap U and D).  After rung 5 the state is in H1, so rungs 6-8 can be solved in any of these frames with
        the same tables; each frame fixes a different corner pair for the inner chain."""
        def conj(r, st):
            return M.normalize_rotation(M.State().apply(r.inverse()).apply(st.as_element()).apply(r))
        k = self.k4; spec = self.spec4
        rots = [r for r in M.ALL_ROTATIONS
                if all(spec.keys(CS.pack_states([conj(r, M.apply_seq(M.State(), g))]))[0] == k for g in ("U", "R2", "F2"))]
        ident = [r for r in rots if M.State().apply(r) == M.State()]
        self.inner_rots = ident + [r for r in rots if r not in ident]
        self._conj = conj

    def rename_inner(self, words, fi, s_std):
        """Words of a solution found in inner frame fi, starting from the standard-frame state s_std, -> U/R/F
        words of the standard frame, found by walking both frames in step: for each move p of the frame solution,
        the standard-frame run g with conj(s . g) == s' . p is emitted (always one of the 21 single-face runs)."""
        r = self.inner_rots[fi]; s = s_std; sp = self._conj(r, s_std); out = []
        for w in words:
            toks = []
            for tok in w.split():
                sp2 = M.apply_seq(sp, tok)
                for g in RUNS:
                    s2 = M.apply_seq(s, g)
                    if self._conj(r, s2) == sp2:
                        break
                else:
                    raise RuntimeError("inner frame renaming: no equivalent standard-frame run")
                toks.append(g); s, sp = s2, sp2
            out.append(" ".join(toks))
        return out


F16_MOVES = [("U", 1), ("U'", 1), ("U2", 2), ("U'2", 2), ("U3", 3), ("U'3", 3), ("U4", 4),
             ("R4", 4), ("F4", 4), ("R'4", 4), ("F'4", 4)] + [(p, 8) for p in (
    "R2 U4 R2", "R2 U4 R'2", "R2 F4 R2", "R2 F4 R'2", "F2 U4 F2", "F2 U4 F'2", "F2 R4 F2", "F2 R4 F'2",
    "R'2 U'4 R'2", "R'2 U'4 R2", "R'2 F'4 R'2", "R'2 F'4 R2", "F'2 U'4 F'2", "F'2 U'4 F2", "F'2 R'4 F'2", "F'2 R'4 F2")]


def _finish_table():
    """Cheapest finish (HTM) from every element of F16: tables/finish_full.pkl.gz (fulltab/finish.py: per element the
    shorter of the Dijkstra over F16's moves and its shortest all-moves finish up to 18 HTM) if present, else the
    Dijkstra alone."""
    import gzip
    fp = os.path.join(HERE, "tables", "finish_full.pkl.gz")
    return pickle.load(gzip.open(fp, "rb")) if os.path.exists(fp) else finish_dijkstra()


def finish_dijkstra():
    """{state key: (HTM, moves from solved)} for every element of F16 over F16's own moves."""
    import heapq
    mv = [(n, M.element(M.parse(n)), word_htm(n)) for n, c in F16_MOVES]
    sol = M.State(); dist = {sol.key(): (0, [])}; pq = [(0, 0, sol.key(), sol)]; cnt = 0
    while pq:
        c, _, k, st = heapq.heappop(pq)
        if dist[k][0] < c:
            continue
        for n, e, cc in mv:
            s2 = st.apply(e); k2 = s2.key()
            if k2 not in dist or dist[k2][0] > c + cc:
                dist[k2] = (c + cc, dist[k][1] + [n]); cnt += 1; heapq.heappush(pq, (c + cc, cnt, k2, s2))
    return dist


def simplify(seq):
    """Merge consecutive turns of the same face into one run (U has order 8 on this puzzle, so counts are mod 8),
    dropping runs that cancel, repeatedly; returns run tokens like U, U'2, R4.  Works on a move string or a list
    of quarter-turn tokens."""
    toks = M.parse(seq) if isinstance(seq, str) else list(seq)
    runs = []                                            # (face, signed quarter turns)
    for t in toks:
        f, k = t[0], -1 if t.endswith("'") else 1
        if runs and runs[-1][0] == f:
            runs[-1][1] = (runs[-1][1] + k) % 8
            if runs[-1][1] == 0:
                runs.pop()
        else:
            runs.append([f, k % 8])
    out = []
    for f, k in runs:
        k = k - 8 if k > 4 else k                        # -3..4
        out.append(f + ("'" if k < 0 else "") + ("" if abs(k) == 1 else str(abs(k))))
    return out


HTM_OF_RUN = {1: 1, 2: 1, 3: 2, 4: 2}                      # single, half, half+single, two halves

# ---- free symmetry: the 120-degree whole-cube rotation r = x.y about the DBL corner keeps DBL fixed, conjugates
# U -> R -> F -> U and cycles the corners DFL -> DBR -> UBL.  Solving r^-k . x . r^k and renaming the moves back
# gives three candidate solutions per state (three frames, three choices of the inner chain's second fixed corner).
ROT = M.State().apply(M.ROTATIONS["x"]).apply(M.ROTATIONS["y"]).as_element()
ROT_INV = ROT.inverse()
RENAME_BACK = {"U": "F", "R": "U", "F": "R"}     # one step of r . m . r^-1


def rotate_state(state, k):
    """r^-k . state . r^k (the same physical cube seen in the frame rotated k times)."""
    st = state
    for _ in range(k % 3):
        st = M.State().apply(ROT_INV).apply(st.as_element()).apply(ROT)
    return st


def rename_plan(plan, k):
    """Map a solution found in frame k back to the original frame (apply the renaming k times)."""
    def ren(tok):
        f = tok[0]
        for _ in range(k % 3):
            f = RENAME_BACK[f]
        return f + tok[1:]
    return [(name, [" ".join(ren(t) for t in w.split()) for w in ws]) for name, ws in plan]


def htm(seq):
    """Half-turn-metric length after simplification: X and X2 (either direction) count 1; X3 and X4 count 2."""
    return sum(HTM_OF_RUN[int(t.lstrip("UDRLFB'") or 1)] for t in simplify(seq))


def word_htm(word):
    """HTM cost of one word on its own (no cross-word cancellation)."""
    return htm(word)


def check(state, plan):
    seq = " ".join(w for name, ws in plan for w in ws)
    return M.apply_seq(state, seq).is_visibly_solved(), len(M.parse(seq)), htm(seq)


def plan_htm(plan):
    """Per-rung HTM (each rung's words simplified on their own) and the overall HTM after simplifying across
    rung boundaries."""
    per = [(name, htm(" ".join(ws)) if ws else 0) for name, ws in plan]
    return per, htm(" ".join(w for name, ws in plan for w in ws))
