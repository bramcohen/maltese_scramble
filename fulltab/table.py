"""The table rungs as solver objects: a backward table (classes of the target's right cosets with the cheapest cost and a
witness) plus a forward list (elements of the starting group with their moves); a probe meets the two, and the move
sequence of a hit is the forward element's moves followed by the inverse of the table representative's.  Every
returned sequence is verified by label.  All arrays are memory-mapped files under fulltab/data/mm/ (built once).

Rungs 1+2: BallTable (Z-cosets of the 9-HTM ball) + BallForward (the 4-HTM ball), 1,296 renamings.
Rung 3:    ChainedTable (Y2-cosets to 20 HTM) + Forward (Z elements to 11 HTM).
Rung 4:    ChainedTable (H1'-cosets to 28 HTM) + Forward (Y elements to 4 HTM), 9,216 renamings.
Rung 5:    ChainedTable (H1-cosets to 28 HTM) + Forward (H1' elements to 9 HTM).
Rung 7:    SlicedTable (F16-cosets to 21 HTM, front/back-mirror canonical) + Forward (A1 elements to 10 HTM), 256 renamings."""
import os, numpy as np, coset as CS, model as M, ladder as L
import ftlib as F
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GROUP = {"3": None, "4": "Y", "5": "H1p", "7": "A1"}           # each rung's starting group (rung 3's, Z, is given by positions)
RUNS = L.RUNS; inv_run = F._inv_run

def group_spec(r):
    if r != "3": return CS.Spec(canon=GROUP[r])
    isUD = np.zeros(18, np.int8); isUD[L.UDE] = 1; isEQ = np.zeros(24, np.int8); isEQ[L.EQG] = 1
    return CS.Spec(pos={"r": (np.ones(18), isUD), "g": (np.ones(24), isEQ)}, fun={"c": (np.zeros(8, np.int8), np.eye(8, dtype=np.int8))})

def memmapped(name, build):
    """The array `name` as a read-only memory map under fulltab/data/mm/ (page cache, shared between processes),
    built by build() the first time."""
    f = os.path.join(D, "mm", name + ".npy")
    if not os.path.exists(f):
        os.makedirs(os.path.dirname(f), exist_ok=True); np.save(f + ".tmp.npy", np.ascontiguousarray(build())); os.replace(f + ".tmp.npy", f)
    return np.load(f, mmap_mode="r")

class Forward:
    """Every element of rung r's starting group up to E HTM, as products p·q⁻¹ of states of the ball of depth ceil(E/2)
    that share a coset of the group, with the moves of each (fulltab/ball.py provides the balls)."""
    def __init__(self, r, E):
        hb = (E + 1) // 2; hf = max(hb, 7); self.h = hf; self.W = F.Witness(hf, D); self.E = E       # the smallest ball on disk that is deep enough
        tag = f"fwd_rung{r}_E{E}"; built = {}
        def build():
            if not built: built.update(self._build(r, E, hb, hf))
            return built
        self.maps = memmapped(tag + "_maps", lambda: build()["maps"]); self.cost = memmapped(tag + "_cost", lambda: build()["cost"])
        self.wp = memmapped(tag + "_wp", lambda: build()["wp"]); self.wq = memmapped(tag + "_wq", lambda: build()["wq"])
    def _build(self, r, E, hb, hf):
        ST = np.load(os.path.join(D, f"ball{hf}.npy"), mmap_mode="r"); C = np.load(os.path.join(D, f"ball{hf}_cost.npy"))
        n = int((C <= hb).sum()); ST = np.ascontiguousarray(ST[:n]); C = np.ascontiguousarray(C[:n])
        spG = group_spec(r)
        glab = np.concatenate([spG.keys(np.ascontiguousarray(ST[c:c + 5_000_000])) for c in range(0, len(ST), 5_000_000)])
        o = np.argsort(glab, kind="stable"); sl = glab[o]; cut = np.flatnonzero(sl[1:] != sl[:-1]) + 1
        gst = np.r_[0, cut, len(o)].astype(np.int64); S = np.ascontiguousarray(ST[o]); Cs = np.ascontiguousarray(C[o]); del ST, glab, sl
        ec, ew, st = F.elems50(S, Cs, gst, E, 30); EM = F.pair_maps50(S, ew); del S
        oo = np.argsort(ec, kind="stable"); ew = ew[oo]
        return dict(maps=EM[oo], cost=ec[oo], wp=o[(ew >> np.uint64(32)).astype(np.int64)], wq=o[(ew & np.uint64(0xffffffff)).astype(np.int64)])
    def moves(self, i): return self.W.element(int(self.wp[i]), int(self.wq[i]))

class ChainedTable:
    """A table built by fulltab/chain.py: the join of the 9-HTM ball to 17 HTM, chained by the rung's pieces to maxcost.
    Files full_rung{r}_chain{maxcost}{suffix}.npy and reps_rung{r}_chain{maxcost}{suffix}.npz."""
    def __init__(self, r, maxcost, target, suffix=""):
        self.r = r; self.spT = CS.Spec(canon=target); zf = os.path.join(D, f"reps_rung{r}_chain{maxcost}{suffix}.npz"); tag = f"rung{r}_chain{maxcost}{suffix}"
        self.tab = memmapped(tag + "_tab", lambda: np.load(os.path.join(D, f"full_rung{r}_chain{maxcost}{suffix}.npy")))
        for k in ("rows", "DER", "bwp", "bwq"): setattr(self, k, memmapped(f"{tag}_{k}", lambda k=k: np.load(zf)[k]))
        z = np.load(zf); self.W = F.Witness(int(z["ball"]), D); self.pseq = np.load(os.path.join(D, str(z["pieces"])))["seq"]
        self.B = 22; self.bidx = F.bucket_index(self.tab, self.B); self.tmax = int((self.tab & F.CM).max()); self.mir = False
    def moves(self, ti):
        """moves of the representative of table entry ti (from solved)"""
        row = int(self.rows[ti]); out = []
        while self.DER[row] != -1: d = int(self.DER[row]); out.append(str(self.pseq[d & ((1 << 20) - 1)])); row = d >> 20
        base = self.W.element(int(self.bwp[row]), int(self.bwq[row]))
        return base + " ".join(out[::-1]).split()


class SlicedTable:
    """A table built from the ball with generated levels: two levels (fulltab/slice2.py + merge2.py, ball h-2; rung 7's
    large table, h = 11, 21 HTM; witnesses refer to the ball and to the level-(h-1) parents in data/t{h}_{r}/) or one
    level (fulltab/slicebuild.py, ball h-1; the small table, h = 10, 20 HTM)."""
    def __init__(self, r, name, h, target, levels=2):
        self.r = r; self.spT = CS.Spec(canon=target)
        self.tab = memmapped(name + "_tab", lambda: np.load(os.path.join(D, name + ".npy")))
        self.wp = memmapped(name + "_wp", lambda: np.load(os.path.join(D, name + "_wit.npz"))["p"]); self.wq = memmapped(name + "_wq", lambda: np.load(os.path.join(D, name + "_wit.npz"))["q"])
        self.W = F.Witness2(h - 2, os.path.join(D, f"t{h}_{r}"), D) if levels == 2 else F.Witness(h - 1, D); self.mir = name.endswith("m")
        self.B = 24; self.bidx = F.bucket_index(self.tab, self.B); self.tmax = int((self.tab & F.CM).max())
    def moves(self, ti): return self.W.element(int(self.wp[ti]), int(self.wq[ti]))

class BallForward:
    """Forward list for rungs 1+2: every state of the ball up to E HTM (all moves), with its moves from solved."""
    def __init__(self, E):
        hf = max(E, 7); self.E = E; self.W = F.Witness(hf, D)
        ST = np.load(os.path.join(D, f"ball{hf}.npy"), mmap_mode="r"); C = np.load(os.path.join(D, f"ball{hf}_cost.npy")); n = int((C <= E).sum())
        self.maps = memmapped(f"fwd_rung12_E{E}_maps", lambda: F.maps50(np.ascontiguousarray(ST[:n]))); self.cost = np.ascontiguousarray(C[:n])
    def moves(self, i): return self.W.seq(int(i))

class BallTable:
    """Backward table for rungs 1+2: Z-coset labels of the h-HTM ball (fulltab/ztable.py), witness = ball row."""
    def __init__(self, h):
        self.r = "12"; self.spT = CS.Spec(canon="Z"); self.tab = memmapped(f"rung12_b{h}_tab", lambda: np.load(os.path.join(D, f"full_rung12_b{h}.npy")))
        self.wit = memmapped(f"rung12_b{h}_wit", lambda: np.load(os.path.join(D, f"full_rung12_b{h}_wit.npy"))); self.W = F.Witness(h, D)
        self.B = 24; self.bidx = F.bucket_index(self.tab, self.B); self.tmax = int((self.tab & F.CM).max()); self.mir = False
    def moves(self, ti): return self.W.seq(int(self.wit[ti]))

class FullRung:
    """A rung for the ladder's beam: solve_many(states, window) returns, per input, the candidates (merged HTM, words,
    [], landing state) within `window` of its best, ranked by merged length, one per landing state, every sequence
    verified by label.  Words are single runs, so the beam's merging works across them."""
    def __init__(self, table, fwd, max_candidates=200):
        self.table = table; self.fwd = fwd; self.spT = table.spT; self.kT = self.spT.keys(CS.pack_states([M.State()]))[0]
        self.max_candidates = max_candidates; self.candidates = []; self.hits = 0; self.bad_witnesses = 0
    def _cands(self, hits, src, fwd, window, cap):
        """Candidates from hit rows (input index, element, table index, forward cost, table cost); src(xi) is the state
        the sequence is applied to."""
        if len(hits) == 0: return []
        tot = hits[:, 3] + hits[:, 4]; best = int(tot.min()); seen = set(); landed = set(); out = []
        for xi, ei, ti, ec, c in hits[np.argsort(tot, kind="stable")]:
            if ec + c > best + window or len(out) >= cap: break
            try: seq = fwd.moves(int(ei)) + [inv_run(w) for w in self.table.moves(int(ti))[::-1]]
            except (IndexError, ValueError): self.bad_witnesses += 1; continue
            key = (int(xi), " ".join(seq))
            if key in seen: continue
            seen.add(key); s2 = M.apply_seq(src(int(xi)), key[1])
            if self.spT.keys(CS.pack_states([s2]))[0] != self.kT: continue
            k2 = s2.key()                                   # one candidate per landing state (many sequences reach the same one)
            if k2 in landed: continue
            landed.add(k2); out.append((L.htm(" ".join(seq)), seq, [], s2))
        out.sort(key=lambda c: c[0]); return out[:cap]
    def _probe_groups(self, xs, groups, fwd, window, hcap=1 << 20):
        """One pass over the forward list for many packed inputs.  groups[g] = indices into xs that belong to input g
        (several copies of one state).  Returns per group the hit rows (input column = index into xs) with cost
        <= that group's best + window; None for a group with no hit at the full bound.  Many copies are probed from
        a tight bound upwards so the hit cap never drops the cheapest; the kernel keeps only hits within the window
        of each input's running best."""
        full = fwd.E + self.table.tmax; out = [None] * len(groups); todo = list(range(len(groups)))
        for bound in ((full - 8, full - 4, full) if len(xs) > 64 else (full,)):
            if not todo: break
            idx = np.concatenate([groups[g] for g in todo]); back = np.empty(len(xs), np.int64); back[idx] = np.arange(len(idx))
            gid = np.concatenate([np.full(len(groups[g]), g) for g in todo])
            h = F.probe50_multi(fwd.maps, fwd.cost, fwd.E, xs[idx], self.table.tab, self.table.bidx, self.table.B, self.spT, bound=bound, hcap=hcap, mir=self.table.mir, window=window)
            if len(h) >= hcap and len(todo) > 1:               # cap reached: redo this bound one group at a time
                for g in list(todo):
                    r = self._probe_groups(xs, [groups[g]], fwd, window, hcap)[0]
                    if r is not None: out[g] = r
                return out
            h[:, 0] = idx[h[:, 0]]; hg = gid[back[h[:, 0]]]; tot = h[:, 3] + h[:, 4]; still = []
            for g in todo:
                m = hg == g
                if m.any() and (int(tot[m].min()) + window <= bound or bound == full): out[g] = h[m]
                elif bound < full: still.append(g)
            todo = still
        return out
    def solve_many(self, states, window=0, max_candidates=None):
        cap = max_candidates or self.max_candidates; xs = CS.pack_states(states); groups = [np.array([i]) for i in range(len(states))]
        hits = self._probe_groups(xs, groups, self.fwd, window, hcap=len(states) << 20); self.hits = sum(len(h) for h in hits if h is not None)
        return [self._cands(h, lambda xi, st=st: st, self.fwd, window, cap) if h is not None else [] for h, st in zip(hits, states)]
    def solve(self, state, window=0, max_candidates=None):
        self.candidates = self.solve_many([state], window, max_candidates)[0]
        return self.candidates[0] if self.candidates else None

class RelabelRung(FullRung):
    """A rung solved up to relabelling: the input is renamed every way the invisible subgroup allows (reps_file holds
    a right transversal of S ∩ target in S ∩ starting group, as states; fulltab/treps*.py) and every
    copy is probed at once; the candidates carry the renamed landing state, which the beam continues from.
    keep_in: a spec whose identity class a copy must lie in to hit the table at all (rung 4's table is built from Y,
    its inputs may lie in either coset of Y2).  alternatives: the inputs of one call are alternatives (orientations)
    and a retry happens only when all of them miss.  prefix_retry: the retry is a shoved-in first move followed by
    the same probe, cheapest moves first; deeper: forward lists tried as a last resort."""
    def __init__(self, table, fwd, reps_file, max_candidates=200, keep_in=None, alternatives=False, prefix_retry=False, deeper=()):
        super().__init__(table, fwd, max_candidates)
        rows = CS.right_transversal(np.load(reps_file), self.spT)
        self.reps = [CS.unpack_state(r) for r in rows]
        self.keep_in = keep_in; self.kin = keep_in.keys(CS.pack_states([M.State()]))[0] if keep_in is not None else None
        self.alternatives = alternatives; self.prefix_retry = prefix_retry; self.deeper = list(deeper)
    def _copies(self, states):
        ys = [k.apply(e) for st in states for e in [st.as_element()] for k in self.reps]; xs = CS.pack_states(ys); R = len(self.reps)
        groups = [np.arange(i * R, (i + 1) * R) for i in range(len(states))]
        if self.keep_in is not None:
            ok = self.keep_in.keys(xs) == self.kin; groups = [g[ok[g]] for g in groups]
        return ys, xs, groups
    def solve_many(self, states, window=0, max_candidates=None):
        cap = max_candidates or self.max_candidates; res = [[] for _ in states]; self.hits = 0
        ys, xs, groups = self._copies(states)
        for g, h in enumerate(self._probe_groups(xs, groups, self.fwd, window)):
            if h is not None: self.hits += len(h); res[g] = self._cands(h, lambda xi: ys[xi], self.fwd, window, cap)
        if any(res) and self.alternatives: return res
        todo = [g for g in range(len(states)) if not res[g]]
        if todo and self.prefix_retry:
            for w in sorted(RUNS, key=L.word_htm):
                yp, xp, gp = self._copies([M.apply_seq(st, w) for st in states]); found = False
                for g, h in zip(todo, self._probe_groups(xp, [gp[g] for g in todo], self.fwd, window)):
                    if h is None: continue
                    self.hits += len(h); c = self._cands(h, lambda xi: yp[xi], self.fwd, window, cap)
                    res[g] = sorted(((L.htm(" ".join([w] + seq)), [w] + seq, rest, land) for _, seq, rest, land in c), key=lambda x: x[0]); found = found or bool(res[g])
                if found and self.alternatives: return res
                todo = [g for g in todo if not res[g]]
                if not todo: return res
        for fwd in self.deeper:
            if not todo or (self.alternatives and any(res)): break
            still = []
            for g, h in zip(todo, self._probe_groups(xs, [groups[g] for g in todo], fwd, window)):
                if h is None: still.append(g); continue
                self.hits += len(h); res[g] = self._cands(h, lambda xi: ys[xi], fwd, window, cap)
            todo = still
        return res

def full_rungs(rung7=None, first_stage="table"):
    """The five table rungs with the tables and depths of the final configuration.  rung7: "small" = the 20-HTM table
    (the default build, 1.9 h, 9 GB) with an 11-HTM forward; "large" = the 21-HTM table (13 h, 31 GB) with a 10-HTM
    forward, 1.3 HTM and 7 s per solve better; None = large if it is on disk, else small."""
    if first_stage not in ("table", "direct"):
        raise ValueError('first_stage must be "table" or "direct"')
    reps = lambda name: os.path.join(D, f"T_reps_{name}.npy")
    if rung7 is None: rung7 = "large" if os.path.exists(os.path.join(D, "mm", "full_rung7_t11m_tab.npy")) else "small"
    r7 = (SlicedTable("7", "full_rung7_t11m", 11, "F16"), Forward("7", 10)) if rung7 == "large" else (SlicedTable("7", "full_rung7_s10m", 10, "F16", levels=1), Forward("7", 11))
    if first_stage == "direct":
        from firststage import DirectFirstStage
        first = DirectFirstStage()
    else:
        first = RelabelRung(BallTable(9), BallForward(4), reps("12"), alternatives=True, prefix_retry=True, deeper=[BallForward(5), BallForward(6)])
    return {"12": first,
            "3": FullRung(ChainedTable("3", 20, "Y2", "y2"), Forward("3", 11)),
            "4": RelabelRung(ChainedTable("4", 28, "H1p"), Forward("4", 4), reps("4_y2"), keep_in=CS.Spec(canon="Y")),
            "5": FullRung(ChainedTable("5", 28, "H1"), Forward("5", 9)),
            "7": RelabelRung(r7[0], r7[1], reps("7"))}
