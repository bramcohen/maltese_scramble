"""Experimental direct visible-state search for rungs 1+2, without the ball or relabelling tables.

The admissible heuristic is the maximum of three kinds of exact pattern distances:
  * corner twists x positions of the eight U/D edges x their total spin parity;
  * positions of the eight U/D-coloured gears (goal: outside the equator);
  * positions of each side colour's four gears (goal: exactly two on the equator).
The remaining equatorial gear spin parity is checked at the goal. Moves and pattern
transitions are derived from model.py; all distances use the repository's HTM.
"""
import ctypes as ct
import hashlib
import itertools
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import model as M
import ladder as L

ROOT = Path(__file__).resolve().parent
NR = math.comb(18, 8) * 2
NC = 3 ** 6
RUNS = L.RUNS
MOVES = [M.element(M.parse(w)) for w in RUNS]
COST = np.array([L.word_htm(w) for w in RUNS], np.uint8)
EQ = sum(1 << i for i in L.EQG)
UD = sum(1 << i for i in L.UDE)


def masks(n, k):
    return np.sort(np.fromiter((sum(1 << i for i in c) for c in itertools.combinations(range(n), k)),
                              np.uint32, count=math.comb(n, k)))


def bitmask(a):
    return sum(1 << i for i in np.flatnonzero(a))


def _library():
    src = ROOT / 'csrc' / 'firststage.c'; lib = ROOT / 'build' / 'firststage.so'
    lib.parent.mkdir(exist_ok=True)
    if not lib.exists() or lib.stat().st_mtime < src.stat().st_mtime:
        tmp = str(lib) + '.tmp'
        subprocess.check_call(['cc', '-O3', '-std=c11', '-shared', '-fPIC', str(src), '-o', tmp])
        Path(tmp).replace(lib)
    dll = ct.CDLL(str(lib))
    ptr = ct.c_void_p; u32 = ct.c_uint32; i32 = ct.c_int
    dll.first_joint.argtypes = [ptr, ptr, ptr, u32, u32, ptr]; dll.first_joint.restype = u32
    dll.first_gears.argtypes = [ptr, ptr, u32, u32, i32, ptr, ptr]; dll.first_gears.restype = u32
    dll.first_search.argtypes = [ptr] * 8 + [u32] * 4 + [ptr, u32, i32, i32, ct.c_uint64, i32, ptr, ptr, ptr]
    dll.first_search.restype = i32
    return dll


def _p(a):
    return a.ctypes.data_as(ct.c_void_p)


def relabel_to_z(state):
    """Choose a legal invisible gear relabelling of a visibly-Z state, in linear time."""
    st = state.copy(); colors = M.GEAR_COLOR[st.gp]; eq = np.zeros(24, bool); eq[L.EQG] = True
    for color in range(6):
        for on_eq in (False, True):
            slots = np.flatnonzero((colors == color) & (eq == on_eq))
            pieces = np.flatnonzero((M.GEAR_COLOR == color) & (eq == on_eq))
            if len(slots) != len(pieces):
                raise ValueError('gear colours do not meet the Z target')
            st.gp[slots] = pieces
    if M._parity(st.gp) != M._parity(st.cp):
        # There are four U-coloured gears, all in the non-equatorial class.
        slots = np.flatnonzero(colors == 0)
        st.gp[slots[:2]] = st.gp[slots[1::-1]]
    if (np.any(st.co) or set(map(int, st.rp[L.UDE])) != set(L.UDE)
            or int(st.ro[L.UDE].sum()) % 2 or int(st.go[L.EQG].sum()) % 2 or M.check(st)):
        raise ValueError('state does not meet the Z target')
    assert st.visible() == state.visible()
    return st


class DirectFirstStage:
    """FullRung-compatible experimental IDA* front end. It never needs a generated subgroup chain.

    max_nodes is a per-input budget (0: unlimited). An exhausted budget can return
    fewer candidates than requested; last_stats records this explicitly.
    """
    def __init__(self, cache=None, max_nodes=20_000_000, max_depth=20, max_candidates=200, verbose=False):
        if max_nodes < 0 or not 0 <= max_depth <= 63 or max_candidates < 1:
            raise ValueError('expected max_nodes >= 0, 0 <= max_depth <= 63, max_candidates >= 1')
        self.lib = _library(); self.max_nodes = max_nodes; self.max_depth = max_depth
        self.max_candidates = max_candidates; self.last_stats = []; self.candidates = []
        self.rmask = masks(18, 8)
        self.rindex = np.full(1 << 18, -1, np.int32); self.rindex[self.rmask] = np.arange(len(self.rmask))
        self.goal = int(self.rindex[UD]) * 2 * NC
        self.ct = np.empty((NC, 21), np.uint16)
        co = (np.arange(NC)[:, None] // (3 ** np.arange(6))) % 3
        co = np.column_stack([co, -co.sum(axis=1) % 3, np.zeros(NC, int)])
        self.rt = np.empty((NR, 21), np.uint32)
        rb = (self.rmask[:, None] >> np.arange(18)) & 1
        self.lut = np.zeros((21, 3, 256), np.uint32)
        self.spin = np.array([bitmask(m.dgo % 2) for m in MOVES], np.uint32)
        for j, m in enumerate(MOVES):
            self.ct[:, j] = (((co[:, m.src.cp] + m.dco) % 3)[:, :6] * (3 ** np.arange(6))).sum(axis=1)
            moved = rb[:, m.src.rp]; newmask = (moved * (1 << np.arange(18))).sum(axis=1)
            parity = ((moved * (m.dro % 2)).sum(axis=1) % 2).astype(np.uint32)
            ri = self.rindex[newmask].astype(np.uint32) * 2
            self.rt[0::2, j] = ri + parity; self.rt[1::2, j] = ri + (parity ^ 1)
            dest = np.argsort(m.src.gp)
            for byte in range(3):
                for b in range(8):
                    self.lut[j, byte] |= (((np.arange(256, dtype=np.uint32) >> b) & 1) << int(dest[byte * 8 + b]))
        signature = hashlib.sha256(b'firststage-v1' + self.ct.tobytes() + self.rt.tobytes()
                                   + self.lut.tobytes() + COST.tobytes() + np.uint32(EQ).tobytes()
                                   + np.uint32(self.goal).tobytes()).hexdigest()[:16]
        directory = (Path(cache) if cache is not None else ROOT / 'tables') / ('firststage_' + signature)
        directory.mkdir(parents=True, exist_ok=True)
        def table(name, size, build):
            path = directory / (name + '.npy')
            if not path.exists():
                start = time.perf_counter(); data = np.empty(size, np.uint8); build(data)
                tmp = directory / (name + '.tmp.npy'); np.save(tmp, data); tmp.replace(path)
                if verbose: print(f'{name}: {size:,} bytes, {time.perf_counter() - start:.2f}s', flush=True)
            a = np.load(path, mmap_mode='r')
            if a.dtype != np.uint8 or a.shape != (size,): raise ValueError(f'invalid pattern table: {path}')
            return a
        def joint(data):
            reached = self.lib.first_joint(_p(data), _p(self.ct), _p(self.rt), NR, self.goal, _p(COST))
            if reached != NR * NC: raise RuntimeError(f'joint pattern reached {reached} / {NR * NC}')
        def gear(data, count, neq):
            ms = masks(24, count)
            reached = self.lib.first_gears(_p(data), _p(ms), len(ms), EQ, neq, _p(self.lut), _p(COST))
            if reached != len(ms): raise RuntimeError(f'gear pattern reached {reached} / {len(ms)}')
        self.joint = table('corner_ring', NR * NC, joint)
        self.side = table('side_gears', 1 << 24, lambda data: gear(data, 4, 2))
        self.ud = table('ud_gears', 1 << 24, lambda data: gear(data, 8, 0))

    def solve_many(self, states, window=0, max_candidates=None):
        if window < 0: raise ValueError('window must be nonnegative')
        cap = self.max_candidates if max_candidates is None else max_candidates
        if cap < 1: raise ValueError('max_candidates must be positive')
        results = []; self.last_stats = []
        for st in states:
            if M.check(st): raise ValueError('expected a reachable state with DBL fixed')
            c = int(np.dot(st.co[:6], 3 ** np.arange(6)))
            r = int(self.rindex[bitmask(np.isin(st.rp, L.UDE))]) * 2
            r += int(st.ro[np.isin(st.rp, L.UDE)].sum()) % 2
            colors = M.GEAR_COLOR[st.gp]
            g = np.array([bitmask(colors < 2)] + [bitmask(colors == i) for i in range(2, 6)], np.uint32)
            spin = bitmask(st.go % 2)
            paths = np.zeros((cap, 64), np.int32); lengths = np.zeros(cap, np.int32); stats = np.zeros(3, np.uint64)
            start = time.perf_counter()
            n = self.lib.first_search(_p(self.joint), _p(self.side), _p(self.ud), _p(self.ct), _p(self.rt),
                _p(self.lut), _p(self.spin), _p(COST), self.goal, EQ, c, r, _p(g), spin,
                self.max_depth, window, self.max_nodes, cap, _p(paths), _p(lengths), _p(stats))
            self.last_stats.append(dict(nodes=int(stats[0]), best=None if not n else int(stats[1]),
                                        exhausted=bool(stats[2]), seconds=time.perf_counter() - start))
            out = []; seen = set()
            for p, length in zip(paths[:n], lengths[:n]):
                words = [RUNS[i] for i in p[:length]]; seq = ' '.join(words)
                landed = relabel_to_z(M.apply_seq(st, seq))
                if landed.key() in seen: continue
                seen.add(landed.key()); out.append((L.htm(seq), words, [], landed))
            results.append(out)
        return results

    def solve(self, state, window=0, max_candidates=None):
        self.candidates = self.solve_many([state], window, max_candidates)[0]
        return self.candidates[0] if self.candidates else None


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--count', type=int, default=10); ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--nodes', type=int, default=20_000_000)
    args = ap.parse_args()
    solver = DirectFirstStage(max_nodes=args.nodes, verbose=True); rng = np.random.default_rng(args.seed)
    for i in range(args.count):
        result = solver.solve(M.random_state(rng), max_candidates=1)
        print(json.dumps(dict(index=i, htm=None if result is None else result[0], **solver.last_stats[0])), flush=True)
