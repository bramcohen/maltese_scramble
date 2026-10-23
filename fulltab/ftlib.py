"""ctypes wrapper for fulltab/ft.c (rebuilt when the source changes)."""
import os, subprocess, ctypes as ct, numpy as np, coset as CS
HERE = os.path.dirname(os.path.abspath(__file__)); SO = os.path.join(HERE, "ft.so"); SRC = os.path.join(HERE, "ft.c")
if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(SRC):
    subprocess.check_call(["cc", "-O3", "-mcpu=apple-m1", "-shared", "-fPIC", "-o", SO + ".tmp", SRC, "-lpthread"]); os.replace(SO + ".tmp", SO)
lib = ct.CDLL(SO)
i8p = ct.POINTER(ct.c_int8); i64p = ct.POINTER(ct.c_int64); u64p = ct.POINTER(ct.c_uint64); dp = ct.POINTER(ct.c_double)
SP = ct.POINTER(CS._Spec); VP = ct.c_void_p
lib.fs_new.restype = VP; lib.fs_new.argtypes = [SP]
lib.labels50.argtypes = [i8p, ct.c_int64, VP, u64p, ct.c_int]
p = lambda a, t: a.ctypes.data_as(t)
CM = np.uint64(31)
_FS = {}

def fs(spec):
    """The 50-point form of a spec (built once per spec object)."""
    if id(spec) not in _FS: _FS[id(spec)] = (spec, lib.fs_new(ct.byref(spec.s)))
    return _FS[id(spec)][1]

def labels50(states, spec, threads=18):
    states = np.ascontiguousarray(states); out = np.empty(len(states), np.uint64)
    lib.labels50(p(states, i8p), len(states), fs(spec), p(out, u64p), threads); return out

u32p = ct.POINTER(ct.c_uint32); u8p = ct.POINTER(ct.c_uint8)
lib.elems50.restype = ct.c_double
lib.elems50.argtypes = [i8p, i8p, i64p, ct.c_int64, ct.c_int, u64p, u64p, ct.c_int, ct.c_int, i64p]
lib.build_bidx.argtypes = [u64p, ct.c_int64, ct.c_int, u32p]
lib.pair_maps50.argtypes = [i8p, u64p, ct.c_int64, u8p]

def elems50(states, cost, gstart, maxcost, hbits, threads=18):
    """Distinct elements p . inverse(q) up to maxcost: (cost array, witness pairs) in hash order."""
    ht = np.zeros(1 << hbits, np.uint64); ew = np.zeros(1 << hbits, np.uint64); st = np.zeros(3, np.int64)
    t = lib.elems50(p(states, i8p), p(cost, i8p), p(gstart, i64p), len(gstart) - 1, maxcost, p(ht, u64p), p(ew, u64p), hbits, threads, p(st, i64p))
    used = ht != 0
    return (ht[used] & CM).astype(np.int8), ew[used], dict(pairs=int(st[0]), seconds=t)

def pair_maps50(states, pq):
    pq = np.ascontiguousarray(pq); out = np.zeros((len(pq), 64), np.uint8)
    lib.pair_maps50(p(states, i8p), p(pq, u64p), len(pq), p(out, u8p)); return out

def bucket_index(tab, bbits):
    b = np.zeros((1 << bbits) + 1, np.uint32); lib.build_bidx(p(tab, u64p), len(tab), bbits, p(b, u32p)); return b

lib.chain_step.restype = ct.c_int64
lib.chain_step.argtypes = [u8p, i8p, ct.c_int64, ct.c_int64, u8p, i8p, ct.c_int64, ct.c_int, VP, u64p, u64p, ct.c_int, ct.c_int]
lib.chain_maps.argtypes = [u8p, u8p, u64p, ct.c_int64, u8p]
lib.maps50.argtypes = [i8p, ct.c_int64, u8p]
lib.seed.argtypes = [u64p, u64p, ct.c_int64, u64p, i8p, u64p, ct.c_int64]
lib.join50w.restype = ct.c_double
lib.join50w.argtypes = [i8p, i8p, i64p, i64p, ct.c_int64, VP, VP, ct.c_int, u64p, u64p, ct.c_int, ct.c_int, i64p]

def maps50(states):
    states = np.ascontiguousarray(states); out = np.zeros((len(states), 64), np.uint8); lib.maps50(p(states, i8p), len(states), p(out, u8p)); return out

def join50w(states, cost, gstart, nrep, spec, maxcost, hbits, threads=18, mir=False):
    """Class join keeping a witness pair per class: returns (label | cost array, witness rows, stats), unsorted."""
    ht = np.zeros(1 << hbits, np.uint64); ew = np.zeros(1 << hbits, np.uint64); st = np.zeros(3, np.int64)
    t = lib.join50w(p(states, i8p), p(cost, i8p), p(gstart, i64p), p(nrep, i64p), len(gstart) - 1, fs(spec), mirror() if mir else None, maxcost, p(ht, u64p), p(ew, u64p), hbits, threads, p(st, i64p))
    used = ht != 0
    return ht[used], ew[used], dict(pairs=int(st[0]), lost=int(st[1]), seconds=t)

def chain(base_maps, base_cost, pieces, pcost, spec, maxcost, start_cost, hbits=28, threads=18, log=None, base_wit=None):
    """Extend a complete table by chaining: every class of cost c (final once all cheaper classes are done, since
    pieces cost >= 1) is multiplied by every piece.  base_maps: one representative map per base class, base_cost its
    cost; the base is taken as complete below start_cost.  Rows of R are kept contiguous per cost (base sorted by cost,
    then one block per new cost), so witnesses can index R directly.
    Returns (sorted label | cost, R maps, RC costs, DER: row * 2^20 + piece for chained rows, -1 for base rows)."""
    import model as _M
    pieces = np.ascontiguousarray(pieces, np.uint8); pcost = np.ascontiguousarray(pcost, np.int8)
    ht = np.zeros(1 << hbits, np.uint64); ew = np.zeros(1 << hbits, np.uint64)
    o = np.argsort(base_cost, kind="stable"); nb = len(o); cap = max(2 * nb, 1 << 20)
    rf = os.path.join(HERE, "data", "mm", f"chainR_{os.getpid()}.npy"); os.makedirs(os.path.dirname(rf), exist_ok=True)
    R = np.lib.format.open_memmap(rf, mode="w+", dtype=np.uint8, shape=(1 << 31, 64))      # sparse: only touched rows take space
    RC = np.zeros(cap, np.int8); DER = np.full(cap, -1, np.int64); n = nb
    R[:nb] = base_maps[o]; RC[:nb] = np.asarray(base_cost)[o]; BW = tuple(np.asarray(w)[o] for w in base_wit) if base_wit is not None else None
    MARK = np.uint64(0xffffffff)
    bl = probe_labels_maps(CS.pack_states([_M.State()])[0], np.ascontiguousarray(R[:nb]), spec, threads)
    lib.seed(p(ht, u64p), p(ew, u64p), hbits, p(bl, u64p), p(np.ascontiguousarray(RC[:nb]), i8p),
             p(np.ascontiguousarray((np.arange(nb, dtype=np.uint64) << np.uint64(32)) | MARK), u64p), nb)
    for c in range(start_cost, maxcost):
        rows = np.flatnonzero(RC[:n] == c); done = 0
        if len(rows):
            cuts = np.flatnonzero(np.diff(rows) != 1) + 1
            for seg in np.split(rows, cuts):                       # contiguous segments of rows with cost c
                a, b = int(seg[0]), int(seg[-1]) + 1
                done += lib.chain_step(p(np.ascontiguousarray(R[a:b]), u8p), p(np.ascontiguousarray(RC[a:b]), i8p), b - a, a,
                                       p(pieces, u8p), p(pcost, i8p), len(pieces), maxcost, fs(spec), p(ht, u64p), p(ew, u64p), hbits, threads)
        used = np.flatnonzero((ht != 0) & ((ht & CM) == c + 1))
        w = ew[used]; new = (w & MARK) != MARK; m = int(new.sum())
        if m:
            while n + m > cap:
                cap *= 2; RC = np.concatenate([RC, np.zeros_like(RC)]); DER = np.concatenate([DER, np.full(len(DER), -1, np.int64)])
            assert n + m <= R.shape[0], "chain: more than 2^31 rows"
            wn = np.ascontiguousarray(w[new]); out = np.zeros((m, 64), np.uint8)
            lib.chain_maps(p(R, u8p), p(pieces, u8p), p(wn, u64p), m, p(out, u8p))
            R[n:n + m] = out; RC[n:n + m] = c + 1
            DER[n:n + m] = (wn >> np.uint64(32)).astype(np.int64) * (1 << 20) + (wn & MARK).astype(np.int64); n += m
            ew[used[new]] = (np.arange(n - m, n, dtype=np.uint64) << np.uint64(32)) | MARK
        if log: log(f"  chain from cost {c}: {len(rows):,} classes ({done:,} products); new classes at {c + 1}: {m:,}; table {int((ht != 0).sum()):,}")
    used = np.flatnonzero(ht != 0); t = ht[used]; rows = (ew[used] >> np.uint64(32)).astype(np.int64); o = np.argsort(t, kind="stable")
    Rn = R[:n]; del R; os.remove(rf) if False else None          # the scratch file is left for the caller's lifetime (memmap view)
    return (t[o], rows[o], Rn, RC[:n], DER[:n]) + ((BW,) if BW is not None else ())

lib.probe_labels_maps.argtypes = [i8p, u8p, ct.c_int64, VP, u64p, ct.c_int]
def probe_labels_maps(x, elmaps, spec, threads=18):
    out = np.empty(len(elmaps), np.uint64)
    lib.probe_labels_maps(p(np.ascontiguousarray(x), i8p), p(elmaps, u8p), len(elmaps), fs(spec), p(out, u64p), threads); return out


def group_order_reps(states, cost, glab, tspec, threads=18, return_order=False):
    """Order a ball by group label glab; within each group, rows with a new (target label, cost) first.
    Returns (states, cost, gstart, nrep)."""
    tl = labels50(states, tspec, threads) & ~CM
    o, g, k = sort_group_key(glab, tl | (np.asarray(cost, np.uint64) & np.uint64(31))); t = k & ~CM; c = (k & CM).astype(cost.dtype)
    newg = np.r_[True, g[1:] != g[:-1]]
    rep = newg | np.r_[True, (t[1:] != t[:-1]) | (c[1:] != c[:-1])]
    gid = np.cumsum(newg) - 1
    o2 = np.argsort(gid * 2 + (~rep), kind="stable"); o = o[o2]; rep = rep[o2]; gid = gid[o2]
    gstart = np.r_[np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]]), len(o)].astype(np.int64)
    nrep = np.bincount(gid, weights=rep, minlength=len(gstart) - 1).astype(np.int64)
    out = (gather_rows(states, o), np.ascontiguousarray(cost[o]), gstart, nrep)
    return out + (o,) if return_order else out

def gather_rows(states, o, limit=8 << 30):
    """states[o]; above `limit` bytes the result is a memory-mapped scratch file (page cache) instead of anonymous memory."""
    if states.nbytes <= limit: return np.ascontiguousarray(states[o])
    f = os.path.join(HERE, "data", "mm", f"gather_{os.getpid()}_{len(o)}.npy"); os.makedirs(os.path.dirname(f), exist_ok=True)
    out = np.lib.format.open_memmap(f, mode="w+", dtype=states.dtype, shape=(len(o),) + states.shape[1:])
    for c0 in range(0, len(o), 20_000_000): out[c0:c0 + 20_000_000] = states[o[c0:c0 + 20_000_000]]
    out.flush(); return out

lib.mir_new.restype = VP; lib.mir_new.argtypes = [u8p, u8p]
lib.labels50m.argtypes = [u8p, ct.c_int64, VP, VP, u64p]
_MIR = None
def mirror():
    """The front/back mirror (fitted offsets in fulltab/data/mirrorFB_offsets.npy)."""
    global _MIR
    if _MIR is None:
        import model as _M
        sw = lambda n: n.translate(str.maketrans("FB", "BF"))
        M_ = np.r_[[_M.CORNER_NAMES.index(sw(n)) for n in _M.CORNER_NAMES], [8 + _M.RING_NAMES.index(sw(n)) for n in _M.RING_NAMES],
                   [26 + _M.GEAR_NAMES.index(sw(n)) for n in _M.GEAR_NAMES]].astype(np.uint8)
        OFF = np.load(os.path.join(HERE, "data", "mirrorFB_offsets.npy")).astype(np.uint8)
        _MIR = (M_, OFF, lib.mir_new(p(np.ascontiguousarray(M_), u8p), p(np.ascontiguousarray(OFF), u8p)))
    return _MIR[2]
def labels50m(maps, spec):
    maps = np.ascontiguousarray(maps, np.uint8); out = np.empty(len(maps), np.uint64); lib.labels50m(p(maps, u8p), len(maps), fs(spec), mirror(), p(out, u64p)); return out

lib.join50maps.restype = ct.c_double
lib.join50maps.argtypes = [u8p, i8p, i64p, i64p, ct.c_int64, VP, VP, ct.c_int, u64p, u64p, u64p, u64p, ct.c_int, ct.c_int, i64p]
lib.gen_slice.restype = ct.c_int64
lib.gen_slice.argtypes = [u8p, ct.c_int64, u8p, ct.c_int, VP, ct.c_int, ct.c_int, u8p, u64p, u64p, ct.POINTER(ct.c_int32), i8p, ct.c_int64, ct.c_int]
lib.maps_hash.argtypes = [u8p, ct.c_int64, u64p]

def maps_hash(maps):
    maps = np.ascontiguousarray(maps, np.uint8); out = np.empty(len(maps), np.uint64); lib.maps_hash(p(maps, u8p), len(maps), p(out, u64p)); return out

def labels_of_maps(maps, spec, threads=18):
    import model as _M
    return probe_labels_maps(CS.pack_states([_M.State()])[0], np.ascontiguousarray(maps, np.uint8), spec, threads)

def gen_slice(parents, runs, spec, k, ns, cap, threads=18):
    """Children parent . run whose group label falls in slice k: (maps, labels, hashes, parent index, run index)."""
    parents = np.ascontiguousarray(parents, np.uint8); runs = np.ascontiguousarray(runs, np.uint8)
    om = np.empty((cap, 64), np.uint8); ol = np.empty(cap, np.uint64); oh = np.empty(cap, np.uint64); op = np.empty(cap, np.int32); orun = np.empty(cap, np.int8)
    n = lib.gen_slice(p(parents, u8p), len(parents), p(runs, u8p), len(runs), fs(spec), k, ns, p(om, u8p), p(ol, u64p), p(oh, u64p), op.ctypes.data_as(ct.POINTER(ct.c_int32)), p(orun, i8p), cap, threads)
    assert n < cap, "slice capacity"
    return om[:n], ol[:n], oh[:n], op[:n], orun[:n]

def sort_group_key(glab, key2):
    """Permutation ordering rows by glab and, within equal glab, grouping equal key2 (numpy's stable 64-bit sorts are
    20x slower than introsort: sort by glab, then by (group id, 37-bit hash of key2); equal key2 stay adjacent, and
    the callers compare the full key2 so a hash collision costs at most a duplicate representative)."""
    o = np.argsort(glab, kind="quicksort"); g = glab[o]; newg = np.r_[True, g[1:] != g[:-1]]; gid = (np.cumsum(newg) - 1).astype(np.uint64)
    comp = (gid << np.uint64(37)) | ((key2[o] * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(27)); o3 = np.argsort(comp, kind="quicksort"); del comp, gid
    o = o[o3]; return o, g[o3], key2[o]

def order_maps_reps(maps, cost, glab, tlab, ids=None):
    """Order map rows by group label, representatives (new (target label, cost)) first within each group.
    Returns (maps, cost, gstart, nrep) and, if ids are given, the ids in the same order."""
    key2 = (np.asarray(tlab, np.uint64) & ~CM) | (np.asarray(cost, np.uint64) & np.uint64(31))     # (target label, cost) as one key
    o, g, k = sort_group_key(glab, key2); del key2
    newg = np.r_[True, g[1:] != g[:-1]]; rep = newg | np.r_[True, k[1:] != k[:-1]]; del g, k
    gid = np.cumsum(newg) - 1; o2 = np.argsort(gid * 2 + (~rep), kind="stable"); o = o[o2]; rep = rep[o2]; gid = gid[o2]; del o2
    gstart = np.r_[np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]]), len(o)].astype(np.int64)
    nrep = np.bincount(gid, weights=rep, minlength=len(gstart) - 1).astype(np.int64)
    out = (np.ascontiguousarray(maps[o]), np.ascontiguousarray(cost[o]), gstart, nrep)
    return out + (np.ascontiguousarray(ids[o]),) if ids is not None else out

def join50maps(maps, cost, gstart, nrep, spec, maxcost, hbits, ht, ew=None, ew2=None, rowid=None, mir=False, threads=18):
    """Class join over map rows.  With ew and ew2, each class keeps the witness ids (rowid[p], rowid[q]) of its cheapest
    product p . inverse(q); rowid maps rows to global state ids."""
    st = np.zeros(3, np.int64)
    t = lib.join50maps(p(maps, u8p), p(cost, i8p), p(gstart, i64p), p(nrep, i64p), len(gstart) - 1, fs(spec), mirror() if mir else None, maxcost,
                       p(ht, u64p), p(ew, u64p) if ew is not None else None, p(ew2, u64p) if ew2 is not None else None,
                       p(np.ascontiguousarray(rowid, np.uint64), u64p) if rowid is not None else None, hbits, threads, p(st, i64p))
    return dict(pairs=int(st[0]), lost=int(st[1]), seconds=t)

# ---- sequences behind table classes (witness ids: ball row, or GEN + parent row * 32 + run for generated states)
RUNS = [f + a for f in "URF" for a in ("", "2", "3", "4", "'", "'2", "'3")]
def _inv_run(w):
    f, rest = w[0], w[1:]; return f + ("" if rest.startswith("'") else "'") + rest.lstrip("'")
class Witness:
    _cache = {}                                   # one copy of each ball's parent arrays per process
    def __init__(self, h, D):
        if (h, D) not in Witness._cache:
            z = np.load(os.path.join(D, f"ball{h}_parents.npz")); Witness._cache[(h, D)] = (z["par"], z["run"])
        self.par, self.run = Witness._cache[(h, D)]; self.GEN = 1 << 40
    def seq(self, sid):
        sid = int(sid); tail = []
        if sid >= self.GEN: tail = [RUNS[sid & 31]]; sid = (sid - self.GEN) >> 5
        out = []
        while sid > 0: out.append(RUNS[int(self.run[sid])]); sid = int(self.par[sid])
        return out[::-1] + tail
    def element(self, p, q):
        """moves of p . inverse(q): p's sequence then q's reversed and inverted"""
        return self.seq(p) + [_inv_run(w) for w in self.seq(q)[::-1]]
class Witness2(Witness):
    """Witnesses of a two-level sliced table (slice2.py): ids are ball rows, GEN1 + level-(b+1) index, or
    GEN2 + (level-(b+1) index << 5 | run) -- with bit 39 set, GEN2 + (ball row << 5 | run) for cost-2 runs from the
    stored level.  The level-(b+1) parents (P1: ball row, R1: run) are the memmaps in the build directory."""
    def __init__(self, b, W, D):
        super().__init__(b, D); self.P1 = np.load(os.path.join(W, "P1.npy"), mmap_mode="r"); self.R1 = np.load(os.path.join(W, "R1.npy"), mmap_mode="r")
        self.GEN1, self.GEN2 = 1 << 40, 1 << 41
    def seq(self, sid):
        sid = int(sid); tail = []
        if sid >= self.GEN2:
            v = sid - self.GEN2; stored = (v >> 39) & 1; v &= (1 << 39) - 1; idx, run = v >> 5, v & 31; tail = [RUNS[run]]
            if stored: sid = idx
            else: tail = [RUNS[int(self.R1[idx])]] + tail; sid = int(self.P1[idx])
        elif sid >= self.GEN1:
            idx = sid - self.GEN1; tail = [RUNS[int(self.R1[idx])]]; sid = int(self.P1[idx])
        out = []
        while sid > 0: out.append(RUNS[int(self.run[sid])]); sid = int(self.par[sid])
        return out[::-1] + tail

lib.fin_mod.argtypes = [u64p, ct.c_int64, ct.c_int, i64p]
def slice_of(labels, ns):
    labels = np.ascontiguousarray(labels, np.uint64); out = np.empty(len(labels), np.int64); lib.fin_mod(p(labels, u64p), len(labels), ns, p(out, i64p)); return out

lib.probe50_multi.restype = ct.c_int64
lib.probe50_multi.argtypes = [u8p, i8p, ct.c_int64, ct.c_int, i8p, ct.c_int64, u64p, u32p, ct.c_int, VP, VP, ct.c_int, ct.c_int, ct.c_int, i64p, ct.c_int64]
def probe50_multi(elmaps, ecost, emax, xs, tab, bidx, bbits, spec, bound, hcap=1 << 16, threads=18, mir=False, window=31):
    """Hits of many inputs (packed states xs) against one forward list and table: array of (input index, element index,
    table index, forward cost, table cost) with forward + table cost <= bound and, per input, within `window` of the
    cheapest hit seen so far (a few hits above the final best + window may remain; the caller filters)."""
    xs = np.ascontiguousarray(xs, np.int8); hits = np.zeros((hcap, 5), np.int64)
    n = lib.probe50_multi(p(elmaps, u8p), p(ecost, i8p), len(elmaps), emax, p(xs, i8p), len(xs), p(tab, u64p), p(bidx, u32p), bbits,
                          fs(spec), mirror() if mir else None, threads, bound, window, p(hits, i64p), hcap)
    return hits[:n]
lib.gen_bins.restype = ct.c_int64
lib.gen_bins.argtypes = [u8p, ct.c_int64, ct.c_int64, u8p, ct.c_int, VP, ct.c_int, ct.POINTER(i64p), ct.POINTER(i8p), i64p, ct.c_int64, ct.c_int]
lib.child_maps.argtypes = [u8p, ct.POINTER(ct.c_int32), i8p, u8p, ct.c_int64, u8p]
lib.bitmap_set.argtypes = [u8p, ct.c_int, u64p, ct.c_int64]
lib.bitmap_test.argtypes = [u8p, ct.c_int, u64p, ct.c_int64, u8p]
def gen_bins(parents, pbase, runs, spec, ns, cap, threads=18):
    """children of the given parent maps under runs, binned by label slice: list of (parent index, run) arrays per slice"""
    parents = np.ascontiguousarray(parents, np.uint8); runs = np.ascontiguousarray(runs, np.uint8)
    bp = [np.empty(cap, np.int64) for _ in range(ns)]; br = [np.empty(cap, np.int8) for _ in range(ns)]; cnt = np.zeros(ns, np.int64)      # int64: global level-(b+1) indices pass 2^31
    BP = (i64p * ns)(*[b.ctypes.data_as(i64p) for b in bp]); BR = (i8p * ns)(*[b.ctypes.data_as(i8p) for b in br])
    lost = lib.gen_bins(p(parents, u8p), len(parents), pbase, p(runs, u8p), len(runs), fs(spec), ns, BP, BR, p(cnt, i64p), cap, threads)
    assert lost == 0, "bin capacity"
    return [(bp[k][:cnt[k]].copy(), br[k][:cnt[k]].copy()) for k in range(ns)]
def child_maps(parents, pidx, run, runs):
    pidx = np.ascontiguousarray(pidx, np.int32); run = np.ascontiguousarray(run, np.int8); out = np.zeros((len(pidx), 64), np.uint8)
    lib.child_maps(p(np.ascontiguousarray(parents, np.uint8), u8p), pidx.ctypes.data_as(ct.POINTER(ct.c_int32)), p(run, i8p), p(np.ascontiguousarray(runs, np.uint8), u8p), len(pidx), p(out, u8p)); return out
class Bitmap:
    def __init__(self, bits): self.bits = bits; self.bm = np.zeros(1 << (bits - 3), np.uint8)
    def add(self, h): lib.bitmap_set(p(self.bm, u8p), self.bits, p(np.ascontiguousarray(h, np.uint64), u64p), len(h))
    def test(self, h): out = np.zeros(len(h), np.uint8); lib.bitmap_test(p(self.bm, u8p), self.bits, p(np.ascontiguousarray(h, np.uint64), u64p), len(h), p(out, u8p)); return out.astype(bool)
lib.child_maps2.argtypes = [u8p, ct.POINTER(ct.c_int32), i8p, i8p, u8p, ct.c_int64, u8p]
def child_maps2(ball, p9, r1, r2, runs):
    p9 = np.ascontiguousarray(p9, np.int32); r1 = np.ascontiguousarray(r1, np.int8); r2 = np.ascontiguousarray(r2, np.int8); out = np.zeros((len(p9), 64), np.uint8)
    lib.child_maps2(p(np.ascontiguousarray(ball, np.uint8), u8p), p9.ctypes.data_as(ct.POINTER(ct.c_int32)), p(r1, i8p), p(r2, i8p), p(np.ascontiguousarray(runs, np.uint8), u8p), len(p9), p(out, u8p)); return out
