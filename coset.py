"""Coset-space meet-in-the-middle over word lists, with the hot loop in C (csrc/coset.c).

States are packed as 100 int8 bytes (cp co rp ro gp go); elements use the same layout.
A `Spec` describes the coset label (see csrc/coset.c): slot classes, sibling slots, Z_m spin
functionals, and optional right multipliers for a canonical label under a small subgroup.
"""
import ctypes as ct
import os
import subprocess
import time

import numpy as np

import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "build", "libcoset.so")
SRC = os.path.join(HERE, "csrc", "coset.c")
SB = 100
NS = {"c": 8, "r": 18, "g": 24}


def _build():
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
        # build to a temporary file and rename, so a running process keeps its mapped copy
        subprocess.check_call(["cc", "-O3", "-shared", "-fPIC", "-o", LIB + ".tmp", SRC, "-lpthread"])
        os.replace(LIB + ".tmp", LIB)


_build()
lib = ct.CDLL(LIB)
i8p = ct.POINTER(ct.c_int8)
u8p = ct.POINTER(ct.c_uint8)
u64p = ct.POINTER(ct.c_uint64)
i64p = ct.POINTER(ct.c_int64)


class _Spec(ct.Structure):
    _fields_ = [("msk_c", u8p), ("msk_r", u8p), ("msk_g", u8p),
                ("cls_c", i8p), ("cls_r", i8p), ("cls_g", i8p),
                ("smsk_c", u8p), ("smsk_r", u8p), ("smsk_g", u8p),
                ("sib_c", i8p), ("sib_r", i8p), ("sib_g", i8p),
                ("typ_c", i8p), ("typ_r", i8p), ("typ_g", i8p),
                ("nt_c", ct.c_int), ("nt_r", ct.c_int), ("nt_g", ct.c_int),
                ("F_c", i8p), ("F_r", i8p), ("F_g", i8p),
                ("nf_c", ct.c_int), ("nf_r", ct.c_int), ("nf_g", ct.c_int),
                ("pot_slot_c", i8p), ("pot_slot_r", i8p), ("pot_slot_g", i8p),
                ("pot_piece_c", i8p), ("pot_piece_r", i8p), ("pot_piece_g", i8p),
                ("rm", i8p), ("nrm", ct.c_int), ("lm", i8p), ("nlm", ct.c_int),
                ("ncan", ct.c_int), ("can_m", ct.POINTER(ct.c_int32)), ("can_off", ct.POINTER(ct.c_int32)),
                ("can_pts", ct.POINTER(ct.c_int16)), ("can_perm", u8p)]


lib.coset_keys.argtypes = [i8p, ct.c_int64, i8p, ct.POINTER(_Spec), u64p, ct.c_int]
lib.coset_apply.argtypes = [i8p, ct.c_int64, i8p, ct.POINTER(_Spec), i8p, u64p, ct.c_int]
lib.coset_apply_seqs.argtypes = [i8p, i8p, i64p, ct.POINTER(ct.c_int32), ct.c_int64, i8p]
lib.kset_new.restype = ct.c_void_p
lib.kset_new.argtypes = [ct.c_int]
lib.kset_free.argtypes = [ct.c_void_p]
lib.kset_count.restype = ct.c_int64
lib.kset_count.argtypes = [ct.c_void_p]
lib.kset_insert.restype = ct.c_int64
lib.kset_insert.argtypes = [ct.c_void_p, u64p, ct.c_int64, u8p]
lib.kset_has.argtypes = [ct.c_void_p, u64p, ct.c_int64, u8p]
lib.coset_sweep.restype = ct.c_int64
lib.coset_sweep.argtypes = [i8p, ct.c_int64, i8p, ct.c_int64, ct.POINTER(_Spec), u64p, ct.c_int64,
                            i64p, i64p, i64p, ct.c_int64, ct.c_int64, i64p, ct.c_int, ct.c_uint64,
                            ct.POINTER(ct.c_int16), ct.c_int64]


def pack_state(st):
    return np.concatenate([st.cp, st.co, st.rp, st.ro, st.gp, st.go]).astype(np.int8)


def pack_states(states):
    return np.ascontiguousarray(np.stack([pack_state(s) for s in states]))


def apply_seqs(start_packed, elems, seqs):
    """States reached from one packed start state by each of several sequences of element indices (C loop)."""
    offs = np.zeros(len(seqs) + 1, np.int64); offs[1:] = np.cumsum([len(q) for q in seqs])
    idx = np.ascontiguousarray(np.concatenate([np.asarray(q, np.int32) for q in seqs]) if offs[-1] else np.zeros(1, np.int32))
    out = np.empty((len(seqs), SB), np.int8); start = np.ascontiguousarray(start_packed.reshape(-1)[:SB])
    elems = np.ascontiguousarray(elems)
    lib.coset_apply_seqs(start.ctypes.data_as(i8p), elems.ctypes.data_as(i8p), offs.ctypes.data_as(i64p),
                         idx.ctypes.data_as(ct.POINTER(ct.c_int32)), len(seqs), out.ctypes.data_as(i8p))
    return out


def unpack_state(row):
    return M.State(row[0:8], row[8:16], row[16:34], row[34:52], row[52:76], row[76:100])


def pack_elem(e):
    return np.concatenate([e.src.cp, e.dco, e.src.rp, e.dro, e.src.gp, e.dgo]).astype(np.int8)


def pack_elems(elems):
    return np.ascontiguousarray(np.stack([pack_elem(e) for e in elems]))


def right_transversal(rows, target):
    """Validate representatives for target.r; repair old inverse-transversal files in memory.

    Relabelling acts on the left of the input, so one representative of each
    (S intersect target).r is required. Inverting a right transversal generally
    makes a left transversal, which need not cover these cosets. Existing rung-12
    files used that inverse convention; no large search tables need rebuilding.
    """
    rows = np.ascontiguousarray(rows, dtype=np.int8)
    if len(np.unique(target.keys(rows))) == len(rows):
        return rows
    inverse = pack_states([unpack_state(row).as_element().inverse().state() for row in rows])
    if len(np.unique(target.keys(inverse))) != len(rows):
        raise ValueError('relabelling representatives do not cover distinct target right cosets')
    return inverse


class Spec:
    """Coset label.  pos: {kind: (slot_mask, piece_class)}; sib: {kind: (slot_mask, sibling_of_piece)};
    fun: {kind: (piece_type, rows)} with rows of shape (nf, nslot*ntype); pot: {kind: (pot_slot, pot_piece)}
    subtracts pot_slot[s] + pot_piece[piece] from the spin at slot s before the functionals (virtual residues);
    rm: right-multiplier elements (label = min over state.rm); lm: left multipliers (min over lm.state)."""

    def __init__(self, pos=None, sib=None, fun=None, rm=None, lm=None, pot=None, canon=None):
        self.keep = []
        self.s = _Spec()
        for k in ("c", "r", "g"):
            n = NS[k]
            if pos and k in pos:
                m, c = pos[k]
                setattr(self.s, "msk_" + k, self._arr(np.asarray(m, np.uint8), n, u8p))
                setattr(self.s, "cls_" + k, self._arr(np.asarray(c, np.int8), n, i8p))
            if sib and k in sib:
                m, c = sib[k]
                setattr(self.s, "smsk_" + k, self._arr(np.asarray(m, np.uint8), n, u8p))
                setattr(self.s, "sib_" + k, self._arr(np.asarray(c, np.int8), n, i8p))
            if fun and k in fun:
                t, rows = fun[k]
                t = np.asarray(t, np.int8); nt = int(t.max()) + 1
                rows = np.ascontiguousarray(np.asarray(rows, np.int8).reshape(-1, n * nt))
                setattr(self.s, "typ_" + k, self._arr(t, n, i8p)); setattr(self.s, "nt_" + k, nt)
                setattr(self.s, "F_" + k, self._arr(rows, rows.size, i8p)); setattr(self.s, "nf_" + k, len(rows))
            if pot and k in pot:
                ps, pp = pot[k]
                setattr(self.s, "pot_slot_" + k, self._arr(np.asarray(ps, np.int8), n, i8p))
                setattr(self.s, "pot_piece_" + k, self._arr(np.asarray(pp, np.int8), n, i8p))
        if rm:
            a = pack_elems(rm); self.keep.append(a)
            self.s.rm = a.ctypes.data_as(i8p); self.s.nrm = len(a)
        if lm:
            a = pack_states([M.State().apply(e) for e in lm]); self.keep.append(a)
            self.s.lm = a.ctypes.data_as(i8p); self.s.nlm = len(a)
        if canon is not None:
            # canon: name of a tables/bsgs_<name>.pkl (stabiliser chain) or its loaded dict; exact label of the left coset
            import pickle
            d = canon if isinstance(canon, dict) else pickle.load(open(os.path.join(HERE, "tables", f"bsgs_{canon}.pkl"), "rb"))
            levels = [(b, pts, pm) for b, pts, pm in d["levels"] if len(pts) > 1]
            if not levels:                      # trivial group: the label is the state itself (element-level tables);
                self.s.ncan = -1                # -1 makes the kernel take the canonical-form path with no levels
            else:
                m = np.array([len(pts) for b, pts, pm in levels], np.int32)
                off = np.concatenate([[0], np.cumsum(m)[:-1]]).astype(np.int32)
                pts = np.concatenate([pts for b, pts, pm in levels]).astype(np.int16)
                perm = np.ascontiguousarray(np.concatenate([pm for b, pts, pm in levels]).astype(np.uint8))
                for a in (m, off, pts, perm):
                    self.keep.append(a)
                self.s.ncan = len(levels); self.s.can_m = m.ctypes.data_as(ct.POINTER(ct.c_int32)); self.s.can_off = off.ctypes.data_as(ct.POINTER(ct.c_int32))
                self.s.can_pts = pts.ctypes.data_as(ct.POINTER(ct.c_int16)); self.s.can_perm = perm.ctypes.data_as(u8p)

    def _arr(self, a, n, ptr):
        a = np.ascontiguousarray(a)
        assert a.size == n, (a.size, n)
        self.keep.append(a)
        return a.ctypes.data_as(ptr)

    def keys(self, states, elem=None, threads=16):
        states = np.ascontiguousarray(states, np.int8)
        out = np.empty(len(states), np.uint64)
        e = None if elem is None else pack_elem(elem) if not isinstance(elem, np.ndarray) else elem
        lib.coset_keys(states.ctypes.data_as(i8p), len(states), None if e is None else e.ctypes.data_as(i8p),
                       ct.byref(self.s), out.ctypes.data_as(u64p), threads)
        return out

    def apply(self, states, elem, threads=16):
        states = np.ascontiguousarray(states, np.int8)
        out = np.empty_like(states); keys = np.empty(len(states), np.uint64)
        e = pack_elem(elem) if not isinstance(elem, np.ndarray) else elem
        lib.coset_apply(states.ctypes.data_as(i8p), len(states), e.ctypes.data_as(i8p), ct.byref(self.s),
                        out.ctypes.data_as(i8p), keys.ctypes.data_as(u64p), threads)
        return out, keys

    def sweep(self, states, elems, table, want=1, cap=100000, threads=16, cmask=0, costs=None, maxcost=-1):
        """Probe key(states . e) for each packed element e in turn; stop after the first element with
        >= want hits.  Returns (elements processed, hit_elem, hit_state, hit_tablepos)."""
        states = np.ascontiguousarray(states, np.int8)
        he = np.empty(cap, np.int64); hi = np.empty(cap, np.int64); hp = np.empty(cap, np.int64)
        nh = ct.c_int64(0)
        done = lib.coset_sweep(states.ctypes.data_as(i8p), len(states), elems.ctypes.data_as(i8p), len(elems),
                               ct.byref(self.s), table.ctypes.data_as(u64p), len(table),
                               he.ctypes.data_as(i64p), hi.ctypes.data_as(i64p), hp.ctypes.data_as(i64p),
                               cap, want, ct.byref(nh), threads, ct.c_uint64(cmask),
                               costs.ctypes.data_as(ct.POINTER(ct.c_int16)) if costs is not None else None, maxcost)
        n = min(nh.value, cap)
        return done, he[:n], hi[:n], hp[:n]


class KeySet:
    """Open-addressing set of uint64 keys in C (for deduplicating coset labels without sorting)."""

    def __init__(self, expected):
        log2 = max(16, int(np.ceil(np.log2(max(expected, 1) * 2))))
        self.ptr = lib.kset_new(log2)

    def __del__(self):
        if getattr(self, "ptr", None):
            lib.kset_free(self.ptr); self.ptr = None

    def __len__(self):
        return lib.kset_count(self.ptr)

    def insert(self, keys):
        """Insert; returns a bool mask of the keys that were new."""
        keys = np.ascontiguousarray(keys, np.uint64); m = np.empty(len(keys), np.uint8)
        lib.kset_insert(self.ptr, keys.ctypes.data_as(u64p), len(keys), m.ctypes.data_as(u8p))
        return m.astype(bool)

    def contains(self, keys):
        keys = np.ascontiguousarray(keys, np.uint64); m = np.empty(len(keys), np.uint8)
        lib.kset_has(self.ptr, keys.ctypes.data_as(u64p), len(keys), m.ctypes.data_as(u8p))
        return m.astype(bool)


def inverse_word(n):
    """Inverse of a zera-notation word (tokens like U, U', U2, U'4)."""
    out = []
    for t in reversed(n.split()):
        face, rest = t[0], t[1:]
        prime = rest.startswith("'")
        count = rest[1:] if prime else rest
        out.append(face + ("" if prime else "'") + count)
    return " ".join(out)


class CMITM:
    """Two-sided search in coset space.  words: (name, element, cost).  spec: label.  verify(state) -> bool.
    Backward levels are built once from the goal representatives (via inverse words).  A query expands
    forward level by level (one representative per label), sweeping the last level word by word against
    the backward keys and stopping at the first word that yields a verified match."""

    def __init__(self, words, spec, verify, back_depth=2, fwd_depth=3, goals=None, verbose=False, threads=16, set_capacity=None):
        self.words, self.spec, self.verify, self.fwd_depth, self.verbose, self.threads = words, spec, verify, fwd_depth, verbose, threads
        self.elems = pack_elems([e for n, e, c in words])
        self.inv_elems = pack_elems([M.element(M.parse(inverse_word(n))) for n, e, c in words])
        self.wc = np.array([c for n, e, c in words], np.int64)
        t = time.time()
        b0 = pack_states(goals) if goals else pack_states([M.State()])
        k0 = spec.keys(b0, threads=threads)
        # levels: (keys, parent, word, cost); representative states kept for all but the last level
        self.levels = [(k0, np.full(len(b0), -1), np.full(len(b0), -1), np.zeros(len(b0), np.int64))]
        self.bstates = [b0]
        self.seen = KeySet(set_capacity or 1 << 20); self.seen.insert(k0)
        cur, curcost = b0, np.zeros(len(b0), np.int64)
        for d in range(back_depth):
            last = d == back_depth - 1
            nk, npar, nwid, ncost, nst = [], [], [], [], []
            # words in cost order, so the first occurrence of a label is (nearly) the cheapest
            for j in np.argsort(self.wc, kind="stable"):
                j = int(j)
                if last:
                    k = spec.keys(cur, self.inv_elems[j], threads)
                else:
                    s, k = spec.apply(cur, self.inv_elems[j], threads)
                new = self.seen.insert(k)
                fi = np.where(new)[0]
                if not len(fi):
                    continue
                nk.append(k[fi]); npar.append(fi.astype(np.int32)); nwid.append(np.full(len(fi), j, np.int16)); ncost.append((curcost[fi] + self.wc[j]).astype(np.int16))
                if not last:
                    nst.append(s[fi])
            nk = np.concatenate(nk) if nk else np.zeros(0, np.uint64)
            npar = np.concatenate(npar) if npar else np.zeros(0, np.int32)
            nwid = np.concatenate(nwid) if nwid else np.zeros(0, np.int16)
            ncost = np.concatenate(ncost) if ncost else np.zeros(0, np.int16)
            self.levels.append((nk, npar, nwid, ncost))
            if not last:
                st = np.concatenate(nst) if nst else np.zeros((0, SB), np.int8)
                self.bstates.append(st)
                cur, curcost = st, ncost.astype(np.int64)
            if verbose:
                print(f"   backward level {d + 1}: {len(nk)} new cosets ({time.time() - t:.0f}s)", flush=True)
        self.flat_keys = np.concatenate([l[0] for l in self.levels])
        self.flat_level = np.concatenate([np.full(len(l[0]), i) for i, l in enumerate(self.levels)])
        self.flat_idx = np.concatenate([np.arange(len(l[0])) for l in self.levels])
        self.flat_cost = np.concatenate([l[3] for l in self.levels]).astype(np.int64)
        o = np.argsort(self.flat_keys, kind="stable")
        self.flat_keys, self.flat_level, self.flat_idx, self.flat_cost = self.flat_keys[o], self.flat_level[o], self.flat_idx[o], self.flat_cost[o]

    def suffix(self, level, idx):
        out = []
        while level > 0:
            keys, par, wid, cst = self.levels[level]
            out.append(self.words[int(wid[idx])][0])
            idx = int(par[idx]); level -= 1
        return out

    def _candidates(self, st, pres, hit, pos):
        """Verify label hits (cheapest first); returns the best verified (cost, prefix words, suffix words, state)."""
        pc = self.wc[pres[hit]].sum(axis=1) if pres.shape[1] else np.zeros(len(hit), np.int64)
        total = pc + self.flat_cost[pos]
        best = None
        for r in np.argsort(total, kind="stable"):
            if best is not None and total[r] >= best[0]:
                break
            suf = self.suffix(int(self.flat_level[pos[r]]), int(self.flat_idx[pos[r]]))
            s2 = st
            for i in pres[hit[r]]:
                s2 = s2.apply(self.words[int(i)][1])
            for w in suf:
                s2 = M.apply_seq(s2, w)
            self.tried += 1
            if self.verify(s2):
                best = (int(total[r]), [self.words[int(i)][0] for i in pres[hit[r]]], suf, s2)
        return best

    def solve(self, st, want=1, max_words=None):
        """Returns (cost, prefix words, suffix words, state) or None."""
        self.hits = 0; self.tried = 0
        cur = pack_states([st]); pre = np.zeros((1, 0), np.int64)
        seen = self.spec.keys(cur, threads=self.threads)
        # depth 0: the start itself may already match
        pos = np.searchsorted(self.flat_keys, seen); pos[pos >= len(self.flat_keys)] = 0
        hit = np.where(self.flat_keys[pos] == seen)[0]
        if len(hit):
            r = self._candidates(st, pre, hit, pos[hit])
            if r:
                return r
        for d in range(self.fwd_depth):
            last = d == self.fwd_depth - 1
            if last:
                elems = self.elems if max_words is None else self.elems[:max_words]
                start = 0
                while start < len(elems):
                    done, he, hi, hp = self.spec.sweep(cur, elems[start:], self.flat_keys, want=want, threads=self.threads)
                    self.hits += len(he)
                    if len(he):
                        pres = np.hstack([pre[hi], (he + start)[:, None]])
                        r = self._candidates(st, pres, np.arange(len(he)), hp)
                        if r:
                            return r
                    start += done
                return None
            nst, nk, npre = [], [], []
            for j in range(len(self.words)):
                s, k = self.spec.apply(cur, self.elems[j], self.threads)
                _, fi = np.unique(k, return_index=True)
                nst.append(s[fi]); nk.append(k[fi]); npre.append(np.hstack([pre[fi], np.full((len(fi), 1), j, np.int64)]))
            nst = np.concatenate(nst); nk = np.concatenate(nk); npre = np.vstack(npre)
            _, first = np.unique(nk, return_index=True)
            keep = first[~np.isin(nk[first], seen)]
            seen = np.concatenate([seen, nk[keep]])
            cur, pre = nst[keep], npre[keep]
            if self.verbose:
                print(f"   forward level {d + 1}: {len(cur)} cosets", flush=True)
            # matches at this level (shorter solutions than the final sweep would give)
            k = nk[keep]
            pos = np.searchsorted(self.flat_keys, k); pos[pos >= len(self.flat_keys)] = 0
            hit = np.where(self.flat_keys[pos] == k)[0]
            self.hits += len(hit)
            if len(hit):
                r = self._candidates(st, pre, hit, pos[hit])
                if r:
                    return r
        return None


# ---------------------------------------------------------------------------------------------------------------
# cost-aware two-sided search: states are bucketed by cumulative cost (e.g. HTM) and the cheapest bucket is
# expanded first, so a label's first appearance is its minimum cost and the two sides meet at minimal total cost.

HTM_RUN = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1}      # HTM of a run of k quarter turns (k mod 8)
NRUN = 25                                                        # run codes: face*8 + k (k mod 8), 24 = "no run"


def _runs(name):
    """Runs of a word: list of (face, k mod 8) with consecutive turns of one face merged (cancelled runs dropped)."""
    out = []
    for q in M.parse(name):
        f, k = q[0], -1 if q.endswith("'") else 1
        if out and out[-1][0] == f:
            out[-1][1] = (out[-1][1] + k) % 8
            if out[-1][1] == 0:
                out.pop()
        else:
            out.append([f, k % 8])
    return out


def _run_code(run):
    return 24 if run is None else "URFDLB".index(run[0]) * 8 + run[1] if run[0] in "URF" else 24


def merged_htm(names):
    """HTM of a word sequence after merging runs across every junction (cancellations included)."""
    runs = _runs(" ".join(names))
    return sum(HTM_RUN[k] for f, k in runs)


def _saving_table():
    """SAV[a, b]: HTM saved when a word whose last run has code a is followed by one whose first run has code b
    (same face: the runs merge modulo 8), e.g. R2 then R3 -> R'3 saves 1, R then R' saves 2, R4 then R4 saves 4."""
    sav = np.zeros((NRUN, NRUN), np.int8)
    for a in range(24):
        for b in range(24):
            if a // 8 == b // 8:
                ka, kb = a % 8, b % 8
                sav[a, b] = HTM_RUN[ka] + HTM_RUN[kb] - HTM_RUN[(ka + kb) % 8]
    return sav


SAV = _saving_table()
MAXSAV = int(SAV.max())


def _last_word(pre):
    """Last real word index of each forward prefix row (-1 if none).  Rows are padded with -1, and the padding can sit
    in the middle (a child row is its parent's padded row plus one word), so this is the last non-negative entry."""
    pre = np.asarray(pre)
    if pre.ndim != 2 or pre.shape[1] == 0:
        return np.full(len(pre), -1, np.int64)
    ok = pre >= 0; w = pre.shape[1]
    li = w - 1 - np.argmax(ok[:, ::-1], axis=1)
    return np.where(ok.any(axis=1), pre[np.arange(len(pre)), li], -1).astype(np.int64)


SAV_ROWMAX = SAV.max(axis=1).astype(np.int64)           # largest saving after a word ending in each run code
CMASK = 31                                                       # compact tables: low 5 bits of each entry = cost


class CostMITM:
    """Like CMITM but by cost instead of word count.  words: (name, element, cost) with integer cost >= 1.
    Backward side: every coset reachable from the target within back_cost (states kept only for buckets that
    still get expanded).  Forward side per state: cosets within fwd_store are stored and probed; then every
    (stored bucket, word) pair whose result exceeds fwd_store is swept in increasing cost order and the search
    stops as soon as no cheaper total is possible."""

    def __init__(self, words, spec, verify, back_cost, fwd_store, goals=None, verbose=False, threads=16, max_cosets=20_000_000, max_fwd=20_000_000, back_words=None, fwd_words=None, sweep_words=None, lean=False, cache=None, compact=False,
                 max_bytes=None):
        """max_cosets decides at which cost the backward expansion stops; the table is then clipped to the costs
        that are complete (every coset up to that cost and none beyond), never a partial bucket.
        back_words: optional subset of `words` (by index) used on the backward side only, e.g. the crossing words
        when the cheap moves lie in the target group (a solution never needs to end in one of those)."""
        self.words, self.spec, self.verify, self.fwd_store, self.verbose, self.threads = words, spec, verify, fwd_store, verbose, threads
        self.max_fwd = max_fwd
        self.elems = pack_elems([e for n, e, c in words])
        self.inv_elems = pack_elems([M.element(M.parse(inverse_word(n))) for n, e, c in words])
        self.wc = np.array([c for n, e, c in words], np.int64)
        assert (self.wc >= 1).all()
        assert len(words) < 32767, "word indices are stored as int16"
        # merge-aware costs: the cost of a word next to another is reduced by the merge of the adjoining runs
        # (R2 then R3 is R'3); a step still costs at least 1 so the cheapest-first expansion stays valid
        runs = [_runs(n) for n, e, c in words]
        self.wfirst = np.array([_run_code(r[0] if r else None) for r in runs] + [24], np.int16)   # index -1 = no word
        self.wlast = np.array([_run_code(r[-1] if r else None) for r in runs] + [24], np.int16)
        self.wruns = [[(f, k) for f, k in r] for r in runs]        # per word: its runs, for exact merged totals without parsing
        self.minw = int(self.wc.min())
        self.back_idx = np.array(sorted(back_words), np.int64) if back_words is not None else np.arange(len(words))
        self.back_minw = int(self.wc[self.back_idx].min())
        # lean: keep no backward states at all; a bucket's states are regenerated from its parents (recursively,
        # from the goal states) only while it is being expanded, so the table can be as deep as labels allow
        self.lean = lean
        self._goal_states = None
        # compact: the finished table is one sorted array of (label with its low 5 bits replaced by the cost), 8 bytes
        # per class; no parent pointers (a solution's tail is rebuilt by stepping down the stored costs)
        self.compact = compact
        self.max_candidates = 200                       # a solve with a window returns at most this many results
        # fwd_words: subset used to expand the stored forward buckets (the final sweep still uses every word)
        self.fwd_idx = np.array(sorted(fwd_words), np.int64) if fwd_words is not None else np.arange(len(words))
        self.sweep_idx = sorted(sweep_words) if sweep_words is not None else list(range(len(words)))   # words applied in the final sweep
        t = time.time()
        b0 = pack_states(goals) if goals else pack_states([M.State()])
        self._goal_states = b0
        k0 = spec.keys(b0, threads=threads)
        # max_bytes: budget for the finished table; the build stops before any level that would exceed it (8 bytes per
        # class compact, ~29 classic), so a table is always complete to its last level and never over budget
        self.max_bytes = max_bytes
        if max_bytes:
            max_cosets = min(max_cosets, int(max_bytes / (8 if compact else 29)))
        # cache: a directory holding the finished backward table (labels, parent pointers, sorted flat arrays);
        # keyed by the words, the backward subset, the budgets, the goals and the labelling function
        self._cache_hash = self._cache_key(k0, back_cost, max_cosets) if cache else None
        cache = f"{cache}-{self._cache_hash[:10]}" if cache else None      # one directory per distinct table
        self.cache = cache
        if cache and self._load_cache(cache):
            if verbose:
                print(f"   backward table: {self.total_back} cosets, complete to cost {self.back_complete} (loaded from {cache}, {time.time() - t:.0f}s)", flush=True)
            return
        if compact:
            back_cost = min(back_cost, CMASK)          # a compact entry stores its cost in 5 bits
        self.seen = KeySet(int(max_cosets / 1.45) if compact else 1 << 20); self.seen.insert(k0)   # compact: sized up front, never doubles
        # Levels are generated by TARGET cost: level n holds every class first reached at cost n, generated from level
        # n-k (k = 1..max word cost) using exactly the rows whose merge-aware step cost is k, and frozen before level
        # n+1 is started.  (Expanding by SOURCE cost let a class reached at c+3 from level c claim its label before a
        # cheaper c+2 path from level c+1 was seen, so some classes were stored too dear.)  Each (row, word) pair is
        # still computed exactly once, and every frozen level is complete.
        import tempfile
        self.levels = {0: (k0, np.full(len(b0), -1, np.int32), np.full(len(b0), -1, np.int16), np.zeros(len(b0), np.int8))}
        self.bstates = {0: b0}
        maxstep = int(self.wc[self.back_idx].max())
        order = [int(j) for j in self.back_idx]
        total = len(k0); self.back_cost_reached = 0
        src, src_files = {0: b0}, {}

        def states_of(c):
            if c in src:
                return src[c]
            if not lean:
                return self.bstates[c]
            nrows = len(self.levels[c][0])
            f = tempfile.NamedTemporaryFile(prefix="bucket", suffix=".i8", dir=os.environ.get("COSET_TMP") or None, delete=False); f.close()
            mm = np.memmap(f.name, np.int8, "w+", shape=(max(nrows, 1), SB))
            for lo in range(0, nrows, self.CHUNK):
                hi = min(lo + self.CHUNK, nrows); mm[lo:hi] = self._materialise(c, np.arange(lo, hi))
            mm.flush(); src[c] = mm; src_files[c] = f.name
            return mm

        groups_of = {}

        def code_groups(c):
            # rows of level c grouped by the first-run code of their stored word (25 codes), computed once per level
            if c not in groups_of:
                codes = self.wfirst[self.levels[c][2]]
                o = np.argsort(codes, kind="stable"); sc = codes[o]
                cuts = np.flatnonzero(np.diff(sc)) + 1
                groups_of[c] = {int(sc[a]): o[a:b] for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(sc)])} if len(sc) else {}
            return groups_of[c]

        try:
            for n in range(1, back_cost + 1):
                if total > max_cosets:
                    break
                # predict level n from the growth of the last two levels; stop before a level that would overshoot
                sz = [len(self.levels[c][0]) if c in self.levels else 0 for c in (n - 1, n - 2)]
                if sz[0] and sz[1] and total + 1.3 * sz[0] * max(1.0, sz[0] / sz[1]) > max_cosets:
                    break                                         # 1.3: the growth rate itself still rises with depth
                keep = (not lean) and n + self.back_minw <= back_cost
                pk, pp, pw, pe, ps = [], [], [], [], []
                for k in range(1, min(maxstep, n) + 1):
                    c = n - k
                    if c not in self.levels or len(self.levels[c][0]) == 0:
                        continue
                    st = states_of(c); groups = code_groups(c)
                    for j in order:
                        wcj = int(self.wc[j])
                        if wcj < k:
                            continue
                        # the step cost depends only on the parent's first-run code: pick the rows whose code gives k
                        sel = [g for code, g in groups.items() if max(1, wcj - int(SAV[self.wlast[j], code])) == k]
                        if not sel:
                            continue
                        allrows = sel[0] if len(sel) == 1 else np.sort(np.concatenate(sel))
                        for lo in range(0, len(allrows), self.CHUNK):
                            rows = allrows[lo:lo + self.CHUNK]
                            stc = np.ascontiguousarray(np.asarray(st[rows]))
                            if keep:
                                s2, kk = spec.apply(stc, self.inv_elems[j], threads)
                            else:
                                kk = spec.keys(stc, self.inv_elems[j], threads); s2 = None
                            new = self.seen.insert(kk); fi = np.where(new)[0]
                            if not len(fi):
                                continue
                            pk.append(kk[fi]); pp.append(rows[fi].astype(np.int32)); pw.append(np.full(len(fi), j, np.int16))
                            pe.append(np.full(len(fi), k, np.int8))
                            if keep:
                                ps.append(s2[fi])
                cat = lambda xs, dt: np.concatenate(xs) if xs else np.empty(0, dt)
                self.levels[n] = (cat(pk, np.uint64), cat(pp, np.int32), cat(pw, np.int16), cat(pe, np.int8))
                if keep:
                    self.bstates[n] = np.concatenate(ps) if ps else np.empty((0, SB), np.int8)
                total += len(self.levels[n][0]); self.back_cost_reached = n
                old = n - maxstep + 1                                 # sources below this are no longer needed
                for c in [c for c in groups_of if c < old]:
                    del groups_of[c]
                for c in [c for c in src if c < old and c != 0]:
                    del src[c]
                    if c in src_files:
                        os.unlink(src_files.pop(c))
                if verbose:
                    print(f"   backward cost {n}: {len(self.levels[n][0])} cosets ({time.time() - t:.0f}s)", flush=True)
                if all(len(self.levels[m][0]) == 0 for m in range(max(0, n - maxstep + 1), n + 1)):
                    break                                             # no level left that can feed a later one
        finally:
            for f in src_files.values():
                try:
                    os.unlink(f)
                except OSError:
                    pass
        self.seen = None                       # the dedup set is only needed during the build; free it before the flat copies
        self.levels = {c: v for c, v in self.levels.items() if len(v[0])}
        self.bstates = {}
        self.back_complete = self.back_cost_reached
        if compact:
            assert self.back_complete <= CMASK, "compact tables store the cost in 5 bits"
            total = sum(len(v[0]) for v in self.levels.values()); ck = np.empty(total, np.uint64); off = 0
            for c in sorted(self.levels):
                keys = self.levels[c][0]; self.levels[c] = None          # free the parent pointers level by level
                ck[off:off + len(keys)] = (keys & ~np.uint64(CMASK)) | np.uint64(c); off += len(keys); del keys
            self.levels = None; self.bstates = {}
            ck.sort()                                                    # in place
            self.flat_keys, self.flat_cost, self.flat_idx = ck, None, None
        else:
            self.flat_keys = np.concatenate([self.levels[c][0] for c in sorted(self.levels)])
            self.flat_cost = np.concatenate([np.full(len(self.levels[c][0]), c, np.int16) for c in sorted(self.levels)])
            self.flat_idx = np.concatenate([np.arange(len(self.levels[c][0]), dtype=np.int32) for c in sorted(self.levels)])
            o = np.argsort(self.flat_keys, kind="stable")
            self.flat_keys, self.flat_cost, self.flat_idx = self.flat_keys[o], self.flat_cost[o], self.flat_idx[o]
        self.total_back = len(self.flat_keys)
        if verbose:
            print(f"   backward table: {self.total_back} cosets, complete to cost {self.back_complete} ({time.time() - t:.0f}s)", flush=True)
        if cache:
            self._save_cache(cache)

    # ------------------------------------------------------------------ disk cache of the backward table
    def _cache_key(self, k0, back_cost, max_cosets):
        import hashlib
        probe = pack_states([M.apply_seq(M.State(), w) for w in ("U", "R", "F", "U R F", "R2 U F'", "F U2 R' U")])
        ident = ["v3-target-order", [(n, int(c)) for n, e, c in self.words], [int(i) for i in self.back_idx], int(back_cost), int(max_cosets), bool(self.lean),
                 [int(x) for x in k0], [int(x) for x in self.spec.keys(probe, threads=self.threads)]]
        if self.compact:
            ident.append("compact-v1")
        if getattr(self, "max_bytes", None):
            ident.append("budget-v2")
        return hashlib.sha1(repr(ident).encode()).hexdigest()

    def _save_cache(self, path):
        import json, os
        tmp = path + ".tmp"
        os.makedirs(tmp, exist_ok=True)
        if self.compact:
            np.save(f"{tmp}/compact_keys.npy", self.flat_keys)
            json.dump(dict(hash=self._cache_hash, compact=True, back_complete=int(self.back_complete),
                           back_cost_reached=int(self.back_cost_reached)), open(f"{tmp}/meta.json", "w"))
        else:
            for c, (keys, par, wid, ec) in self.levels.items():
                np.save(f"{tmp}/k{c}.npy", keys); np.save(f"{tmp}/p{c}.npy", par); np.save(f"{tmp}/w{c}.npy", wid); np.save(f"{tmp}/e{c}.npy", ec)
            np.save(f"{tmp}/flat_keys.npy", self.flat_keys); np.save(f"{tmp}/flat_cost.npy", self.flat_cost); np.save(f"{tmp}/flat_idx.npy", self.flat_idx)
            json.dump(dict(hash=self._cache_hash, costs=sorted(self.levels), back_complete=int(self.back_complete),
                           back_cost_reached=int(self.back_cost_reached)), open(f"{tmp}/meta.json", "w"))
        if os.path.isdir(path):
            import shutil; shutil.rmtree(path)
        os.replace(tmp, path)

    def _load_cache(self, path):
        import json, os
        if not os.path.isfile(f"{path}/meta.json"):
            return False
        meta = json.load(open(f"{path}/meta.json"))
        if meta.get("hash") != self._cache_hash:
            return False
        if meta.get("compact"):
            self.levels = None; self.bstates = {}
            self.flat_keys = np.load(f"{path}/compact_keys.npy"); self.flat_cost = self.flat_idx = None
            self.back_complete = meta["back_complete"]; self.back_cost_reached = meta["back_cost_reached"]
            self.total_back = len(self.flat_keys)
            return True
        self.levels = {int(c): (np.load(f"{path}/k{c}.npy"), np.load(f"{path}/p{c}.npy"), np.load(f"{path}/w{c}.npy"), np.load(f"{path}/e{c}.npy")) for c in meta["costs"]}
        self.bstates = {}
        self.flat_keys = np.load(f"{path}/flat_keys.npy"); self.flat_cost = np.load(f"{path}/flat_cost.npy"); self.flat_idx = np.load(f"{path}/flat_idx.npy")
        self.back_complete = meta["back_complete"]; self.back_cost_reached = meta["back_cost_reached"]
        self.total_back = len(self.flat_keys)
        return True

    CHUNK = 4_000_000

    def _materialise(self, c, rows=None):
        """States of the given rows of backward bucket c (all rows if None), regenerated from the goal states
        through the parent chain.  Two passes over the costs below c: top-down to collect which rows of each
        bucket are needed (each bucket visited once, so the work is linear in the chain depth), then bottom-up to
        regenerate exactly those rows; memory is bounded by the rows requested times the chain depth."""
        if c == 0:
            return self._goal_states if rows is None else self._goal_states[rows]
        rows = np.arange(len(self.levels[c][0]), dtype=np.int64) if rows is None else np.asarray(rows, np.int64)
        need = {}; pending = {c: rows}; order = []
        while pending:
            cost = max(pending); r = pending.pop(cost)            # every request to `cost` has arrived (they come from above)
            need[cost] = r; order.append(cost)
            if cost == 0:
                continue
            keys, par, wid, ec = self.levels[cost]; p = par[r]; pcs = cost - ec[r].astype(np.int64)
            for pc in np.unique(pcs):
                sel = np.unique(p[pcs == pc]); pc = int(pc)
                pending[pc] = np.union1d(pending[pc], sel) if pc in pending else sel
        states = {}
        for cost in reversed(order):                               # ascending cost
            r = need[cost]
            if cost == 0:
                states[0] = self._goal_states[r]; continue
            keys, par, wid, ec = self.levels[cost]; w = wid[r]; p = par[r]; pcs = cost - ec[r].astype(np.int64)
            out = np.empty((len(r), SB), np.int8)
            for j in np.unique(w):
                for pc in np.unique(pcs[w == j]):
                    sel = np.where((w == j) & (pcs == pc))[0]; pc = int(pc)
                    idx = np.searchsorted(need[pc], p[sel])
                    out[sel] = self.spec.apply(states[pc][idx], self.inv_elems[int(j)], self.threads)[0]
            states[cost] = out
        return states[c]

    def _table_cost(self, keys):
        """(cost or -1, position, hit) for each label, in either table format."""
        keys = np.asarray(keys, np.uint64); n = len(self.flat_keys)
        if self.compact:
            kk = keys & ~np.uint64(CMASK); p = np.searchsorted(self.flat_keys, kk); p = np.minimum(p, n - 1)
            hit = (self.flat_keys[p] & ~np.uint64(CMASK)) == kk
            return np.where(hit, (self.flat_keys[p] & np.uint64(CMASK)).astype(np.int64), -1), p, hit
        p = np.searchsorted(self.flat_keys, keys); p[p >= n] = 0; hit = self.flat_keys[p] == keys
        return np.where(hit, self.flat_cost[p].astype(np.int64), -1), p, hit

    def _descend(self, s, c, req=None, memo=None):
        """Compact tables: a tail of word indices from state s (packed, or a model State) whose class is stored at
        cost c, into the target, found by stepping to a stored neighbour whose cost is lower by exactly the step's
        merge-aware cost.  req = (last-run code, cost, step cost) of the word chosen one level up: that step's
        cost depends on the first word chosen here, as it did when the table was built.  None if no consistent tail
        exists (a false match).  memo is shared across the matches of one solve."""
        if not isinstance(s, np.ndarray):
            s = pack_states([s])
        memo = self._memo if memo is None else memo
        key = (s.tobytes(), c, req)
        if key in memo:
            return memo[key]
        if c == 0:
            ok = req is None or max(1, req[1] - int(SAV[req[0], 24])) == req[2]
            memo[key] = [] if ok else None
            return memo[key]
        out = None
        js = self.back_idx
        if req is not None:                             # the junction with the word chosen one level up must cost req[2]
            ok = np.maximum(1, req[1] - SAV[req[0], self.wfirst[js]]) == req[2]
            js = js[ok]
        js = js[self.wc[js] >= 1]
        if len(js):
            # every child in one C call, every child's stored cost in one lookup (rung 3 has 30,000 words)
            kids = apply_seqs(s, self.elems, [[int(j)] for j in js])
            costs = self._table_cost(self.spec.keys(kids, threads=self.threads))[0]
            wcs = self.wc[js]
            good = np.where((costs >= 0) & (costs < c) & (c - costs <= wcs))[0]
            for g in good[np.argsort(costs[good], kind="stable")]:
                j = int(js[g]); c2 = int(costs[g]); wcj = int(wcs[g])
                rest = self._descend(kids[g:g + 1], c2, (int(self.wlast[j]), wcj, c - c2), memo)
                if rest is not None:
                    out = [j] + rest; break
        memo[key] = out
        return out

    def suffix_idx(self, cost, idx):
        out = []
        while cost > 0:
            keys, par, wid, ec = self.levels[cost]
            j = int(wid[idx]); out.append(j)
            idx, cost = int(par[idx]), cost - int(ec[idx])
        return out

    def suffix(self, cost, idx):
        return [self.words[j][0] for j in self.suffix_idx(cost, idx)]

    def _merged_cost(self, idx):
        """Exact merged HTM of a sequence of word indices (runs merged across every junction, mod 8)."""
        out = []
        for j in idx:
            for f, k in self.wruns[j]:
                if out and out[-1][0] == f:
                    k2 = (out[-1][1] + k) % 8
                    if k2:
                        out[-1] = (f, k2)
                    else:
                        out.pop()
                else:
                    out.append((f, k))
        return sum(HTM_RUN[k] for f, k in out)

    def _prefix_cost(self, prefix):
        """Merged cost of a forward word sequence: word costs minus junction savings, at least 1 per step."""
        tot = 0; last = -1
        for j in prefix:
            tot += max(1, int(self.wc[j]) - (int(SAV[self.wlast[last], self.wfirst[j]]) if last >= 0 else 0)); last = j
        return tot

    def _try(self, st, prefix, hit_pos):
        """Verify matches (prefix: list of word indices applied to st, -1 = padding); returns (total, pre, suf, state) or None.
        Compact tables defer the work: the match is recorded with an upper bound on its total (merges only shorten),
        and _resolve rebuilds tails only for matches that can still be the best or fall inside the window."""
        prefix = [int(i) for i in prefix if int(i) >= 0]
        if self.compact and not getattr(self, "_resolving", False):
            best = None
            for pos in hit_pos:
                est = self._prefix_cost(prefix) + int(self.flat_keys[pos] & np.uint64(CMASK))
                self._pending.append((est, prefix, int(pos)))
                if best is None or est < best[0]:
                    best = (est, None, None, None)
            return best
        best = None
        for pos in hit_pos:
            if self.compact:
                s_pre = st
                for i in prefix:
                    s_pre = s_pre.apply(self.words[int(i)][1])
                sidx = self._descend(s_pre, int(self.flat_keys[pos] & np.uint64(CMASK)))
                if sidx is None:
                    continue                            # false match (labels equal above the cost bits only)
            else:
                sidx = self.suffix_idx(int(self.flat_cost[pos]), int(self.flat_idx[pos]))
            # exact merged HTM of the whole solution, from precomputed runs (a junction saving can overlap merges
            # already made on either side when three runs of one face meet, so the per-step accounting is only used
            # inside the tables; the reported total is the truth)
            total = self._merged_cost(prefix + sidx)
            lim = getattr(self, "_best_total", None)
            if lim is not None and total > lim + getattr(self, "window", 0) and (best is None or total >= best[0]):
                continue                                # cannot be the best or inside the window: skip the state
            # the state is built later, in one C call for every kept result (_finish_results)
            suf = [self.words[j][0] for j in sidx]
            cand = (total, [self.words[int(i)][0] for i in prefix], suf, None)
            self._seq_of[id(cand)] = prefix + sidx
            if True:
                improves = lim is None or total < lim
                if improves:
                    self._best_total = total
                if getattr(self, "window", 0) and (len(self.candidates) < self.max_candidates or improves):
                    self.candidates.append(cand)           # a new best always joins the list, even past the cap
                if best is None or total < best[0]:
                    best = cand
        return best

    def lookup(self, st):
        """Table-only solve for a complete table (every class stored): the stored path from st's class, with no
        forward search.  Returns the same tuple as solve(), or None if the class is not in the table."""
        k = self.spec.keys(pack_states([st]), threads=self.threads)
        cost, p, hit = self._table_cost(k)
        if not hit[0]:
            return None
        self.window = 0; self._memo = {}; self._best_total = None; self._seq_of = {}; self.candidates = []
        self._resolving = True
        try:
            r = self._try(st, [], p)
        finally:
            self._resolving = False
        return self._finish_results(st, r)[0]

    def solve(self, st, count_all=False, window=0):
        """count_all: do not stop early; explore the whole budget and count every match (for calibrating budgets).
        window: also keep every verified match whose total is within `window` of the best, in self.candidates
        (list of (total, prefix words, suffix words, state)), e.g. for choosing by the next rung's cost."""
        self.hits = 0
        best = None
        self.count_all = count_all
        self.window = window
        self.candidates = []
        self._pending = []; self._chunks = []; self._memo = {}; self._best_total = None; self._seq_of = {}
        start = pack_states([st]); k = self.spec.keys(start, threads=self.threads)
        seen = KeySet(1 << 16); seen.insert(k)
        buckets = {0: (start, np.zeros((1, 0), np.int16))}       # forward prefixes: word indices (< 32k words), int16 to keep 1e8 stored states affordable
        _, pos, hit0 = self._table_cost(k)
        if hit0[0]:
            r = self._try(st, [], pos); best = r if r else best
        # forward levels by TARGET cost, as in the table build: level n is generated from level n-k with exactly the
        # rows whose merge-aware step cost is k, probed as it is created, and frozen before level n+1; max_fwd stops
        # the generation, and every generated level is complete
        stored = 1; complete = 0
        fmax = int(self.wc[self.fwd_idx].max()) if len(self.fwd_idx) else 1
        lastw_of = {}

        def lastw(c):
            if c not in lastw_of:
                lastw_of[c] = _last_word(buckets[c][1])
            return lastw_of[c]

        fgroups_of = {}

        def fgroups(c):
            # forward rows of level c grouped by the last-run code of their last word, computed once per level
            if c not in fgroups_of:
                codes = self.wlast[lastw(c)]
                o = np.argsort(codes, kind="stable"); sc = codes[o]
                cuts = np.flatnonzero(np.diff(sc)) + 1
                fgroups_of[c] = {int(sc[a]): o[a:b] for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(sc)])} if len(sc) else {}
            return fgroups_of[c]

        for n in range(1, self.fwd_store + 1):
            if best is not None and n - MAXSAV >= best[0] + self.window and not self.count_all:
                break
            if stored > self.max_fwd:
                break
            sizes = [len(buckets[c][0]) if c in buckets else 0 for c in (n - 1, n - 2)]
            if sizes[0] and sizes[1] and stored + sizes[0] * sizes[0] / sizes[1] > self.max_fwd:
                break                                  # the next level would overshoot the cap: max_fwd is an upper bound
            # (filling the last level partially up to the cap was measured on rung 7: same misses, +18% time)
            ps, pp = [], []
            for k in range(1, min(fmax, n) + 1):
                c = n - k
                if c not in buckets or len(buckets[c][0]) == 0:
                    continue
                cur, pre = buckets[c]; groups = fgroups(c)
                for j in self.fwd_idx:
                    j = int(j); wcj = int(self.wc[j])
                    if wcj < k:
                        continue
                    sel = [g for code, g in groups.items() if max(1, wcj - int(SAV[code, self.wfirst[j]])) == k]
                    if not sel:
                        continue
                    rows = sel[0] if len(sel) == 1 else np.sort(np.concatenate(sel))
                    s, kk = self.spec.apply(cur[rows], self.elems[j], self.threads)
                    new = seen.insert(kk); fi = np.where(new)[0]
                    if not len(fi):
                        continue
                    npre = np.hstack([pre[rows[fi]], np.full((len(fi), 1), j, np.int16)])
                    ps.append(s[fi]); pp.append(npre)
                    kf = kk[fi]
                    tc, p, hm = self._table_cost(kf)
                    hit = np.where(hm)[0]; self.hits += len(hit)
                    if len(hit) and self.compact:
                        # compact tables: record every match in bulk (no Python per match); forward cost is exactly n,
                        # so est = n + stored cost is the same bound _try would compute, and est - MAXSAV a lower bound
                        best = self._record(n + tc[hit], p[hit], npre[hit], -1, best)
                    elif len(hit):
                        est = n + tc[hit]                          # forward cost (exactly n) + stored cost: an upper bound
                        for h in hit[np.argsort(est, kind="stable")]:
                            if best is not None and n + int(tc[h]) - MAXSAV > best[0] + self.window and not self.count_all:
                                break                              # sorted: no later match can be within reach
                            r = self._try(st, npre[h], [p[h]])
                            if r and (best is None or r[0] < best[0]):
                                best = r
            if ps:
                w = max(x.shape[1] for x in pp)
                pp = [np.hstack([x, np.full((len(x), w - x.shape[1]), -1, np.int16)]) if x.shape[1] < w else x for x in pp]
                buckets[n] = (np.concatenate(ps), np.vstack(pp)); stored += len(buckets[n][0])
            complete = n
            if self.verbose:
                print(f"   forward cost {n}: {len(buckets[n][0]) if n in buckets else 0} cosets", flush=True)
        buckets = {c: v for c, v in buckets.items() if c <= complete}
        pairs = sorted(((c + int(self.wc[j]), c, j) for c in buckets for j in self.sweep_idx if c + int(self.wc[j]) > complete))
        for rc, c, j in pairs:
            if best is not None and rc - 2 * MAXSAV >= best[0] + self.window and not self.count_all:
                break
            cur, pre = buckets[c]
            # only matches that can still come within the window of the best are recorded (in C), so the hit buffer
            # never truncates the useful ones; they are then tried cheapest first
            maxcost = -1 if (best is None or self.count_all) else best[0] + self.window + 2 * MAXSAV - rc
            if best is not None and not self.count_all and maxcost < 0:
                continue                                   # nothing stored is cheap enough to help
            done, he, hi, hp = self.spec.sweep(cur, self.elems[j:j + 1], self.flat_keys, want=10 ** 9, cap=2_000_000, threads=self.threads,
                                               cmask=CMASK if self.compact else 0,
                                               costs=None if self.compact else self.flat_cost, maxcost=maxcost)
            self.hits += len(he)
            if not len(hp):
                continue
            tcs = (self.flat_keys[hp] & np.uint64(CMASK)).astype(np.int64) if self.compact else self.flat_cost[hp].astype(np.int64)
            if self.compact:
                # exact forward cost of each matched prefix: c, plus word j's step cost after the row's last word
                hi = np.asarray(hi, np.int64); lw = lastw(c)[hi].astype(np.int64)
                sav = np.where(lw >= 0, SAV[self.wlast[lw], self.wfirst[j]], 0)
                step = np.maximum(1, int(self.wc[j]) - sav.astype(np.int64))
                best = self._record(c + step + tcs, np.asarray(hp, np.int64), pre[hi], j, best)
                continue
            for o in np.argsort(tcs, kind="stable"):
                tot = rc + int(tcs[o])
                if best is not None and tot - 2 * MAXSAV >= best[0] + self.window and not self.count_all:
                    break
                r = self._try(st, list(pre[hi[o]]) + [j], [hp[o]])
                if r and (best is None or r[0] < best[0]):
                    best = r
        if self.compact:
            best = self._resolve(st)
        if self.window and best is not None:
            self.candidates = [c for c in self.candidates if c[0] <= best[0] + self.window]
        best, self.candidates = self._finish_results(st, best)
        if not self.window:
            self.candidates = []
        return best

    def _finish_results(self, st, best):
        """Build the states of the best result and the window candidates in one C call, verify them, and return
        (best, candidates) with states filled in; results that fail verification are dropped."""
        if best is None:
            return None, []
        need = [best] + [c for c in self.candidates if c is not best]
        need = [c for c in need if id(c) in self._seq_of]
        if not need:
            return None, []
        states = apply_seqs(pack_states([st]), self.elems, [self._seq_of[id(c)] for c in need])
        done = {}
        for c, row in zip(need, states):
            s2 = unpack_state(row)
            if self.verify(s2):
                done[id(c)] = (c[0], c[1], c[2], s2)
        cands = [done[id(c)] for c in self.candidates if id(c) in done]
        b = done.get(id(best))
        if b is None:                                   # the best failed verification (never expected): next valid one
            vals = sorted(done.values(), key=lambda c: c[0]); b = vals[0] if vals else None
        return b, cands

    def _record(self, est, pos, pre, j, best):
        """Compact tables: record a batch of matches (bounds est, table positions pos, forward prefix rows pre, plus
        word j appended if j >= 0) and return the updated bound.  Matches that can no longer come within the window
        of the bound are dropped here: est - MAXSAV is a lower bound on a match's total (only the junction between
        the forward prefix and the table's tail can still merge), and the bound is an upper bound on the best."""
        est = np.asarray(est, np.int64)
        # per-match lower bound: only the junction between the prefix's last word and the tail's first word can still
        # merge, and its saving is at most the largest entry of SAV's row for that last run
        lastw = np.full(len(est), j, np.int64) if j >= 0 else _last_word(pre)
        lb = est - SAV_ROWMAX[self.wlast[lastw]]
        b = int(est.min())
        if best is None or b < best[0]:
            best = (b, None, None, None)
        if not self.count_all:
            keep = lb <= best[0] + self.window
            if not keep.all():
                est, lb, pos, pre = est[keep], lb[keep], pos[keep], pre[keep]
        if len(est):
            self._chunks.append((est, pos, pre, int(j), lb))
        return best

    def _resolve(self, st):
        """Compact tables: rebuild tails for the recorded matches, cheapest bound first, until no recorded match can
        still beat the best (or reach the window); returns the best exact result."""
        self._resolving = True; best = None
        try:
            chunks = list(self._chunks)
            if self._pending:                              # matches recorded one at a time by _try
                e0 = np.array([e for e, _, _ in self._pending], np.int64)
                chunks.append((e0, np.array([q for _, _, q in self._pending], np.int64), [pf for _, pf, _ in self._pending], -1, e0 - MAXSAV))
            if not chunks:
                return None
            est = np.concatenate([ch[0] for ch in chunks]); lbs = np.concatenate([ch[4] for ch in chunks])
            cid = np.concatenate([np.full(len(ch[0]), k, np.int64) for k, ch in enumerate(chunks)])
            iid = np.concatenate([np.arange(len(ch[0])) for ch in chunks])
            for o in np.argsort(est, kind="stable"):
                # est is the forward cost plus the stored cost; only the junction between the two can still merge
                if best is not None and int(est[o]) - MAXSAV > best[0] + self.window:
                    break
                full = best is not None and len(self.candidates) >= self.max_candidates
                if full and int(est[o]) - MAXSAV >= best[0]:
                    break                                  # candidate list full and nothing left can strictly beat the best
                if best is not None and (int(lbs[o]) > best[0] + self.window or (full and int(lbs[o]) >= best[0])):
                    continue                               # this match's own bound rules it out
                ch = chunks[cid[o]]; i = int(iid[o])
                prefix = [int(x) for x in ch[2][i] if int(x) >= 0] + ([ch[3]] if ch[3] >= 0 else [])
                r = self._try(st, prefix, [int(ch[1][i])])
                if r and (best is None or r[0] < best[0]):
                    best = r
        finally:
            self._resolving = False; self._pending = []; self._chunks = []
        return best
