/* Full-table kernels: labels, joins, chaining, slices and probes on 50-point maps (fulltab/ftlib.py wraps them).
 *
 * States are 100-byte arrays as in coset.c.  Labels are the stabiliser-chain canonical form of the left coset T.x,
 * hashed to 64 bits, computed in the 50-point form: one byte per piece, slot*4 + orientation of its orientation-0
 * point; the other points follow (same slot, orientation + o mod 3 for corners, mod 4 otherwise).  The map stays in
 * four NEON registers; composition and every canonical-form level are one 64-byte table lookup per 16 lanes plus an
 * orientation add.  Two labels are computed at once so their dependency chains interleave.
 */
#include "../csrc/coset.c"
#include <stdatomic.h>
#include <arm_neon.h>
#include <time.h>
#include <stdio.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static inline void inv_elem(const int8_t *e, int8_t *f) {
    for (int j = 0; j < NC; j++) f[OCP + e[OCP + j]] = (int8_t)j;
    for (int j = 0; j < NR; j++) f[ORP + e[ORP + j]] = (int8_t)j;
    for (int j = 0; j < NG; j++) f[OGP + e[OGP + j]] = (int8_t)j;
    for (int i = 0; i < NC; i++) f[OCO + i] = (int8_t)((3 - e[OCO + f[OCP + i]]) % 3);
    for (int i = 0; i < NR; i++) f[ORO + i] = (int8_t)((4 - e[ORO + f[ORP + i]]) & 3);
    for (int i = 0; i < NG; i++) f[OGO + i] = (int8_t)((4 - e[OGO + f[OGP + i]]) & 3);
}

/* ================================================= 50-point form ================================================== */
typedef struct {
    int ncan, m[64], off[64], sv[64];
    uint8_t *ub, *uo;                               /* per transversal permutation: 64 lanes (source piece, orientation) */
    uint8_t sb[64][64], so[64][64], spad[64][64];    /* per level: orbit points (piece, orientation), pad lanes 0xFF */
} fs_t;
static inline void pt_decode(int p, uint8_t *b, uint8_t *o) {
    if (p < 24) { *b = (uint8_t)(p / 3); *o = (uint8_t)(p % 3); }
    else if (p < 96) { *b = (uint8_t)(8 + (p - 24) / 4); *o = (uint8_t)((p - 24) % 4); }
    else { *b = (uint8_t)(26 + (p - 96) / 4); *o = (uint8_t)((p - 96) % 4); }
}
static inline int base_point(int b) { return b < 8 ? b * 3 : b < 26 ? 24 + (b - 8) * 4 : 96 + (b - 26) * 4; }
fs_t *fs_new(const spec_t *sp) {
    fs_t *F = calloc(1, sizeof *F); F->ncan = sp->ncan; int tot = 0;
    if (sp->ncan > 64) { fprintf(stderr, "too many levels\n"); abort(); }
    for (int l = 0; l < sp->ncan; l++) {
        F->m[l] = sp->can_m[l]; F->off[l] = sp->can_off[l];
        if (F->m[l] > 64) { fprintf(stderr, "orbit too big for the vector scan\n"); abort(); }
        if (F->off[l] + F->m[l] > tot) tot = F->off[l] + F->m[l];
    }
    F->ub = malloc((size_t)tot * 64); F->uo = malloc((size_t)tot * 64);
    for (int i = 0; i < tot; i++) {
        const uint8_t *u = sp->can_perm + (size_t)i * NPTS;
        for (int b = 0; b < 64; b++) {
            if (b < 50) pt_decode(u[base_point(b)], &F->ub[(size_t)i * 64 + b], &F->uo[(size_t)i * 64 + b]);
            else { F->ub[(size_t)i * 64 + b] = 0xFF; F->uo[(size_t)i * 64 + b] = 0; }
        }
    }
    for (int l = 0; l < sp->ncan; l++) {
        F->sv[l] = (F->m[l] + 15) / 16;
        for (int t = 0; t < 64; t++) {
            if (t < F->m[l]) { pt_decode(sp->can_pts[F->off[l] + t], &F->sb[l][t], &F->so[l][t]); F->spad[l][t] = 0; }
            else { F->sb[l][t] = 0xFF; F->so[l][t] = 0; F->spad[l][t] = 0xFF; }
        }
    }
    return F;
}
static inline void pm50(const int8_t *s, uint8_t *P) {             /* lanes 50..63 are zero */
    memset(P + 48, 0, 16);
    for (int j = 0; j < NC; j++) P[s[OCP + j]] = (uint8_t)(j * 4 + s[OCO + j]);
    for (int j = 0; j < NR; j++) P[8 + s[ORP + j]] = (uint8_t)((8 + j) * 4 + s[ORO + j]);
    for (int j = 0; j < NG; j++) P[26 + s[OGP + j]] = (uint8_t)((26 + j) * 4 + s[OGO + j]);
}
static inline void pm50_inv(const uint8_t *P, uint8_t *Q) {       /* piece b -> (b', o') gives piece b' -> (b, -o') */
    memset(Q + 48, 0, 16);
    for (int b = 0; b < 50; b++) { int c = P[b], bb = c >> 2, o = c & 3, m = b < 8 ? 3 : 4; Q[bb] = (uint8_t)(b * 4 + (m - o) % m); }
}
static const uint8_t MOD3[16] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
static const uint16_t IDX16[64] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
                                   32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63};
/* r[b] = t at (piece ib[b], orientation io[b]) */
static inline uint8x16x4_t step50(uint8x16x4_t t, const uint8x16_t ib[4], const uint8x16_t io[4]) {
    const uint8x16_t hi = vdupq_n_u8(0xFC), lo = vdupq_n_u8(3), mod3 = vld1q_u8(MOD3);
    const uint8x16_t cmask = vcombine_u8(vdup_n_u8(0xFF), vdup_n_u8(0));
    uint8x16x4_t r;
    for (int k = 0; k < 4; k++) {
        uint8x16_t y = vqtbl4q_u8(t, ib[k]), sm = vaddq_u8(vandq_u8(y, lo), io[k]);
        uint8x16_t v = vorrq_u8(vandq_u8(y, hi), vandq_u8(sm, lo));
        if (k == 0) v = vbslq_u8(cmask, vorrq_u8(vandq_u8(y, hi), vqtbl1q_u8(mod3, sm)), v);
        r.val[k] = v;
    }
    return r;
}
static inline uint8x16x4_t step50p(uint8x16x4_t t, const uint8_t *ub, const uint8_t *uo) {
    uint8x16_t ib[4] = {vld1q_u8(ub), vld1q_u8(ub + 16), vld1q_u8(ub + 32), vld1q_u8(ub + 48)};
    uint8x16_t io[4] = {vld1q_u8(uo), vld1q_u8(uo + 16), vld1q_u8(uo + 32), vld1q_u8(uo + 48)};
    return step50(t, ib, io);
}
/* the map of apply1(p, q) from P = pm50(p), Q = pm50(q): V[b] = Q at P[b] */
static inline uint8x16x4_t compose50(const uint8_t *Q, const uint8_t *P) {
    uint8x16_t ib[4], io[4];
    for (int k = 0; k < 4; k++) { uint8x16_t c = vld1q_u8(P + 16 * k); ib[k] = vshrq_n_u8(c, 2); io[k] = vandq_u8(c, vdupq_n_u8(3)); }
    ib[3] = vorrq_u8(ib[3], vcombine_u8(vcreate_u8(0xFFFFFFFFFFFF0000ULL), vdup_n_u8(0xFF)));   /* pad lanes read 0 */
    return step50(vld1q_u8_x4(Q), ib, io);
}
/* minimum over level l's orbit points of (value << 6 | index) */
static inline uint16_t scan_min(uint8x16x4_t t, const fs_t *F, int l) {
    const uint8x16_t hi = vdupq_n_u8(0xFC), lo = vdupq_n_u8(3), mod3 = vld1q_u8(MOD3), eight = vdupq_n_u8(8);
    uint16_t mn = 0xFFFF;
    for (int k = 0; k < F->sv[l]; k++) {
        uint8x16_t ib = vld1q_u8(F->sb[l] + 16 * k), io = vld1q_u8(F->so[l] + 16 * k);
        uint8x16_t y = vqtbl4q_u8(t, ib), sm = vaddq_u8(vandq_u8(y, lo), io);
        uint8x16_t r4 = vorrq_u8(vandq_u8(y, hi), vandq_u8(sm, lo)), r3 = vorrq_u8(vandq_u8(y, hi), vqtbl1q_u8(mod3, sm));
        uint8x16_t v = vorrq_u8(vbslq_u8(vcltq_u8(ib, eight), r3, r4), vld1q_u8(F->spad[l] + 16 * k));
        uint16x8_t a = vorrq_u16(vshlq_n_u16(vmovl_u8(vget_low_u8(v)), 6), vld1q_u16(IDX16 + 16 * k));
        uint16x8_t b = vorrq_u16(vshlq_n_u16(vmovl_u8(vget_high_u8(v)), 6), vld1q_u16(IDX16 + 16 * k + 8));
        uint16_t m = vminvq_u16(vminq_u16(a, b)); if (m < mn) mn = m;
    }
    return mn;
}
static inline uint64_t hash50(uint8x16x4_t t) {
    /* every word passes through a full 64-bit mixer, so differences in one word cannot be cancelled by another */
    uint64_t w[8]; vst1q_u8_x4((uint8_t *)w, t);
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 7; i++) h = fin(h ^ w[i]) + (w[i] * 0xC2B2AE3D27D4EB4FULL);
    return fin(h);
}
static inline uint64_t canon50(uint8x16x4_t t, const fs_t *F) {
    for (int l = 0; l < F->ncan; l++) {
        int best = scan_min(t, F, l) & 63, off = F->off[l];
        t = step50p(t, F->ub + (size_t)(off + best) * 64, F->uo + (size_t)(off + best) * 64);
    }
    return hash50(t);
}
static inline void canon50x2(uint8x16x4_t ta, uint8x16x4_t tb, const fs_t *F, uint64_t *ka, uint64_t *kb) {
    for (int l = 0; l < F->ncan; l++) {
        int ba = scan_min(ta, F, l) & 63, bb = scan_min(tb, F, l) & 63, off = F->off[l];
        ta = step50p(ta, F->ub + (size_t)(off + ba) * 64, F->uo + (size_t)(off + ba) * 64);
        tb = step50p(tb, F->ub + (size_t)(off + bb) * 64, F->uo + (size_t)(off + bb) * 64);
    }
    *ka = hash50(ta); *kb = hash50(tb);
}
static inline uint64_t label50(const int8_t *s, const fs_t *F) {
    uint8_t P[64] __attribute__((aligned(16))); pm50(s, P); return canon50(vld1q_u8_x4(P), F);
}
typedef struct { const int8_t *s; int64_t n; const fs_t *F; uint64_t *out; _Atomic int64_t next; } lab50_t;
static void *lab50_worker(void *a) {
    lab50_t *L = a; uint8_t P[64] __attribute__((aligned(16))), Q[64] __attribute__((aligned(16)));
    for (;;) {
        int64_t i0 = atomic_fetch_add(&L->next, 4096); if (i0 >= L->n) break;
        int64_t i1 = i0 + 4096 < L->n ? i0 + 4096 : L->n, i = i0;
        for (; i + 1 < i1; i += 2) {
            pm50(L->s + i * SB, P); pm50(L->s + (i + 1) * SB, Q);
            canon50x2(vld1q_u8_x4(P), vld1q_u8_x4(Q), L->F, &L->out[i], &L->out[i + 1]);
        }
        if (i < i1) L->out[i] = label50(L->s + i * SB, L->F);
    }
    return NULL;
}
void labels50(const int8_t *s, int64_t n, const fs_t *F, uint64_t *out, int threads) {
    lab50_t L = {s, n, F, out, 0}; pthread_t th[64];
    for (int t = 0; t < threads; t++) pthread_create(&th[t], NULL, lab50_worker, &L);
    for (int t = 0; t < threads; t++) pthread_join(th[t], NULL);
}

/* ===================================== the front/back mirror on 50-point maps ======================================
 * Piece b at (slot s, orientation o) goes to piece M[b] at slot M[s] with orientation (-o + OFF[M[b]] - OFF[M[s]]) mod
 * m (m = 3 for corners, 4 otherwise); M is the F<->B relabelling of slots/pieces within each kind, OFF the fitted spin
 * offsets (fulltab/mirrorFB.py).  Lanes 50..63 stay 0. */
typedef struct { uint8_t M[64], OFF[64]; } mir_t;
mir_t *mir_new(const uint8_t *M, const uint8_t *OFF) {
    mir_t *R = calloc(1, sizeof *R); memcpy(R->M, M, 50); memcpy(R->OFF, OFF, 50);
    for (int b = 50; b < 64; b++) { R->M[b] = 0xFF; R->OFF[b] = 0; }
    return R;
}
static inline uint8x16x4_t mirror50(uint8x16x4_t P, const mir_t *R) {
    uint8x16x4_t M = vld1q_u8_x4(R->M), OFF = vld1q_u8_x4(R->OFF); uint8x16x4_t V;
    const uint8x16_t lo = vdupq_n_u8(3), four = vdupq_n_u8(4), mod3 = vld1q_u8(MOD3);
    const uint8x16_t cmask = vcombine_u8(vdup_n_u8(0xFF), vdup_n_u8(0));
    for (int k = 0; k < 4; k++) {
        uint8x16_t y = vqtbl4q_u8(P, M.val[k]);                    /* y[b'] = P[M[b']] = slot*4 + o of piece M[b'] */
        uint8x16_t slot = vshrq_n_u8(y, 2), o = vandq_u8(y, lo);
        uint8x16_t ms = vqtbl4q_u8(M, slot), offs = vqtbl4q_u8(OFF, ms);
        uint8x16_t t = vaddq_u8(vsubq_u8(vaddq_u8(OFF.val[k], four), o), vsubq_u8(four, offs));   /* OFF[b'] - o - OFF[M[s]] + 8 */
        uint8x16_t r4 = vorrq_u8(vshlq_n_u8(ms, 2), vandq_u8(t, lo));
        if (k == 0) {
            uint8x16_t t3 = vaddq_u8(vsubq_u8(vaddq_u8(OFF.val[0], vdupq_n_u8(3)), o), vsubq_u8(vdupq_n_u8(3), offs));  /* + 6, in 0..11 */
            uint8x16_t r3 = vorrq_u8(vshlq_n_u8(ms, 2), vqtbl1q_u8(mod3, t3));
            r4 = vbslq_u8(cmask, r3, r4);
        }
        V.val[k] = r4;
    }
    /* padding lanes: M = 0xFF gives y = 0 -> slot 0 -> garbage; clear them */
    V.val[3] = vandq_u8(V.val[3], vcombine_u8(vcreate_u8(0x000000000000FFFFULL), vdup_n_u8(0)));
    return V;
}
/* mirror-canonical label: the smaller of the labels of y and mirror(y) */
static inline uint64_t canon50m(uint8x16x4_t y, const fs_t *F, const mir_t *R) {
    uint64_t a, b; canon50x2(y, mirror50(y, R), F, &a, &b); return a < b ? a : b;
}
void labels50m(const uint8_t *maps, int64_t n, const fs_t *F, const mir_t *R, uint64_t *out) {
    for (int64_t i = 0; i < n; i++) out[i] = canon50m(vld1q_u8_x4(maps + i * 64), F, R);
}

/* ===================================== hash table of (label, cost), 8 bytes per slot ===================================== */
static inline uint64_t ht_slot(uint64_t key, uint64_t mask) { uint64_t hk = key & ~31ULL; if (!hk) hk = 32; return fin(hk) & mask; }
static inline int ht_insert_at(_Atomic uint64_t *ht, uint64_t mask, uint64_t slot, uint64_t key, uint64_t c) {
    uint64_t hk = key & ~31ULL; if (!hk) hk = 32;
    uint64_t v = hk | c;
    for (uint64_t probes = 0; probes <= mask; ) {
        uint64_t cur = atomic_load_explicit(&ht[slot], memory_order_relaxed);
        if (cur == 0) { if (atomic_compare_exchange_strong(&ht[slot], &cur, v)) return 1; continue; }
        if ((cur & ~31ULL) == hk) { while ((cur & 31ULL) > c) if (atomic_compare_exchange_weak(&ht[slot], &cur, v)) return 1; return 1; }
        slot = (slot + 1) & mask; probes++;
    }
    return 0;
}
static inline void ins_w(_Atomic uint64_t *ht, _Atomic uint64_t *ew, uint64_t mask, uint64_t k, uint64_t c, uint64_t w);
static inline void ins_w2(_Atomic uint64_t *ht, _Atomic uint64_t *ew, _Atomic uint64_t *ew2, uint64_t mask, uint64_t k, uint64_t c, uint64_t w, uint64_t w2);
#define LMASK (~1023ULL)
#define PF 16
typedef struct { uint64_t k, s; int c; } pend_t;
typedef struct { pend_t buf[PF]; int pos, filled; int64_t lost; } pipe_t;
/* insert with the slot prefetched PF inserts ahead */
static inline void pipe_put(pipe_t *Pp, _Atomic uint64_t *ht, uint64_t mask, uint64_t k, int c) {
    uint64_t s = ht_slot(k, mask); __builtin_prefetch((const void *)&ht[s], 1, 0);
    pend_t *e = &Pp->buf[Pp->pos];
    if (Pp->filled == PF) { if (!ht_insert_at(ht, mask, e->s, e->k, (uint64_t)e->c)) Pp->lost++; } else Pp->filled++;
    e->k = k; e->s = s; e->c = c; Pp->pos = (Pp->pos + 1) % PF;
}
static inline void pipe_flush(pipe_t *Pp, _Atomic uint64_t *ht, uint64_t mask) {
    for (int i = 0; i < Pp->filled; i++) if (!ht_insert_at(ht, mask, Pp->buf[i].s, Pp->buf[i].k, (uint64_t)Pp->buf[i].c)) Pp->lost++;
    Pp->filled = 0;
}

/* ======================= pair join over a ball stored in group order (each group contiguous) ===========================
 * Every ordered pair (p, q) of distinct members of a group with cp - cq in -1..2 and cp + cq <= maxcost gives
 * g = p . inverse(q), inserted under its label with cost cp + cq.  Groups are processed in tiles of GMAX per side.
 * join50: mode 0 full, 1 no insert, 2 compose only, 3 distinct elements (elems50). */
#define GMAX 4096
typedef struct {
    const int8_t *st; const int8_t *cost; const int64_t *gstart; int64_t ng; const spec_t *sp; const fs_t *F; int maxcost, mode;
    _Atomic uint64_t *ht; uint64_t mask; _Atomic int64_t next, next2, pairs, lost; _Atomic uint64_t sink;
    _Atomic uint64_t *ew; int64_t base; const int64_t *nrep; const mir_t *mir; int rb; _Atomic uint64_t *ew2; const uint64_t *rowid;
} join_t;
static const int64_t *g_items; static int64_t g_nitems;
typedef struct {
    uint8_t (*P)[64], (*Qi)[64]; uint8x16x4_t Yp[2]; int Lp[2], np2; uint64_t Wp[2], Wq[2]; pipe_t PP; int64_t pairs; uint64_t sink;
} jw_t;
/* one tile pair (q tile at qt, p tile at pt) of group g */
static void join50_tile(join_t *J, jw_t *W, int64_t g, int64_t qt, int64_t pt) {
    int64_t a0 = J->gstart[g], n = J->gstart[g + 1] - a0;
    uint8_t Q[64] __attribute__((aligned(16)));
    int rb = J->rb; const int8_t *S0 = J->st + a0 * rb; const int8_t *C0 = J->cost + a0;
    int64_t nr = J->nrep ? J->nrep[g] : n;              /* p side: the first nr rows (one per target class and cost) */
    int64_t nq = n - qt < GMAX ? n - qt : GMAX, np_ = nr - pt < GMAX ? nr - pt : GMAX;
    if (np_ <= 0) return;
    const int8_t *SQ = S0 + qt * rb, *SP = S0 + pt * rb, *CQ = C0 + qt, *CP = C0 + pt;
    for (int64_t i = 0; i < np_; i++) { if (rb == SB) pm50(SP + i * SB, W->P[i]); else memcpy(W->P[i], SP + i * 64, 64); }
    for (int64_t i = 0; i < nq; i++) { if (rb == SB) pm50(SQ + i * SB, Q); else memcpy(Q, SQ + i * 64, 64); pm50_inv(Q, W->Qi[i]); }
    for (int64_t iq = 0; iq < nq; iq++) {
        int cq = CQ[iq];
        for (int64_t ip = 0; ip < np_; ip++) {
            int cp = CP[ip], d = cp - cq, L = cp + cq;
            if (pt + ip == qt + iq || d < -1 || d > 2 || L > J->maxcost) continue;
            W->pairs++;
            if (J->mode == 3) {             /* distinct elements: full-map hash, witness rows (p, q) */
                uint8x16x4_t y = compose50(W->Qi[iq], W->P[ip]);
                uint64_t k = hash50(y), hk = k & ~31ULL; if (!hk) hk = 32;
                uint64_t v = hk | (uint64_t)L, slot = fin(hk) & J->mask;
                uint64_t pq = ((uint64_t)(a0 + pt + ip) << 32) | (uint64_t)(a0 + qt + iq);
                for (;;) {
                    uint64_t cur = atomic_load_explicit(&J->ht[slot], memory_order_relaxed);
                    if (cur == 0) { if (atomic_compare_exchange_strong(&J->ht[slot], &cur, v)) { atomic_store(&J->ew[slot], pq); break; } continue; }
                    if ((cur & ~31ULL) == hk) {
                        while ((cur & 31ULL) > (uint64_t)L) if (atomic_compare_exchange_weak(&J->ht[slot], &cur, v)) { atomic_store(&J->ew[slot], pq); break; }
                        break;
                    }
                    slot = (slot + 1) & J->mask;
                }
                continue;
            }
            W->Yp[W->np2] = compose50(W->Qi[iq], W->P[ip]); W->Lp[W->np2] = L;
            W->Wp[W->np2] = ((uint64_t)(a0 + pt + ip) << 32) | (uint64_t)(a0 + qt + iq);
            W->np2++;
            if (J->ew2) { W->Wq[W->np2 - 1] = J->rowid ? J->rowid[a0 + qt + iq] : (uint64_t)(a0 + qt + iq); W->Wp[W->np2 - 1] = J->rowid ? J->rowid[a0 + pt + ip] : (uint64_t)(a0 + pt + ip); }
            if (J->mode == 2) { W->sink += vgetq_lane_u8(W->Yp[0].val[0], 5); W->np2 = 0; continue; }
            if (J->mir) {                   /* mirror-canonical label: both labels of this one product */
                uint64_t k = canon50m(W->Yp[0], J->F, J->mir); W->np2 = 0;
                if (J->mode == 1) { W->sink += k; continue; }
                if (J->ew2) ins_w2(J->ht, J->ew, J->ew2, J->mask, k, (uint64_t)W->Lp[0], W->Wp[0], W->Wq[0]);
                else if (J->ew) ins_w(J->ht, J->ew, J->mask, k, (uint64_t)W->Lp[0], W->Wp[0]); else pipe_put(&W->PP, J->ht, J->mask, k, W->Lp[0]);
                continue;
            }
            if (W->np2 < 2) continue;
            uint64_t k0, k1; canon50x2(W->Yp[0], W->Yp[1], J->F, &k0, &k1); W->np2 = 0;
            if (J->mode == 1) { W->sink += k0 + k1; continue; }
            if (J->ew2) { ins_w2(J->ht, J->ew, J->ew2, J->mask, k0, (uint64_t)W->Lp[0], W->Wp[0], W->Wq[0]); ins_w2(J->ht, J->ew, J->ew2, J->mask, k1, (uint64_t)W->Lp[1], W->Wp[1], W->Wq[1]); continue; }
            if (J->ew) { ins_w(J->ht, J->ew, J->mask, k0, (uint64_t)W->Lp[0], W->Wp[0]); ins_w(J->ht, J->ew, J->mask, k1, (uint64_t)W->Lp[1], W->Wp[1]); continue; }
            pipe_put(&W->PP, J->ht, J->mask, k0, W->Lp[0]); pipe_put(&W->PP, J->ht, J->mask, k1, W->Lp[1]);
        }
    }
}
static void *join50_worker(void *a) {
    join_t *J = a; jw_t W; memset(&W, 0, sizeof W);
    W.P = aligned_alloc(16, (size_t)GMAX * 64); W.Qi = aligned_alloc(16, (size_t)GMAX * 64);
    for (;;) {                                           /* groups up to GMAX: whole groups */
        int64_t g0 = atomic_fetch_add(&J->next, 64); if (g0 >= J->ng) break;
        int64_t g1 = g0 + 64 < J->ng ? g0 + 64 : J->ng;
        for (int64_t g = g0; g < g1; g++) {
            int64_t n = J->gstart[g + 1] - J->gstart[g];
            if (n >= 2 && n <= GMAX) join50_tile(J, &W, g, 0, 0);
        }
    }
    for (;;) {                                           /* bigger groups: tile pairs, shared among the threads */
        int64_t i = atomic_fetch_add(&J->next2, 1); if (i >= g_nitems) break;
        join50_tile(J, &W, g_items[3 * i], g_items[3 * i + 1], g_items[3 * i + 2]);
    }
    if (W.np2 == 1 && J->mode == 0) {
        uint64_t k = canon50(W.Yp[0], J->F);
        if (J->ew2) ins_w2(J->ht, J->ew, J->ew2, J->mask, k, (uint64_t)W.Lp[0], W.Wp[0], W.Wq[0]);
        else if (J->ew) ins_w(J->ht, J->ew, J->mask, k, (uint64_t)W.Lp[0], W.Wp[0]); else pipe_put(&W.PP, J->ht, J->mask, k, W.Lp[0]);
    }
    pipe_flush(&W.PP, J->ht, J->mask);
    free(W.P); free(W.Qi);
    atomic_fetch_add(&J->pairs, W.pairs); atomic_fetch_add(&J->lost, W.PP.lost); atomic_fetch_add(&J->sink, W.sink);
    return NULL;
}
static uint64_t *g_ew, *g_ew2; static const uint64_t *g_rowid; static const int64_t *g_nrep; static const mir_t *g_mir; static int g_rb = SB;
static double run_join(void *(*w)(void *), const int8_t *st, const int8_t *cost, const int64_t *gstart, int64_t ng, const spec_t *sp,
                       const fs_t *F, int maxcost, int mode, uint64_t *ht, int hbits, int threads, int64_t *stats) {
    join_t J; memset(&J, 0, sizeof J);
    J.st = st; J.cost = cost; J.gstart = gstart; J.ng = ng; J.sp = sp; J.F = F; J.maxcost = maxcost; J.mode = mode;
    J.ht = (_Atomic uint64_t *)ht; J.mask = (1ULL << hbits) - 1; J.ew = (_Atomic uint64_t *)g_ew; J.ew2 = (_Atomic uint64_t *)g_ew2; J.rowid = g_rowid; J.nrep = g_nrep; J.mir = g_mir; J.rb = g_rb;
    int64_t ni = 0;
    for (int64_t g = 0; g < ng; g++) { int64_t n = gstart[g + 1] - gstart[g], k = (n + GMAX - 1) / GMAX, nr = g_nrep ? g_nrep[g] : n; if (n > GMAX) ni += k * ((nr + GMAX - 1) / GMAX); }
    int64_t *items = malloc(sizeof(int64_t) * 3 * (ni + 1)); ni = 0;
    for (int64_t g = 0; g < ng; g++) {
        int64_t n = gstart[g + 1] - gstart[g]; if (n <= GMAX) continue;
        int64_t nr = g_nrep ? g_nrep[g] : n;
        for (int64_t qt = 0; qt < n; qt += GMAX) for (int64_t pt = 0; pt < nr; pt += GMAX) { items[3 * ni] = g; items[3 * ni + 1] = qt; items[3 * ni + 2] = pt; ni++; }
    }
    g_items = items; g_nitems = ni;
    pthread_t th[64]; double t0 = now();
    for (int t = 0; t < threads; t++) pthread_create(&th[t], NULL, w, &J);
    for (int t = 0; t < threads; t++) pthread_join(th[t], NULL);
    stats[0] = J.pairs; stats[1] = J.lost; stats[2] = (int64_t)J.sink;
    free(items); g_items = NULL; g_nitems = 0; g_nrep = NULL; g_ew = NULL; g_ew2 = NULL; g_rowid = NULL; g_mir = NULL; g_rb = SB;
    return now() - t0;
}
/* distinct elements up to maxcost: ht gets (hash | cost), ew the witness (p << 32 | q) rows */
double elems50(const int8_t *st, const int8_t *cost, const int64_t *gstart, int64_t ng, int maxcost,
               uint64_t *ht, uint64_t *ew, int hbits, int threads, int64_t *stats) {
    g_ew = ew; return run_join(join50_worker, st, cost, gstart, ng, NULL, NULL, maxcost, 3, ht, hbits, threads, stats);
}

/* ============================ forward probe: x . a for stored elements a, against a sorted table =====================
 * Elements are 50-point maps (64-byte rows).  The table is sorted (label | cost) with a bucket index on the top bits.
 * Matches are counted in hist[forward cost * 32 + table cost].  Lookups are pipelined with prefetches. */
static inline int64_t bucket_find(const uint64_t *t, const uint32_t *bidx, int bbits, uint64_t k) {
    uint64_t kk = k & ~31ULL, b = kk >> (64 - bbits);
    for (uint32_t i = bidx[b]; i < bidx[b + 1]; i++) { uint64_t v = t[i] & ~31ULL; if (v == kk) return (int64_t)i; if (v > kk) break; }
    return -1;
}
static inline int64_t bucket_cost(const uint64_t *t, const uint32_t *bidx, int bbits, uint64_t k) {
    uint64_t kk = k & ~31ULL, b = kk >> (64 - bbits);
    for (uint32_t i = bidx[b]; i < bidx[b + 1]; i++) { uint64_t v = t[i] & ~31ULL; if (v == kk) return (int64_t)(t[i] & 31ULL); if (v > kk) break; }
    return -1;
}
void build_bidx(const uint64_t *t, int64_t m, int bbits, uint32_t *bidx) {
    int64_t nb = 1LL << bbits, j = 0;
    for (int64_t b = 0; b <= nb; b++) { while (j < m && (int64_t)((t[j] & ~31ULL) >> (64 - bbits)) < b) j++; bidx[b] = (uint32_t)j; }
}
/* 50-point maps (64-byte rows) of p . inverse(q) for witness pairs into a state array */
void pair_maps50(const int8_t *st, const uint64_t *pq, int64_t n, uint8_t *out) {
    uint8_t P[64] __attribute__((aligned(16))), Q[64] __attribute__((aligned(16))), Qi[64] __attribute__((aligned(16)));
    for (int64_t i = 0; i < n; i++) {
        pm50(st + (int64_t)(pq[i] >> 32) * SB, P); pm50(st + (int64_t)(pq[i] & 0xffffffffULL) * SB, Q); pm50_inv(Q, Qi);
        vst1q_u8_x4(out + i * 64, compose50(Qi, P));
    }
}

/* ===================== chaining: classes of g . a for class representatives g and pieces a ========================
 * reps: 50-point maps (64-byte rows) of representatives with costs rc; pieces: maps with costs pc.  Every (i, j) with
 * rc[i] + pc[j] <= maxcost is labelled; ht keeps (label | cost) with the minimum cost and ew the witness (i << 32 | j). */
typedef struct {
    const uint8_t *reps; const int8_t *rc; int64_t n; const uint8_t *pcs; const int8_t *pc; int64_t m; int maxcost; const fs_t *F;
    _Atomic uint64_t *ht; _Atomic uint64_t *ew; uint64_t mask; _Atomic int64_t next, done; int64_t base;
} chain_t;
static inline void ins_w2(_Atomic uint64_t *ht, _Atomic uint64_t *ew, _Atomic uint64_t *ew2, uint64_t mask, uint64_t k, uint64_t c, uint64_t w, uint64_t w2) {
    uint64_t hk = k & ~31ULL; if (!hk) hk = 32;
    uint64_t v = hk | c, slot = fin(hk) & mask;
    for (;;) {
        uint64_t cur = atomic_load_explicit(&ht[slot], memory_order_relaxed);
        if (cur == 0) { if (atomic_compare_exchange_strong(&ht[slot], &cur, v)) { atomic_store(&ew[slot], w); if (ew2) atomic_store(&ew2[slot], w2); return; } continue; }
        if ((cur & ~31ULL) == hk) {
            while ((cur & 31ULL) > c) if (atomic_compare_exchange_weak(&ht[slot], &cur, v)) { atomic_store(&ew[slot], w); if (ew2) atomic_store(&ew2[slot], w2); return; }
            return;
        }
        slot = (slot + 1) & mask;
    }
}
static inline void ins_w(_Atomic uint64_t *ht, _Atomic uint64_t *ew, uint64_t mask, uint64_t k, uint64_t c, uint64_t w) {
    uint64_t hk = k & ~31ULL; if (!hk) hk = 32;
    uint64_t v = hk | c, slot = fin(hk) & mask;
    for (;;) {
        uint64_t cur = atomic_load_explicit(&ht[slot], memory_order_relaxed);
        if (cur == 0) { if (atomic_compare_exchange_strong(&ht[slot], &cur, v)) { atomic_store(&ew[slot], w); return; } continue; }
        if ((cur & ~31ULL) == hk) {
            while ((cur & 31ULL) > c) if (atomic_compare_exchange_weak(&ht[slot], &cur, v)) { atomic_store(&ew[slot], w); return; }
            return;
        }
        slot = (slot + 1) & mask;
    }
}
static void *chain_worker(void *a) {
    chain_t *Ch = a; int64_t done = 0;
    for (;;) {
        int64_t i0 = atomic_fetch_add(&Ch->next, 16); if (i0 >= Ch->n) break;
        int64_t i1 = i0 + 16 < Ch->n ? i0 + 16 : Ch->n;
        for (int64_t i = i0; i < i1; i++) {
            const uint8_t *X = Ch->reps + i * 64; int c = Ch->rc[i];
            int64_t j = 0; uint8x16x4_t y[2]; int64_t jj[2]; int k = 0;
            for (; j < Ch->m; j++) {
                if (c + Ch->pc[j] > Ch->maxcost) continue;
                y[k] = compose50(Ch->pcs + j * 64, X); jj[k] = j; k++;
                if (k == 2) {
                    uint64_t k0, k1; canon50x2(y[0], y[1], Ch->F, &k0, &k1); done += 2;
                    ins_w(Ch->ht, Ch->ew, Ch->mask, k0, (uint64_t)(c + Ch->pc[jj[0]]), ((uint64_t)(Ch->base + i) << 32) | (uint64_t)jj[0]);
                    ins_w(Ch->ht, Ch->ew, Ch->mask, k1, (uint64_t)(c + Ch->pc[jj[1]]), ((uint64_t)(Ch->base + i) << 32) | (uint64_t)jj[1]);
                    k = 0;
                }
            }
            if (k == 1) { uint64_t k0 = canon50(y[0], Ch->F); done++; ins_w(Ch->ht, Ch->ew, Ch->mask, k0, (uint64_t)(c + Ch->pc[jj[0]]), ((uint64_t)(Ch->base + i) << 32) | (uint64_t)jj[0]); }
        }
    }
    atomic_fetch_add(&Ch->done, done);
    return NULL;
}
int64_t chain_step(const uint8_t *reps, const int8_t *rc, int64_t n, int64_t base, const uint8_t *pcs, const int8_t *pc, int64_t m, int maxcost,
                   const fs_t *F, uint64_t *ht, uint64_t *ew, int hbits, int threads) {
    chain_t Ch; memset(&Ch, 0, sizeof Ch);
    Ch.reps = reps; Ch.rc = rc; Ch.n = n; Ch.base = base; Ch.pcs = pcs; Ch.pc = pc; Ch.m = m; Ch.maxcost = maxcost; Ch.F = F;
    Ch.ht = (_Atomic uint64_t *)ht; Ch.ew = (_Atomic uint64_t *)ew; Ch.mask = (1ULL << hbits) - 1;
    pthread_t th[64];
    for (int t = 0; t < threads; t++) pthread_create(&th[t], NULL, chain_worker, &Ch);
    for (int t = 0; t < threads; t++) pthread_join(th[t], NULL);
    return Ch.done;
}
/* maps of apply1(rep i, piece j) for witnesses (i << 32 | j) */
void chain_maps(const uint8_t *reps, const uint8_t *pcs, const uint64_t *w, int64_t n, uint8_t *out) {
    for (int64_t t = 0; t < n; t++) vst1q_u8_x4(out + t * 64, compose50(pcs + (w[t] & 0xffffffffULL) * 64, reps + (w[t] >> 32) * 64));
}
/* 50-point maps of states (64-byte rows) */
void maps50(const int8_t *s, int64_t n, uint8_t *out) { for (int64_t i = 0; i < n; i++) { pm50(s + i * SB, out + i * 64); } }
/* labels of apply1(x, a) for elements given as 50-point maps (64-byte rows), parallel */
typedef struct { const uint8_t *X; const uint8_t *el; int64_t n; const fs_t *F; uint64_t *out; _Atomic int64_t next; } plm_t;
static void *plm_worker(void *a) {
    plm_t *L = a;
    for (;;) {
        int64_t i0 = atomic_fetch_add(&L->next, 8192); if (i0 >= L->n) break;
        int64_t i1 = i0 + 8192 < L->n ? i0 + 8192 : L->n, i = i0;
        for (; i + 1 < i1; i += 2) canon50x2(compose50(L->el + i * 64, L->X), compose50(L->el + (i + 1) * 64, L->X), L->F, &L->out[i], &L->out[i + 1]);
        if (i < i1) L->out[i] = canon50(compose50(L->el + i * 64, L->X), L->F);
    }
    return NULL;
}
void probe_labels_maps(const int8_t *x, const uint8_t *el, int64_t n, const fs_t *F, uint64_t *out, int threads) {
    uint8_t X[64] __attribute__((aligned(16))); pm50(x, X);
    plm_t L = {X, el, n, F, out, 0}; pthread_t th[64];
    for (int t = 0; t < threads; t++) pthread_create(&th[t], NULL, plm_worker, &L);
    for (int t = 0; t < threads; t++) pthread_join(th[t], NULL);
}

/* join50 with the p side limited to the first nrep[g] rows of each group (rows ordered so those come first) */

/* class join with representatives: ht gets (label | cost), ew the witness rows (p << 32 | q) of the ordered ball */
double join50w(const int8_t *st, const int8_t *cost, const int64_t *gstart, const int64_t *nrep, int64_t ng, const fs_t *F, const mir_t *mir, int maxcost,
               uint64_t *ht, uint64_t *ew, int hbits, int threads, int64_t *stats) {
    g_nrep = nrep; g_ew = ew; g_mir = mir; return run_join(join50_worker, st, cost, gstart, ng, NULL, F, maxcost, 0, ht, hbits, threads, stats);
}

/* seed a (label | cost) table with witnesses */
void seed(uint64_t *ht, uint64_t *ew, int hbits, const uint64_t *labels, const int8_t *costs, const uint64_t *w, int64_t n) {
    uint64_t mask = (1ULL << hbits) - 1;
    for (int64_t i = 0; i < n; i++) ins_w((_Atomic uint64_t *)ht, (_Atomic uint64_t *)ew, mask, labels[i], (uint64_t)costs[i], w[i]);
}


/* class join over 64-byte map rows (ordered, with representatives first in each group), optional mirror and witnesses */
double join50maps(const uint8_t *maps, const int8_t *cost, const int64_t *gstart, const int64_t *nrep, int64_t ng, const fs_t *F, const mir_t *mir,
                  int maxcost, uint64_t *ht, uint64_t *ew, uint64_t *ew2, const uint64_t *rowid, int hbits, int threads, int64_t *stats) {
    g_nrep = nrep; g_ew = ew; g_ew2 = ew2; g_rowid = rowid; g_mir = mir; g_rb = 64;
    return run_join(join50_worker, (const int8_t *)maps, cost, gstart, ng, NULL, F, maxcost, 0, ht, hbits, threads, stats);
}

/* ===================== generating the next ball level, one label slice at a time =====================
 * For every parent map (level h-1) and every run r (21 maps, cost cr), the child map c = parent . r; its group label
 * (under F, the starting group's spec) decides its slice: fin(label) % nslices == k.  Children in slice k are written
 * out (map, label, full-map hash, parent index, run index); duplicates and states of lower levels are removed by the
 * caller.  Returns the count (stops at cap). */
typedef struct {
    const uint8_t *par; int64_t n; const uint8_t *runs; int nr; const fs_t *F; int k, ns;
    uint8_t *om; uint64_t *ol, *oh; int32_t *op; int8_t *orun; int64_t cap; _Atomic int64_t next, cnt;
} gen_t;
static void *gen_worker(void *a) {
    gen_t *G = a; uint8x16x4_t Y[2]; int64_t pi[2]; int ri[2]; uint64_t H[2]; int np2 = 0;
    for (;;) {
        int64_t i0 = atomic_fetch_add(&G->next, 1024); if (i0 >= G->n) break;
        int64_t i1 = i0 + 1024 < G->n ? i0 + 1024 : G->n;
        for (int64_t i = i0; i < i1; i++) for (int r = 0; r < G->nr; r++) {
            Y[np2] = compose50(G->runs + r * 64, G->par + i * 64); pi[np2] = i; ri[np2] = r; np2++;
            if (np2 < 2) continue;
            uint64_t l0, l1; canon50x2(Y[0], Y[1], G->F, &l0, &l1); np2 = 0;
            uint64_t L[2] = {l0, l1};
            for (int e = 0; e < 2; e++) {
                if ((int)(fin(L[e]) % (uint64_t)G->ns) != G->k) continue;
                int64_t j = atomic_fetch_add(&G->cnt, 1); if (j >= G->cap) continue;
                vst1q_u8_x4(G->om + j * 64, Y[e]); G->ol[j] = L[e]; G->oh[j] = hash50(Y[e]); G->op[j] = (int32_t)pi[e]; G->orun[j] = (int8_t)ri[e];
            }
        }
    }
    if (np2 == 1) {
        uint64_t l = canon50(Y[0], G->F);
        if ((int)(fin(l) % (uint64_t)G->ns) == G->k) { int64_t j = atomic_fetch_add(&G->cnt, 1); if (j < G->cap) { vst1q_u8_x4(G->om + j * 64, Y[0]); G->ol[j] = l; G->oh[j] = hash50(Y[0]); G->op[j] = (int32_t)pi[0]; G->orun[j] = (int8_t)ri[0]; } }
    }
    return NULL;
}
int64_t gen_slice(const uint8_t *par, int64_t n, const uint8_t *runs, int nr, const fs_t *F, int k, int ns,
                  uint8_t *om, uint64_t *ol, uint64_t *oh, int32_t *op, int8_t *orun, int64_t cap, int threads) {
    gen_t G; memset(&G, 0, sizeof G);
    G.par = par; G.n = n; G.runs = runs; G.nr = nr; G.F = F; G.k = k; G.ns = ns; G.om = om; G.ol = ol; G.oh = oh; G.op = op; G.orun = orun; G.cap = cap;
    pthread_t th[64];
    for (int t = 0; t < threads; t++) pthread_create(&th[t], NULL, gen_worker, &G);
    for (int t = 0; t < threads; t++) pthread_join(th[t], NULL);
    return G.cnt < G.cap ? G.cnt : G.cap;
}
/* full-map hashes and group labels of map rows */
void maps_hash(const uint8_t *maps, int64_t n, uint64_t *out) { for (int64_t i = 0; i < n; i++) out[i] = hash50(vld1q_u8_x4(maps + i * 64)); }
void fin_mod(const uint64_t *l, int64_t n, int ns, int64_t *out) { for (int64_t i = 0; i < n; i++) out[i] = (int64_t)(fin(l[i]) % (uint64_t)ns); }

/* probe with hit collection: hits[4 * j] = element index, table index, forward cost, table cost; returns the hit count */
/* probe many inputs (e.g. the relabelled copies of one state) against one forward list and table; threads over inputs.
 * hits[5 * j] = input index, element index, table index, forward cost, table cost; returns the hit count */
typedef struct {
    const uint8_t *el; const int8_t *ec; int64_t n; int emax; const uint8_t *xs; int64_t nx; const uint64_t *tab; const uint32_t *bidx; int bbits;
    const fs_t *F; const mir_t *mir; _Atomic int64_t next; int64_t *hits; _Atomic int64_t nh; int64_t hcap; int bound; int window; _Atomic int *best;
} pmulti_t;
#define PM_BLOCK 65536                     /* elements per task: tasks are (input, element block), so few inputs still use every thread */
static void *pmulti_worker(void *a) {
    pmulti_t *P = a; int64_t nb = (P->n + PM_BLOCK - 1) / PM_BLOCK;
    for (;;) {
        int64_t t = atomic_fetch_add(&P->next, 1); if (t >= P->nx * nb) break;
        int64_t xi = t / nb, i0 = (t % nb) * PM_BLOCK, i1 = i0 + PM_BLOCK < P->n ? i0 + PM_BLOCK : P->n;
        const uint8_t *X = P->xs + xi * 64;
        for (int64_t i = i0; i < i1; i += 2) {
            int two = i + 1 < i1;
            uint8x16x4_t ya = compose50(P->el + i * 64, X), yb = two ? compose50(P->el + (i + 1) * 64, X) : ya;
            uint64_t ka, kb;
            if (P->mir) { ka = canon50m(ya, P->F, P->mir); kb = two ? canon50m(yb, P->F, P->mir) : ka; } else canon50x2(ya, yb, P->F, &ka, &kb);
            for (int e = 0; e < 1 + two; e++) {
                uint64_t k = e ? kb : ka; int ec = P->ec[i + e]; if (ec > P->emax) continue;
                int64_t ti = bucket_find(P->tab, P->bidx, P->bbits, k);
                if (ti < 0) continue;
                int c = (int)(P->tab[ti] & 31ULL), tot = ec + c;
                if (tot > P->bound) continue;
                int b = atomic_load_explicit(&P->best[xi], memory_order_relaxed);       /* per-input running best: record only hits within window of it */
                while (tot < b && !atomic_compare_exchange_weak(&P->best[xi], &b, tot)) ;
                if (tot > (b < tot ? b : tot) + P->window) continue;
                int64_t j = atomic_fetch_add(&P->nh, 1); if (j < P->hcap) { int64_t *h = P->hits + 5 * j; h[0] = xi; h[1] = i + e; h[2] = ti; h[3] = ec; h[4] = c; }
            }
        }
    }
    return NULL;
}
int64_t probe50_multi(const uint8_t *el, const int8_t *ec, int64_t n, int emax, const int8_t *xs, int64_t nx, const uint64_t *tab, const uint32_t *bidx, int bbits,
                      const fs_t *F, const mir_t *mir, int threads, int bound, int window, int64_t *hits, int64_t hcap) {
    uint8_t *X = aligned_alloc(64, (size_t)nx * 64); for (int64_t i = 0; i < nx; i++) pm50(xs + i * 100, X + i * 64);
    _Atomic int *best = malloc(sizeof(_Atomic int) * nx); for (int64_t i = 0; i < nx; i++) atomic_store(&best[i], bound);
    pmulti_t P; memset(&P, 0, sizeof P);
    P.el = el; P.ec = ec; P.n = n; P.emax = emax; P.xs = X; P.nx = nx; P.tab = tab; P.bidx = bidx; P.bbits = bbits; P.F = F; P.mir = mir; P.hits = hits; P.hcap = hcap; P.bound = bound; P.window = window; P.best = best;
    pthread_t th[64];
    for (int t = 0; t < threads; t++) pthread_create(&th[t], NULL, pmulti_worker, &P);
    for (int t = 0; t < threads; t++) pthread_join(th[t], NULL);
    free(X); free(best); return P.nh < P.hcap ? P.nh : P.hcap;
}
/* debug: per-level choice and map of the 50-point canonical form */

/* =================== two generated levels (for the 21-HTM table): level h as (parent, run) lists ===================
 * gen_children: for parent maps (regenerated by the caller) and runs, label each child under F and append
 * (parent index, run) to the bin of its slice (fin(label) % ns); bins are caller-provided arrays per slice with
 * counts; cap per bin.  Used for level h+1 from level h. */
typedef struct {
    const uint8_t *par; int64_t n; int64_t pbase; const uint8_t *runs; int nr; const fs_t *F; int ns;
    int64_t **bin_par; int8_t **bin_run; _Atomic int64_t *bin_cnt; int64_t cap; _Atomic int64_t next; _Atomic int64_t lost;
} genb_t;
static void *genb_worker(void *a) {
    genb_t *G = a; uint8x16x4_t Y[2]; int64_t pi[2]; int ri[2]; int np2 = 0; int64_t lost = 0;
    for (;;) {
        int64_t i0 = atomic_fetch_add(&G->next, 1024); if (i0 >= G->n) break;
        int64_t i1 = i0 + 1024 < G->n ? i0 + 1024 : G->n;
        for (int64_t i = i0; i < i1; i++) for (int r = 0; r < G->nr; r++) {
            Y[np2] = compose50(G->runs + r * 64, G->par + i * 64); pi[np2] = i; ri[np2] = r; np2++;
            if (np2 < 2) continue;
            uint64_t l0, l1; canon50x2(Y[0], Y[1], G->F, &l0, &l1); np2 = 0; uint64_t L[2] = {l0, l1};
            for (int e = 0; e < 2; e++) {
                int k = (int)(fin(L[e]) % (uint64_t)G->ns); int64_t j = atomic_fetch_add(&G->bin_cnt[k], 1);
                if (j >= G->cap) { lost++; continue; }
                G->bin_par[k][j] = (G->pbase + pi[e]); G->bin_run[k][j] = (int8_t)ri[e];
            }
        }
    }
    if (np2 == 1) { uint64_t l = canon50(Y[0], G->F); int k = (int)(fin(l) % (uint64_t)G->ns); int64_t j = atomic_fetch_add(&G->bin_cnt[k], 1); if (j < G->cap) { G->bin_par[k][j] = (G->pbase + pi[0]); G->bin_run[k][j] = (int8_t)ri[0]; } else lost++; }
    atomic_fetch_add(&G->lost, lost); return NULL;
}
int64_t gen_bins(const uint8_t *par, int64_t n, int64_t pbase, const uint8_t *runs, int nr, const fs_t *F, int ns,
                 int64_t **bin_par, int8_t **bin_run, int64_t *bin_cnt, int64_t cap, int threads) {
    genb_t G; memset(&G, 0, sizeof G);
    G.par = par; G.n = n; G.pbase = pbase; G.runs = runs; G.nr = nr; G.F = F; G.ns = ns; G.bin_par = bin_par; G.bin_run = bin_run; G.bin_cnt = (_Atomic int64_t *)bin_cnt; G.cap = cap;
    pthread_t th[64];
    for (int t = 0; t < threads; t++) pthread_create(&th[t], NULL, genb_worker, &G);
    for (int t = 0; t < threads; t++) pthread_join(th[t], NULL);
    return G.lost;
}
/* maps of children: out[i] = compose(run[i], parent map[i]) where the parent map is looked up by index into `par` */
void child_maps(const uint8_t *par, const int32_t *pidx, const int8_t *run, const uint8_t *runs, int64_t n, uint8_t *out) {
    for (int64_t i = 0; i < n; i++) vst1q_u8_x4(out + i * 64, compose50(runs + run[i] * 64, par + (int64_t)pidx[i] * 64));
}
/* bitmap membership: set / test bits (hash >> (64 - bits)) */
void bitmap_set(uint8_t *bm, int bits, const uint64_t *h, int64_t n) { for (int64_t i = 0; i < n; i++) { uint64_t b = h[i] >> (64 - bits); bm[b >> 3] |= (uint8_t)(1u << (b & 7)); } }
void bitmap_test(const uint8_t *bm, int bits, const uint64_t *h, int64_t n, uint8_t *out) { for (int64_t i = 0; i < n; i++) { uint64_t b = h[i] >> (64 - bits); out[i] = (bm[b >> 3] >> (b & 7)) & 1; } }
/* grandchildren maps: compose(run2, compose(run1, ball map[parent])) */
void child_maps2(const uint8_t *ball, const int32_t *p9, const int8_t *r1, const int8_t *r2, const uint8_t *runs, int64_t n, uint8_t *out) {
    uint8_t T[64] __attribute__((aligned(16)));
    for (int64_t i = 0; i < n; i++) { vst1q_u8_x4(T, compose50(runs + r1[i] * 64, ball + (int64_t)p9[i] * 64)); vst1q_u8_x4(out + i * 64, compose50(runs + r2[i] * 64, T)); }
}
