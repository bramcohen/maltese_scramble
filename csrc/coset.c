/* Coset-space kernels for meet-in-the-middle over word lists.
 *
 * A state is 100 signed bytes: cp[8] co[8] rp[18] ro[18] gp[24] go[24] (same as model.State).
 * An element has the same layout (source permutations + orientation deltas), as in model.Move.
 *
 * A "spec" describes a coset label: which slot positions count (by piece class), which sibling
 * relations count (slot of the sibling of the piece on a slot), which Z_m-linear spin functionals
 * count (by slot and piece type), and an optional list of right multipliers (the label is the
 * minimum over them, giving a canonical label under a small subgroup acting on the right).
 */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

#define NC 8
#define NR 18
#define NG 24
#define SB 100
enum { OCP = 0, OCO = 8, ORP = 16, ORO = 34, OGP = 52, OGO = 76 };

typedef struct {
    const uint8_t *msk_c, *msk_r, *msk_g;      /* per slot: include class-of-piece feature */
    const int8_t *cls_c, *cls_r, *cls_g;       /* per piece: class value */
    const uint8_t *smsk_c, *smsk_r, *smsk_g;   /* per slot: include slot-of-sibling feature */
    const int8_t *sib_c, *sib_r, *sib_g;       /* per piece: sibling piece or -1 */
    const int8_t *typ_c, *typ_r, *typ_g;       /* per piece: type index for functionals */
    int nt_c, nt_r, nt_g;
    const int8_t *F_c, *F_r, *F_g;             /* nf x (nslot*nt) coefficient rows */
    int nf_c, nf_r, nf_g;
    const int8_t *pot_slot_c, *pot_slot_r, *pot_slot_g;   /* optional potential: spin -= pot_slot[s] + pot_piece[piece] */
    const int8_t *pot_piece_c, *pot_piece_r, *pot_piece_g;
    const int8_t *rm;                          /* nrm right multipliers (elements): label = min over state.rm */
    int nrm;
    const int8_t *lm;                          /* nlm left multipliers: label = min over lm.state */
    int nlm;
    /* canonical form of the left coset T.x via a stabiliser chain of T on the 192 (piece, orientation) points */
    int ncan;                                  /* number of nontrivial levels (0 = unused) */
    const int32_t *can_m, *can_off;            /* orbit size and offset per level */
    const int16_t *can_pts;                    /* orbit points, concatenated */
    const uint8_t *can_perm;                   /* transversal permutations (192 bytes each), concatenated */
} spec_t;
#define NPTS 192

static inline void apply1(const int8_t *s, const int8_t *e, int8_t *o) {
    for (int i = 0; i < NC; i++) { int j = e[OCP + i]; o[OCP + i] = s[OCP + j]; o[OCO + i] = (int8_t)((s[OCO + j] + e[OCO + i]) % 3); }
    for (int i = 0; i < NR; i++) { int j = e[ORP + i]; o[ORP + i] = s[ORP + j]; o[ORO + i] = (int8_t)((s[ORO + j] + e[ORO + i]) & 3); }
    for (int i = 0; i < NG; i++) { int j = e[OGP + i]; o[OGP + i] = s[OGP + j]; o[OGO + i] = (int8_t)((s[OGO + j] + e[OGO + i]) & 3); }
}

static inline uint64_t mix(uint64_t h, uint64_t v) {
    h ^= v + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
    h *= 0x100000001B3ULL;
    return h;
}
static inline uint64_t fin(uint64_t z) {
    z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ULL;
    z ^= z >> 27; z *= 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return z;
}

static inline uint64_t feat_part(uint64_t h, const int8_t *p, const int8_t *o0, int n, int m,
                                 const uint8_t *msk, const int8_t *cls, const uint8_t *smsk, const int8_t *sib,
                                 const int8_t *typ, int nt, const int8_t *F, int nf,
                                 const int8_t *pot_slot, const int8_t *pot_piece) {
    int loc[NG]; int8_t ob[NG]; const int8_t *o = o0;
    if (pot_slot) {
        for (int s = 0; s < n; s++) { int v = (o0[s] - pot_slot[s] - pot_piece[p[s]]) % m; if (v < 0) v += m; ob[s] = (int8_t)v; }
        o = ob;
    }
    if (msk) for (int s = 0; s < n; s++) if (msk[s]) h = mix(h, (uint64_t)(cls[p[s]] + 1));
    if (smsk) {
        for (int s = 0; s < n; s++) loc[p[s]] = s;
        for (int s = 0; s < n; s++) if (smsk[s]) { int sb = sib[p[s]]; h = mix(h, (uint64_t)(sb < 0 ? 0 : loc[sb] + 1)); }
    }
    if (nf) {
        int t[NG];
        for (int s = 0; s < n; s++) t[s] = s * nt + typ[p[s]];
        for (int k = 0; k < nf; k++) {
            const int8_t *row = F + (size_t)k * n * nt;
            int acc = 0;
            for (int s = 0; s < n; s++) acc += row[t[s]] * o[s];
            acc %= m; if (acc < 0) acc += m;
            h = mix(h, (uint64_t)acc);
        }
    }
    return h;
}

static inline uint64_t feat(const int8_t *s, const spec_t *sp) {
    uint64_t h = 1469598103934665603ULL;
    h = feat_part(h, s + OCP, s + OCO, NC, 3, sp->msk_c, sp->cls_c, sp->smsk_c, sp->sib_c, sp->typ_c, sp->nt_c, sp->F_c, sp->nf_c, sp->pot_slot_c, sp->pot_piece_c);
    h = feat_part(h, s + ORP, s + ORO, NR, 4, sp->msk_r, sp->cls_r, sp->smsk_r, sp->sib_r, sp->typ_r, sp->nt_r, sp->F_r, sp->nf_r, sp->pot_slot_r, sp->pot_piece_r);
    h = feat_part(h, s + OGP, s + OGO, NG, 4, sp->msk_g, sp->cls_g, sp->smsk_g, sp->sib_g, sp->typ_g, sp->nt_g, sp->F_g, sp->nf_g, sp->pot_slot_g, sp->pot_piece_g);
    return fin(h);
}

/* point map of a state: (piece i, orient o) -> (slot j, o + d_j) */
static inline void pointmap(const int8_t *s, uint8_t *X) {
    for (int j = 0; j < NC; j++) { int i = s[OCP + j], d = s[OCO + j]; for (int o = 0; o < 3; o++) X[i * 3 + o] = (uint8_t)(j * 3 + (o + d) % 3); }
    for (int j = 0; j < NR; j++) { int i = s[ORP + j], d = s[ORO + j]; for (int o = 0; o < 4; o++) X[24 + i * 4 + o] = (uint8_t)(24 + j * 4 + ((o + d) & 3)); }
    for (int j = 0; j < NG; j++) { int i = s[OGP + j], d = s[OGO + j]; for (int o = 0; o < 4; o++) X[96 + i * 4 + o] = (uint8_t)(96 + j * 4 + ((o + d) & 3)); }
}

static inline uint64_t canon(const int8_t *s, const spec_t *sp) {
    uint8_t X[NPTS], U[NPTS], V[NPTS];
    pointmap(s, X);
    for (int p = 0; p < NPTS; p++) U[p] = (uint8_t)p;
    for (int l = 0; l < sp->ncan; l++) {
        int m = sp->can_m[l], off = sp->can_off[l], best = -1, bv = 1000;
        for (int t = 0; t < m; t++) { int j = sp->can_pts[off + t]; int v = X[U[j]]; if (v < bv) { bv = v; best = t; } }
        const uint8_t *u = sp->can_perm + (size_t)(off + best) * NPTS;
        for (int p = 0; p < NPTS; p++) V[p] = U[u[p]];
        memcpy(U, V, NPTS);
    }
    uint64_t h = 1469598103934665603ULL;
    for (int p = 0; p < NPTS; p++) h = mix(h, (uint64_t)X[U[p]]);
    return fin(h);
}

static inline uint64_t key1(const int8_t *s, const spec_t *sp) {
    if (sp->ncan) return canon(s, sp);
    if (!sp->nrm && !sp->nlm) return feat(s, sp);
    uint64_t best = UINT64_MAX; int8_t t[SB];
    for (int m = 0; m < sp->nrm; m++) {
        apply1(s, sp->rm + (size_t)m * SB, t);
        uint64_t h = feat(t, sp);
        if (h < best) best = h;
    }
    for (int m = 0; m < sp->nlm; m++) {
        apply1(sp->lm + (size_t)m * SB, s, t);     /* lm . s : the state s applied to the element lm */
        uint64_t h = feat(t, sp);
        if (h < best) best = h;
    }
    return best;
}

/* ---- threading ---- */
typedef struct {
    const int8_t *states; int64_t lo, hi;
    const int8_t *elem; const spec_t *sp;
    uint64_t *keys; int8_t *out;
    const uint64_t *table; int64_t m;
    int64_t *hit_i, *hit_pos; int64_t cap; int64_t *nhit; pthread_mutex_t *mu;
    int mode;
    uint64_t cmask;                            /* compact tables: low bits of each entry hold the cost; compare above them */
    const int16_t *costs; int64_t maxcost;     /* sweep: record a match only if its table cost <= maxcost */
} job_t;

static inline int64_t bsearch64(const uint64_t *t, int64_t m, uint64_t k) {
    int64_t lo = 0, hi = m;
    while (lo < hi) { int64_t mid = (lo + hi) >> 1; if (t[mid] < k) lo = mid + 1; else hi = mid; }
    return (lo < m && t[lo] == k) ? lo : -1;
}

/* compact table: entries are (label & ~cmask) | cost, sorted; find the entry whose label bits equal k's */
static inline int64_t bsearch64m(const uint64_t *t, int64_t m, uint64_t k, uint64_t cmask) {
    uint64_t kk = k & ~cmask; int64_t lo = 0, hi = m;
    while (lo < hi) { int64_t mid = (lo + hi) >> 1; if (t[mid] < kk) lo = mid + 1; else hi = mid; }
    return (lo < m && (t[lo] & ~cmask) == kk) ? lo : -1;
}

static void *worker(void *arg) {
    job_t *j = (job_t *)arg;
    int8_t tmp[SB];
    for (int64_t i = j->lo; i < j->hi; i++) {
        const int8_t *s = j->states + i * SB;
        if (j->elem) { apply1(s, j->elem, tmp); s = tmp; }
        if (j->mode == 0) { memcpy(j->out + i * SB, s, SB); if (j->keys) j->keys[i] = key1(s, j->sp); }
        else if (j->mode == 1) j->keys[i] = key1(s, j->sp);
        else {
            uint64_t k = key1(s, j->sp);
            int64_t pos = j->cmask ? bsearch64m(j->table, j->m, k, j->cmask) : bsearch64(j->table, j->m, k);
            if (pos >= 0 && j->maxcost >= 0) {
                int64_t cst = j->costs ? j->costs[pos] : (j->cmask ? (int64_t)(j->table[pos] & j->cmask) : 0);
                if (cst > j->maxcost) pos = -1;
            }
            if (pos >= 0) {
                pthread_mutex_lock(j->mu);
                if (*j->nhit < j->cap) { j->hit_i[*j->nhit] = i; j->hit_pos[*j->nhit] = pos; }
                (*j->nhit)++;
                pthread_mutex_unlock(j->mu);
            }
        }
    }
    return NULL;
}

static int64_t run(job_t base, int64_t n, int threads) {
    if (threads < 1) threads = 1;
    if (threads > 64) threads = 64;
    if (n < 4096) threads = 1;
    pthread_t th[64]; job_t jobs[64];
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; int64_t nhit = 0;
    for (int t = 0; t < threads; t++) {
        jobs[t] = base; jobs[t].lo = n * t / threads; jobs[t].hi = n * (t + 1) / threads;
        jobs[t].mu = &mu; jobs[t].nhit = &nhit;
        pthread_create(&th[t], NULL, worker, &jobs[t]);
    }
    for (int t = 0; t < threads; t++) pthread_join(th[t], NULL);
    return nhit;
}

/* states (n x 100) -> keys (n); elem may be NULL */
void coset_keys(const int8_t *states, int64_t n, const int8_t *elem, const spec_t *sp, uint64_t *keys, int threads) {
    job_t j = {0}; j.states = states; j.elem = elem; j.sp = sp; j.keys = keys; j.mode = 1;
    run(j, n, threads);
}

/* states . elem -> out (n x 100), and keys if non-NULL */
void coset_apply(const int8_t *states, int64_t n, const int8_t *elem, const spec_t *sp, int8_t *out, uint64_t *keys, int threads) {
    job_t j = {0}; j.states = states; j.elem = elem; j.sp = sp; j.out = out; j.keys = keys; j.mode = 0;
    run(j, n, threads);
}

/* For each element in turn: key(states . elem) probed against the sorted table.  Hits are appended
 * (element index, state index, table position) up to cap.  Stops after the first element that
 * produced at least `want` hits.  Returns the number of elements processed; *nhit_out = hits found. */
int64_t coset_sweep(const int8_t *states, int64_t n, const int8_t *elems, int64_t nelem, const spec_t *sp,
                    const uint64_t *table, int64_t m, int64_t *hit_e, int64_t *hit_i, int64_t *hit_pos,
                    int64_t cap, int64_t want, int64_t *nhit_out, int threads, uint64_t cmask,
                    const int16_t *costs, int64_t maxcost) {
    int64_t total = 0, e;
    for (e = 0; e < nelem; e++) {
        job_t j = {0}; j.states = states; j.elem = elems + e * SB; j.sp = sp; j.mode = 2; j.cmask = cmask;
        j.costs = costs; j.maxcost = maxcost;
        j.table = table; j.m = m; j.hit_i = hit_i + total; j.hit_pos = hit_pos + total; j.cap = cap - total;
        int64_t h = run(j, n, threads);
        if (h > cap - total) h = cap - total;
        for (int64_t k = 0; k < h; k++) hit_e[total + k] = e;
        total += h;
        if (total >= want) { e++; break; }
    }
    *nhit_out = total;
    return e;
}

/* apply a different sequence of elements to the same start state for each of ncand candidates:
   candidate c uses elems[idx[t]] for t in offs[c] .. offs[c+1]-1, in order; result in out[c] */
void coset_apply_seqs(const int8_t *start, const int8_t *elems, const int64_t *offs, const int32_t *idx,
                      int64_t ncand, int8_t *out) {
    int8_t a[SB], b[SB];
    for (int64_t c = 0; c < ncand; c++) {
        memcpy(a, start, SB);
        for (int64_t t = offs[c]; t < offs[c + 1]; t++) { apply1(a, elems + (int64_t)idx[t] * SB, b); memcpy(a, b, SB); }
        memcpy(out + c * SB, a, SB);
    }
}

/* ---- open-addressing hash set of uint64 keys (key 0 is remapped to 1); used to dedupe coset labels ---- */
typedef struct { uint64_t *slots; uint64_t mask; int64_t count; } kset_t;

void *kset_new(int log2cap) {
    kset_t *s = (kset_t *)malloc(sizeof(kset_t));
    s->mask = (1ULL << log2cap) - 1; s->count = 0;
    s->slots = (uint64_t *)calloc(1ULL << log2cap, sizeof(uint64_t));
    return s;
}
void kset_free(void *p) { kset_t *s = (kset_t *)p; free(s->slots); free(s); }
int64_t kset_count(void *p) { return ((kset_t *)p)->count; }

static void kset_grow(kset_t *s) {
    uint64_t oldcap = s->mask + 1; uint64_t *old = s->slots;
    s->mask = oldcap * 2 - 1; s->slots = (uint64_t *)calloc(oldcap * 2, sizeof(uint64_t));
    for (uint64_t i = 0; i < oldcap; i++) {
        uint64_t k = old[i]; if (!k) continue;
        uint64_t h = fin(k) & s->mask;
        while (s->slots[h]) h = (h + 1) & s->mask;
        s->slots[h] = k;
    }
    free(old);
}

/* insert keys; newmask[i] = 1 if key i was not present before (first occurrence within this call wins).
 * The table doubles when more than half full. */
int64_t kset_insert(void *p, const uint64_t *keys, int64_t n, uint8_t *newmask) {
    kset_t *s = (kset_t *)p; int64_t added = 0;
    for (int64_t i = 0; i < n; i++) {
        if ((uint64_t)s->count * 4 > (uint64_t)s->mask * 3) kset_grow(s);   /* load factor 0.75 */
        uint64_t k = keys[i] ? keys[i] : 1;
        uint64_t h = fin(k) & s->mask;
        for (;;) {
            uint64_t v = s->slots[h];
            if (v == k) { newmask[i] = 0; break; }
            if (v == 0) { s->slots[h] = k; s->count++; newmask[i] = 1; added++; break; }
            h = (h + 1) & s->mask;
        }
    }
    return added;
}
/* membership test: out[i] = 1 if present */
void kset_has(void *p, const uint64_t *keys, int64_t n, uint8_t *out) {
    kset_t *s = (kset_t *)p;
    for (int64_t i = 0; i < n; i++) {
        uint64_t k = keys[i] ? keys[i] : 1;
        uint64_t h = fin(k) & s->mask;
        for (;;) {
            uint64_t v = s->slots[h];
            if (v == k) { out[i] = 1; break; }
            if (v == 0) { out[i] = 0; break; }
            h = (h + 1) & s->mask;
        }
    }
}
