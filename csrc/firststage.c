/* Direct, visible-state IDA* for the first rung. All move data come from model.py. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NC 729
#define NM 21
#define GB (1u << 24)

static uint32_t permute(uint32_t mask, const uint32_t *lut) {
    return lut[mask & 255] | lut[256 + ((mask >> 8) & 255)] | lut[512 + (mask >> 16)];
}

/* Unit-cost moves are enough for BFS; the search also uses the cost-2 runs. */
uint32_t first_joint(uint8_t *dist, const uint16_t *ct, const uint32_t *rt,
                     uint32_t nr, uint32_t goal, const uint8_t *cost) {
    uint32_t n = nr * NC, *queue = malloc((size_t)n * sizeof *queue);
    if (!queue) return 0;
    memset(dist, 255, n);
    uint32_t head = 0, tail = 1; queue[0] = goal; dist[goal] = 0;
    while (head < tail) {
        uint32_t x = queue[head++], r = x / NC, c = x % NC;
        uint8_t d = dist[x] + 1;
        for (int m = 0; m < NM; m++) if (cost[m] == 1) {
            uint32_t y = rt[r * NM + m] * NC + ct[c * NM + m];
            if (dist[y] == 255) { dist[y] = d; queue[tail++] = y; }
        }
    }
    free(queue); return tail;
}

uint32_t first_gears(uint8_t *dist, const uint32_t *masks, uint32_t n,
                     uint32_t eq, int neq, const uint32_t *lut, const uint8_t *cost) {
    uint32_t *queue = malloc((size_t)n * sizeof *queue);
    if (!queue) return 0;
    memset(dist, 255, GB);
    uint32_t head = 0, tail = 0;
    for (uint32_t i = 0; i < n; i++) if (__builtin_popcount(masks[i] & eq) == neq) {
        queue[tail++] = masks[i]; dist[masks[i]] = 0;
    }
    while (head < tail) {
        uint32_t x = queue[head++]; uint8_t d = dist[x] + 1;
        for (int m = 0; m < NM; m++) if (cost[m] == 1) {
            uint32_t y = permute(x, lut + m * 768);
            if (dist[y] == 255) { dist[y] = d; queue[tail++] = y; }
        }
    }
    free(queue); return tail;
}

typedef struct {
    const uint8_t *joint, *side, *ud, *cost;
    const uint16_t *ct;
    const uint32_t *rt, *lut, *spin;
    uint32_t goal, eq;
    uint64_t nodes, limit;
    int path[64], *paths, *lengths, cap, found, stopped;
} search_t;

static void dfs(search_t *s, uint32_t c, uint32_t r, const uint32_t *g, uint32_t spin,
                int left, int prev, int level) {
    if (s->stopped) return;
    if (s->limit && s->nodes >= s->limit) { s->stopped = 1; return; }
    s->nodes++;
    if (s->joint[r * NC + c] > left || s->ud[g[0]] > left) return;
    for (int i = 1; i < 5; i++) if (s->side[g[i]] > left) return;
    if (left == 0) {
        if (r * NC + c != s->goal || (__builtin_popcount(spin & s->eq) & 1)) return;
        memcpy(s->paths + s->found * 64, s->path, level * sizeof(int));
        s->lengths[s->found++] = level;
        if (s->found == s->cap) s->stopped = 1;
        return;
    }
    if (level == 63) return;
    for (int m = 0; m < NM; m++) {
        if (m / 7 == prev || s->cost[m] > left) continue;
        uint32_t c2 = s->ct[c * NM + m], r2 = s->rt[r * NM + m];
        int rem = left - s->cost[m];
        if (s->joint[r2 * NC + c2] > rem) continue;
        const uint32_t *lut = s->lut + m * 768;
        uint32_t g2[5]; g2[0] = permute(g[0], lut);
        if (s->ud[g2[0]] > rem) continue;
        int i;
        for (i = 1; i < 5; i++) {
            g2[i] = permute(g[i], lut);
            if (s->side[g2[i]] > rem) break;
        }
        if (i != 5) continue;
        s->path[level] = m;
        dfs(s, c2, r2, g2, permute(spin, lut) ^ s->spin[m], rem, m / 7, level + 1);
        if (s->stopped) return;
    }
}

int first_search(const uint8_t *joint, const uint8_t *side, const uint8_t *ud,
                  const uint16_t *ct, const uint32_t *rt, const uint32_t *lut,
                  const uint32_t *dspin, const uint8_t *cost, uint32_t goal, uint32_t eq,
                  uint32_t c, uint32_t r, const uint32_t *g, uint32_t spin,
                  int maxdepth, int window, uint64_t limit, int cap,
                  int *paths, int *lengths, uint64_t *stats) {
    search_t s = { .joint=joint, .side=side, .ud=ud, .cost=cost, .ct=ct, .rt=rt,
        .lut=lut, .spin=dspin, .goal=goal, .eq=eq, .limit=limit, .cap=cap,
        .paths=paths, .lengths=lengths };
    int bound = joint[r * NC + c], best = -1;
    for (; bound <= maxdepth && !s.stopped; bound++) {
        dfs(&s, c, r, g, spin, bound, -1, 0);
        if (s.found && best < 0) best = bound;
        if (best >= 0 && bound >= best + window) break;
    }
    stats[0] = s.nodes; stats[1] = best < 0 ? UINT64_MAX : (uint64_t)best;
    stats[2] = s.stopped && s.found < cap;
    return s.found;
}
