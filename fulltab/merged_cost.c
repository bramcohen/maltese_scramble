/* Included by ft.c. Decode existing witnesses and score complete prefixes
 * before applying the candidate window. No new generated tables are needed. */
#define MC_WORD 256
#define MC_PREFIX 512
typedef struct {
    int kind, levels; /* 0 ball row; 1 indirect ball row; 2 pair; 3 chain */
    const int32_t *par, *p1;
    const int8_t *run, *r1;
    const uint64_t *p, *q;
    const int64_t *rows, *der;
    const int8_t *pieces;
    const int32_t *offset;
    int64_t npar, n1, nitem, nbase, nder, npiece;
} mc_source;
static const int mc_power[7] = {1,2,3,4,7,6,5};
static const int mc_index[8] = {-1,0,1,2,3,6,5,4};
static const int mc_inverse[7] = {4,5,6,3,0,1,2};
static int mc_inv(int r) { return r/7*7+mc_inverse[r%7]; }
static int mc_cost(int r) { int p=mc_power[r%7]; return p==1 || p==2 || p==6 || p==7 ? 1 : 2; }
static int mc_ball(const mc_source *w, uint64_t sid, int8_t *out) {
    int8_t rev[MC_WORD], tail[2]; int n=0, nt=0;
    if (w->levels==2 && sid >= (1ULL<<41)) {
        uint64_t v=sid-(1ULL<<41), idx=(v&((1ULL<<39)-1))>>5;
        if ((v&31)>=21) return -1;
        if ((v>>39)&1) sid=idx;
        else {
            if (idx >= (uint64_t)w->n1) return -1;
            tail[nt++]=w->r1[idx]; sid=w->p1[idx];
        }
        tail[nt++]=v&31;
    } else if (sid >= (1ULL<<40)) {
        if (w->levels==2) {
            uint64_t idx=sid-(1ULL<<40); if (idx >= (uint64_t)w->n1) return -1;
            tail[nt++]=w->r1[idx]; sid=w->p1[idx];
        } else { tail[nt++]=sid&31; sid=(sid-(1ULL<<40))>>5; }
    }
    while (sid) {
        if (sid >= (uint64_t)w->npar || n >= MC_WORD-2) return -1;
        int r=w->run[sid]; if (r<0 || r>=21) return -1;
        rev[n++]=r; sid=w->par[sid];
    }
    for (int i=0;i<n;i++) out[i]=rev[n-1-i];
    for (int i=0;i<nt;i++) { if (tail[i]<0 || tail[i]>=21) return -1; out[n++]=tail[i]; }
    return n;
}
static int mc_pair(const mc_source *w, int64_t row, int8_t *out) {
    if (row<0 || row>=w->nbase) return -1;
    int8_t a[MC_WORD],b[MC_WORD]; int na=mc_ball(w,w->p[row],a), nb=mc_ball(w,w->q[row],b);
    if (na<0 || nb<0 || na+nb>MC_WORD) return -1;
    memcpy(out,a,na); for (int i=0;i<nb;i++) out[na+i]=mc_inv(b[nb-1-i]);
    return na+nb;
}
int merged_decode(const mc_source *w, int64_t item, int8_t *out) {
    if (item<0 || item>=w->nitem) return -1;
    if (w->kind==0) return mc_ball(w,item,out);
    if (w->kind==1) return mc_ball(w,w->p[item],out);
    if (w->kind==2) return mc_pair(w,item,out);
    int64_t row=w->rows[item]; int path[MC_WORD], np=0;
    for (;;) {
        if (row<0 || row>=w->nder) return -1;
        int64_t d=w->der[row]; if (d==-1) break;
        int p=d&((1<<20)-1); if (p>=w->npiece || np>=MC_WORD) return -1;
        path[np++]=p; row=d>>20;
    }
    int n=mc_pair(w,row,out); if (n<0) return -1;
    for (int j=np-1;j>=0;j--) {
        int p=path[j], count=w->offset[p+1]-w->offset[p];
        if (count<0 || n+count>MC_WORD) return -1;
        memcpy(out+n,w->pieces+w->offset[p],count); n+=count;
    }
    return n;
}
static void mc_push(int8_t *stack,int *n,int *cost,int r) {
    if (*n && stack[*n-1]/7==r/7) {
        int prev=stack[--*n], p=(mc_power[prev%7]+mc_power[r%7])&7;
        *cost-=mc_cost(prev);
        if (p) { r=r/7*7+mc_index[p]; stack[(*n)++]=r; *cost+=mc_cost(r); }
    } else { stack[(*n)++]=r; *cost+=mc_cost(r); }
}
int merged_score(const int8_t *prefix,int np,const int8_t *forward,int nf,const int8_t *back,int nb) {
    if (np>MC_PREFIX || nf>MC_WORD || nb>MC_WORD) return -1;
    int8_t stack[MC_PREFIX+2*MC_WORD]; int n=0,cost=0;
    for (int i=0;i<np;i++) mc_push(stack,&n,&cost,prefix[i]);
    for (int i=0;i<nf;i++) mc_push(stack,&n,&cost,forward[i]);
    for (int i=nb-1;i>=0;i--) mc_push(stack,&n,&cost,mc_inv(back[i]));
    return cost;
}
typedef struct {
    pmulti_t base;
    const mc_source *fw,*bw;
    const int32_t *groups,*poff;
    const int8_t *prefix;
    const uint8_t *runmaps;
    uint64_t goal;
} mc_probe;
static void *mc_worker(void *arg) {
    mc_probe *Q=arg; pmulti_t *P=&Q->base;
    int64_t nb=(P->n+PM_BLOCK-1)/PM_BLOCK;
    for (;;) {
        int64_t task=atomic_fetch_add(&P->next,1); if (task>=P->nx*nb) break;
        int64_t xi=task/nb, begin=(task%nb)*PM_BLOCK, end=begin+PM_BLOCK<P->n?begin+PM_BLOCK:P->n;
        int g=Q->groups[xi], np=Q->poff[g+1]-Q->poff[g];
        const int8_t *prefix=Q->prefix+Q->poff[g]; const uint8_t *X=P->xs+xi*64;
        for (int64_t i=begin;i<end;i+=2) {
            int two=i+1<end;
            uint8x16x4_t a=compose50(P->el+i*64,X), b=two?compose50(P->el+(i+1)*64,X):a;
            uint64_t ka,kb;
            if (P->mir) { ka=canon50m(a,P->F,P->mir); kb=two?canon50m(b,P->F,P->mir):ka; }
            else canon50x2(a,b,P->F,&ka,&kb);
            for (int e=0;e<1+two;e++) {
                int64_t ti=bucket_find(P->tab,P->bidx,P->bbits,e?kb:ka); if (ti<0) continue;
                int8_t f[MC_WORD],w[MC_WORD];
                int nf=merged_decode(Q->fw,i+e,f), nw=merged_decode(Q->bw,ti,w);
                if (nf<0 || nw<0) continue;
                /* Mirror-canonical matches need not be actual target matches.
                 * Reject them before they can lower the running best. */
                if (P->mir) {
                    uint8_t st[64]; uint8x16x4_t v=e?b:a; vst1q_u8_x4(st,v);
                    for (int j=nw-1;j>=0;j--) { v=compose50(Q->runmaps+mc_inv(w[j])*64,st); vst1q_u8_x4(st,v); }
                    if ((canon50(v,P->F)&~31ULL)!=Q->goal) continue;
                }
                int cost=merged_score(prefix,np,f,nf,w,nw);
                int best=atomic_load_explicit(&P->best[g],memory_order_relaxed);
                while (cost<best && !atomic_compare_exchange_weak(&P->best[g],&best,cost)) ;
                if (cost>(best<cost?best:cost)+P->window) continue;
                int64_t at=atomic_fetch_add(&P->nh,1);
                if (at<P->hcap) {
                    int64_t *h=P->hits+at*6;
                    h[0]=xi;h[1]=i+e;h[2]=ti;h[3]=P->ec[i+e];h[4]=P->tab[ti]&31ULL;h[5]=cost;
                }
            }
        }
    }
    return NULL;
}
int64_t probe50_merged(const uint8_t *el,const int8_t *ec,int64_t nf,const int8_t *xs,int64_t nx,
                       const uint64_t *tab,const uint32_t *bidx,int bits,const fs_t *F,const mir_t *mir,
                       const mc_source *fw,const mc_source *bw,const int32_t *groups,int ng,
                       const int8_t *prefix,const int32_t *poff,const uint8_t *runmaps,uint64_t goal,
                       int window,int threads,int64_t *hits,int64_t cap) {
    uint8_t *X=aligned_alloc(64,(size_t)nx*64);
    for (int64_t i=0;i<nx;i++) pm50(xs+i*100,X+i*64);
    _Atomic int *best=malloc(sizeof(*best)*ng); for (int i=0;i<ng;i++) atomic_store(&best[i],INT32_MAX);
    mc_probe Q={0}; pmulti_t *P=&Q.base;
    P->el=el;P->ec=ec;P->n=nf;P->xs=X;P->nx=nx;P->tab=tab;P->bidx=bidx;P->bbits=bits;P->F=F;P->mir=mir;
    P->best=best;P->window=window;P->hits=hits;P->hcap=cap;
    Q.fw=fw;Q.bw=bw;Q.groups=groups;Q.prefix=prefix;Q.poff=poff;Q.runmaps=runmaps;Q.goal=goal;
    pthread_t th[64]; for (int i=0;i<threads;i++) pthread_create(&th[i],NULL,mc_worker,&Q);
    for (int i=0;i<threads;i++) pthread_join(th[i],NULL);
    free(X);free(best);return P->nh;
}
