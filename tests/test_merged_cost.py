import ctypes as ct
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from sympy.combinatorics import PermutationGroup

import coset as C
import groupfacts as GF
import ladder as L
import model as M
sys.path.insert(0,str(Path(L.__file__).parent/'fulltab'))
import ftlib as F
import table as T


def packed(words):
    return np.array([F.RUNS.index(r) for r in L.simplify(' '.join(words))],np.int8)


def test_merged_score_matches_python():
    rng=np.random.default_rng(91)
    cases=[(['R','U','F4'],['F4',"U'"],['R']),([],[],[])]
    for _ in range(1000):
        cases.append(tuple([F.RUNS[r] for r in rng.integers(21,size=int(rng.integers(35)))] for _ in range(3)))
    for prefix,forward,back in cases:
        a,b,c=map(packed,(prefix,forward,back))
        score=F.lib.merged_score(F.p(a,F.i8p),len(a),F.p(b,F.i8p),len(b),F.p(c,F.i8p),len(c))
        assert score==L.htm(' '.join(prefix+forward+[C.inverse_word(w) for w in reversed(back)]))


def cyclic_spec(word):
    group=PermutationGroup([GF.perm_of(M.element(M.parse(word)))])
    group.schreier_sims();levels=[]
    for base,orbit,trans in zip(group.base,group.basic_orbits,group.basic_transversals):
        orbit=sorted(orbit)
        levels.append((base,np.array(orbit),np.array([trans[p].array_form for p in orbit],np.uint8)))
    return C.Spec(canon={'levels':levels})


def test_prefix_cancellation_wins_before_window_and_cap():
    w=object.__new__(F.Witness);w.par=np.array([-1,0,1],np.int32);w.run=np.array([-1,8,8],np.int8);w.GEN=1<<40
    states=[M.State(),M.apply_seq(M.State(),'R2'),M.apply_seq(M.State(),'R4')]
    fwd=SimpleNamespace(W=w,maps=F.maps50(C.pack_states(states)),cost=np.array([0,1,2],np.int8),E=2,moves=w.seq)
    sp=cyclic_spec('R');tab=np.array([F.labels50(C.pack_states([M.State()]),sp)[0]&~F.CM],np.uint64)
    table=SimpleNamespace(r='toy',W=w,wit=np.array([0],np.int64),tab=tab,B=4,bidx=F.bucket_index(tab,4),spT=sp,mir=False,tmax=0,moves=lambda i:[])
    solver=T.FullRung(table,fwd,max_candidates=1)
    assert solver.solve_many([M.State()],window=0)[0][0][0]==0
    answer=solver.solve_many([M.State()],window=0,prefixes=[['R4']])[0]
    assert len(answer)==1
    assert L.htm(' '.join(['R4']+answer[0][1]))==0
    assert answer[0][0]==2  # This would lose the old raw-cost window to the empty path.
    assert answer[0][3]==states[2]
    # The winning hit arrives last. A deliberately tiny buffer must grow and
    # retry rather than silently keeping only the first, more expensive hit.
    hits=F.probe_merged(fwd,table,C.pack_states([M.State()]),[0],[['R4']],0,1,threads=1)
    assert hits[:,5].min()==0 and any(hits[:,1]==2)


def test_witness_encodings():
    w=object.__new__(F.Witness2)
    w.par=np.array([-1,0,1],np.int32);w.run=np.array([-1,0,8],np.int8)
    w.P1=np.array([2],np.int32);w.R1=np.array([14],np.int8)
    w.GEN1,w.GEN2=1<<40,1<<41
    ids=[0,1,2,w.GEN1,w.GEN2+5,w.GEN2+(1<<39)+(2<<5)+4]
    obj=SimpleNamespace(W=w,wp=np.array(ids,np.uint64),wq=np.array(ids[::-1],np.uint64))
    source=F.MoveSource(obj)
    for i in range(len(ids)):
        assert L.simplify(' '.join(source.words(i)))==L.simplify(' '.join(w.element(ids[i],ids[-i-1])))
    chain=SimpleNamespace(W=w,rows=np.array([0,1,2],np.int64),DER=np.array([-1,0,1<<20],np.int64),
                          bwp=np.array([2],np.int64),bwq=np.array([0],np.int64),pseq=np.array(["R'4 U U' F2"]))
    source=F.MoveSource(chain)
    for i in range(3):
        assert L.simplify(' '.join(source.words(i)))==L.simplify(' '.join(w.seq(2)+[str(chain.pseq[0])]*i))


def test_prefix_cost_in_inner_frames():
    root=Path(L.__file__).parent
    if not (root/'tables/bsgs_H1.pkl').exists(): pytest.skip('requires H1 chain')
    lad=object.__new__(L.Ladder);lad.spec4=C.Spec(canon='H1');lad.k4=lad.spec4.keys(C.pack_states([M.State()]))[0]
    lad._init_inner_frames();rng=np.random.default_rng(19)
    for _ in range(3):
        end=M.apply_seq(M.State(),' '.join(rng.choice(['U','R2','F2'],size=25)))
        prefix=list(rng.choice(L.RUNS,size=30))
        for fi in range(1,len(lad.inner_rots)):
            inner=lad._prefix_in_frame(prefix,fi,end)
            for seq in ([C.inverse_word(w) for w in inner[-4:][::-1]],['R','U2',"F'2"]):
                standard=lad.rename_inner(seq,fi,end)
                assert L.htm(' '.join(inner+seq))==L.htm(' '.join(prefix+standard))


def test_generated_table_witnesses():
    """C decoding must agree with Python for each real storage format."""
    root=Path(L.__file__).parent;data=root/'fulltab/data';mm=data/'mm'
    required=['rung12_b9_tab.npy','rung3_chain20y2_tab.npy','rung4_chain28_tab.npy',
              'rung5_chain28_tab.npy','full_rung7_s10m_tab.npy','fwd_rung7_E11_maps.npy']
    if not all((mm/name).exists() for name in required): pytest.skip('requires generated solver tables')
    factories=[lambda:T.BallForward(4),lambda:T.BallTable(9),lambda:T.Forward('7',11),
               lambda:T.ChainedTable('3',20,'Y2','y2'),lambda:T.ChainedTable('4',28,'H1p'),
               lambda:T.ChainedTable('5',28,'H1'),lambda:T.SlicedTable('7','full_rung7_s10m',10,'F16',levels=1)]
    if (mm/'full_rung7_t11m_tab.npy').exists():
        factories.append(lambda:T.SlicedTable('7','full_rung7_t11m',11,'F16'))
    rng=np.random.default_rng(18)
    for factory in factories:
        obj=factory();source=F.MoveSource(obj)
        for i in rng.integers(source.s.nitem,size=40):
            expected=L.simplify(' '.join(obj.moves(int(i))))
            assert L.simplify(' '.join(source.words(int(i))))==expected
