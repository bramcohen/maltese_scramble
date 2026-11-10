"""Coset direction matters even when every representative is a legal invisible state."""
import itertools
from pathlib import Path
import numpy as np
import pytest
from sympy.combinatorics import PermutationGroup

import coset as C
import groupfacts as GF
import model as M


def small_example():
    # S3 on three white gears, with odd permutations compensated by a swap
    # of two yellow gears. All six states are legal invisible relabellings.
    states = []
    for perm in itertools.permutations(range(3)):
        st = M.State(); st.gp[:3] = perm
        if M._parity(st.gp): st.gp[[4, 5]] = st.gp[[5, 4]]
        assert not M.check(st) and st.is_visibly_solved()
        states.append(st)
    swap = next(s for s in states if tuple(s.gp[:3]) == (1, 0, 2))
    subgroup = PermutationGroup([GF.perm_of(swap.as_element())])
    subgroup.schreier_sims()
    levels = []
    for b, orbit, trans in zip(subgroup.base, subgroup.basic_orbits, subgroup.basic_transversals):
        orbit = sorted(orbit)
        levels.append((b, np.array(orbit), np.array([trans[p].array_form for p in orbit], np.uint8)))
    spec = C.Spec(canon={'levels': levels})
    for choice in itertools.combinations(states, 3):
        right = C.pack_states(choice)
        wrong = C.pack_states([s.as_element().inverse().state() for s in choice])
        if len(np.unique(spec.keys(right))) == 3 and len(np.unique(spec.keys(wrong))) < 3:
            return spec, right, wrong
    raise AssertionError('expected a non-normal subgroup example')


def test_repairs_inverse_transversal():
    spec, right, wrong = small_example()
    repaired = C.right_transversal(wrong, spec)
    assert np.array_equal(repaired, right)
    assert len(np.unique(spec.keys(repaired))) == 3
    assert np.array_equal(C.right_transversal(right, spec), right)
    # Each inverted representative is a state that can be visibly renamed
    # into the target; all must be covered by the repaired transversal.
    goal = spec.keys(C.pack_states([M.State()]))[0]
    for row in right:
        state = C.unpack_state(row).as_element().inverse().state()
        copies = [C.unpack_state(rep).apply(state.as_element()) for rep in repaired]
        assert np.any(spec.keys(C.pack_states(copies)) == goal)


def test_rejects_duplicate_representatives():
    spec, right, _ = small_example()
    with pytest.raises(ValueError): C.right_transversal(np.repeat(right[:1], 3, axis=0), spec)


def test_existing_first_stage_regression():
    root = Path(C.__file__).parent
    path = root / 'fulltab/data/T_reps_12.npy'
    if not path.exists() or not (root / 'tables/bsgs_Z.pkl').exists():
        pytest.skip('requires the small generated relabelling and subgroup files')
    spec = C.Spec(canon='Z'); rows = C.right_transversal(np.load(path), spec)
    assert len(np.unique(spec.keys(rows))) == 1296
    state = M.random_state(np.random.default_rng(20261109))
    word = "R F U' F'2 U2 R'3 F U' R' U2 R"
    state = M.apply_seq(state, word)
    copies = [C.unpack_state(row).apply(state.as_element()) for row in rows]
    goal = spec.keys(C.pack_states([M.State()]))[0]
    assert np.any(spec.keys(C.pack_states(copies)) == goal)
