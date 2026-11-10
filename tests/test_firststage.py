"""The small direct-search tables build in seconds; no full solver tables are required."""
from collections import deque
import numpy as np
import pytest

import model as M
import ladder as L
from firststage import DirectFirstStage, MOVES, COST, NC, EQ, bitmask, relabel_to_z


@pytest.fixture(scope='module')
def solver():
    return DirectFirstStage()


def target(st):
    """Independent physical predicate, without pattern tables or C coordinates."""
    colors = M.GEAR_COLOR[st.gp[L.EQG]]
    return (not st.co.any() and sorted(st.rp[L.UDE]) == L.UDE
            and int(st.ro[L.UDE].sum()) % 2 == 0
            and np.array_equal(np.bincount(colors, minlength=6), [0, 0, 2, 2, 2, 2])
            and int(st.go[L.EQG].sum()) % 2 == 0)


def coordinates(solver, st):
    selected = np.isin(st.rp, L.UDE)
    c = int(np.dot(st.co[:6], 3 ** np.arange(6)))
    r = int(solver.rindex[bitmask(selected)]) * 2 + int(st.ro[selected].sum()) % 2
    return c, r


def test_coordinate_transitions(solver):
    rng = np.random.default_rng(42)
    for _ in range(50):
        st = M.random_state(rng); c, r = coordinates(solver, st)
        masks = [bitmask(M.GEAR_COLOR[st.gp] == col) for col in range(6)]
        for j, move in enumerate(MOVES):
            after = st.apply(move); cc, rr = coordinates(solver, after)
            assert (solver.ct[c, j], solver.rt[r, j]) == (cc, rr)
            def permute(mask):
                return int(solver.lut[j, 0, mask & 255] | solver.lut[j, 1, (mask >> 8) & 255]
                           | solver.lut[j, 2, mask >> 16])
            assert [permute(x) for x in masks] == [bitmask(M.GEAR_COLOR[after.gp] == col) for col in range(6)]
            assert permute(bitmask(st.go % 2)) ^ int(solver.spin[j]) == bitmask(after.go % 2)


def brute_distance(st, bound):
    queue = deque([(st, 0)]); seen = {st.key()}
    while queue:
        state, depth = queue.popleft()
        if target(state): return depth
        if depth == bound: continue
        for move, cost in zip(MOVES, COST):
            if cost != 1: continue
            child = state.apply(move)
            if child.key() not in seen:
                seen.add(child.key()); queue.append((child, depth + 1))
    return None


def test_shallow_optimality(solver):
    rng = np.random.default_rng(123)
    unit = [m for m, cost in zip(MOVES, COST) if cost == 1]
    for depth in range(4):
        for _ in range(5):
            st = M.State()
            for j in rng.integers(len(unit), size=depth): st = st.apply(unit[j])
            result = solver.solve(st, max_candidates=1)
            assert result[0] == brute_distance(st, depth)


def test_random_landings_and_continuations(solver):
    rng = np.random.default_rng(19)
    states = [M.random_state(rng) for _ in range(100)]
    answers = solver.solve_many(states, window=1, max_candidates=10)
    for st, candidates in zip(states, answers):
        assert candidates
        assert candidates[0][0] <= candidates[-1][0] <= candidates[0][0] + 1
        for cost, words, _, landed in candidates:
            physical = M.apply_seq(st, ' '.join(words))
            assert target(physical) and target(landed)
            assert not M.check(landed)
            assert physical.visible() == landed.visible()
            assert physical.apply(MOVES[9]).visible() == landed.apply(MOVES[9]).visible()
            assert cost == L.htm(' '.join(words))


def test_solved_and_budget(solver):
    assert solver.solve(M.State(), max_candidates=1)[0] == 0
    old = solver.max_nodes
    try:
        solver.max_nodes = 1
        result = solver.solve(M.random_state(np.random.default_rng(7)), max_candidates=1)
        assert result is None and solver.last_stats[0]['exhausted']
    finally:
        solver.max_nodes = old
    assert solver.solve_many([]) == []


def test_invalid_goal():
    st = M.State(); st.co[0] = 1; st.co[1] = 2
    with pytest.raises(ValueError): relabel_to_z(st)


def test_finish_uses_prefix_cancellation():
    # U4 and R4 commute. Both stored paths reach the same state at cost 4,
    # but choosing the first on a tie produces 6 HTM after prefix R4; the
    # second produces 2. This also catches discarding equal-cost finishes.
    solver = object.__new__(L.Ladder)
    first, second = ['R4', 'U4'], ['U4', 'R4']
    st = M.apply_seq(M.State(), ' '.join(first))
    assert st == M.apply_seq(M.State(), ' '.join(second))
    solver._finishes = lambda state: iter([(4, first), (4, second)])
    cost, tail = solver._best_finish(st, ['R4'])
    assert cost == 2
    assert M.apply_seq(st, ' '.join(tail)).is_visibly_solved()
    assert L.htm(' '.join(['R4'] + [L.CS.inverse_word(w) for w in reversed(first)])) == 6
