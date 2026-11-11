"""Pure-Python tests for spectral_mapping_heavyhex.py (no Qiskit required)."""

import networkx as nx

from spectral_mapping_heavyhex import (
    _adjacent_token_swaps,
    _build_schedule,
    _choose_spectral_order,
    extract_hardware_path,
    validate_path,
)


class _FakeBackend:
    def __init__(self, n, edges):
        self.num_qubits = n
        self.coupling_map = list(edges)


def test_extracts_a_valid_hardware_path():
    graph = nx.Graph()
    graph.add_nodes_from(range(8))
    graph.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
        (1, 7), (2, 6),
    ])
    backend = _FakeBackend(8, graph.edges())
    path = extract_hardware_path(backend, 6, coupling_graph=graph)
    assert len(path) == 6
    assert validate_path(path, graph)


def test_token_swaps_reach_target_order():
    start = [0, 1, 2, 3, 4]
    target = [3, 0, 4, 1, 2]
    swaps = _adjacent_token_swaps(start, target)
    current = start[:]
    for left, right in swaps:
        assert right == left + 1
        current[left], current[right] = current[right], current[left]
    assert current == target


def test_forced_coupling_makes_a_front_gate_adjacent():
    gates = [(0, 0, 3), (1, 1, 2), (2, 0, 1), (3, 2, 3)]
    order = _choose_spectral_order(4, gates, None, alpha=0.8, beta=0.2, tau=4)
    positions = {q: i for i, q in enumerate(order)}
    # At least one front-layer operation must be placeable on the line.
    assert any(abs(positions[q0] - positions[q1]) == 1 for _, q0, q1 in gates[:2])


def test_full_schedule_keeps_every_two_qubit_gate_adjacent():
    n = 6
    ops = [(0, 1), (2, 3), (4, 5), (1, 2), (3, 4), (0, 5), (0, 3), (1, 4)]
    cargs = [()] * len(ops)
    kinds = ["cx"] * len(ops)
    path = [10, 11, 12, 13, 14, 15]
    schedule = _build_schedule(
        n, ops, cargs, kinds, path, alpha=0.8, beta=0.2, tau=None
    )

    order = schedule.initial_order[:]
    seen = []
    for event in schedule.events:
        if event[0] == "swap":
            _, left, right = event
            assert right == left + 1
            order[left], order[right] = order[right], order[left]
        else:
            _, instruction_index, positions = event
            q0, q1 = ops[instruction_index]
            assert order[positions[0]] == q0
            assert order[positions[1]] == q1
            assert abs(positions[0] - positions[1]) == 1
            seen.append(instruction_index)

    assert sorted(seen) == list(range(len(ops)))
    assert order == schedule.final_order
