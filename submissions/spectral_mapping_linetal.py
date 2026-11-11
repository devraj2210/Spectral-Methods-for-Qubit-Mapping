"""Spectral qubit mapping on a path extracted from a heavy-hex coupling map.

Implements the one-dimensional spectral mapper of:
  J. X. Lin, E. R. Anschuetz, and A. W. Harrow,
  "Using Spectral Graph Theory to Map Qubits onto Connectivity-limited Devices",
  ACM TQC 2(1), 2021, DOI: 10.1145/3436752.

The paper's full heuristic is designed for a linear-nearest-neighbour (LNN)
architecture.  This module therefore extracts a hardware path (a path subgraph,
not the graph-theoretic line graph) and runs the mapper on that LNN subgraph.

Qiskit is imported lazily: path extraction and the spectral/token-swap core can
be used without Qiskit installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math

import networkx as nx
import numpy as np
from scipy.linalg import eigh


# Ten (alpha, beta) pairs selected by Lin et al. for their meta-algorithm.
PAPER_META_CONFIGS: Tuple[Tuple[float, float], ...] = (
    (0.2, 0.3),
    (0.3, 0.4),
    (0.4, 0.1),
    (0.5, 0.1),
    (0.5, 0.6),
    (0.7, 0.1),
    (0.8, 0.1),
    (0.8, 0.2),
    (0.8, 0.6),
    (0.9, 0.9),
)


@dataclass
class SpectralMappingResult:
    """Result returned by :func:`route_circuit_on_heavyhex_path`."""

    circuit: Any
    path: List[int]
    initial_layout: Dict[int, int]  # logical qubit -> hardware physical qubit
    final_layout: Dict[int, int]    # logical qubit -> hardware physical qubit
    swap_count: int
    swap_sequence: List[Tuple[int, int]]  # hardware physical-qubit pairs
    mapping_history: List[Dict[int, int]]
    alpha: float
    beta: float
    tau: int


@dataclass
class _Schedule:
    events: List[Tuple[Any, ...]]
    initial_order: List[int]  # logical tokens, indexed by path position
    final_order: List[int]
    swap_positions: List[Tuple[int, int]]
    mapping_orders: List[List[int]]


def _backend_num_qubits(backend: Any) -> int:
    n = getattr(backend, "num_qubits", None)
    if n is not None:
        return int(n)
    try:
        return int(backend.configuration().num_qubits)
    except Exception as exc:
        raise ValueError("Could not determine the backend's number of qubits.") from exc


def _backend_coupling_edges(backend: Any) -> List[Tuple[int, int]]:
    """Read coupling edges from BackendV1, BackendV2, or a CouplingMap-like object."""
    coupling = getattr(backend, "coupling_map", None)

    if coupling is None:
        target = getattr(backend, "target", None)
        if target is not None:
            builder = getattr(target, "build_coupling_map", None)
            if callable(builder):
                coupling = builder()

    if coupling is not None:
        if hasattr(coupling, "get_edges"):
            edges = list(coupling.get_edges())
        else:
            edges = list(coupling)
    else:
        try:
            edges = list(backend.configuration().coupling_map)
        except Exception as exc:
            raise ValueError(
                "Could not read a coupling map from this backend. Pass `path=` "
                "explicitly or provide a backend with coupling_map/target."
            ) from exc

    # The paper assumes undirected, symmetric connectivity.  Treat a directed
    # backend coupling map as undirected for path extraction and token swaps.
    clean = set()
    for edge in edges:
        if len(edge) < 2:
            continue
        u, v = int(edge[0]), int(edge[1])
        if u != v:
            clean.add((min(u, v), max(u, v)))
    return sorted(clean)


def _backend_edge_reliability(backend: Any) -> Dict[Tuple[int, int], float]:
    """Best available two-qubit edge reliability, defaulting to 1.0.

    When BackendV1 properties are available, the smallest reported two-qubit
    gate_error for a physical pair is used.  This is only a path-selection
    preference; the spectral interaction weights themselves follow the paper.
    """
    reliability: Dict[Tuple[int, int], float] = {}
    try:
        props = backend.properties()
        for gate in getattr(props, "gates", []):
            qubits = getattr(gate, "qubits", ())
            if len(qubits) != 2:
                continue
            errors = [
                float(param.value)
                for param in getattr(gate, "parameters", [])
                if getattr(param, "name", "") == "gate_error"
                and getattr(param, "value", None) is not None
            ]
            if not errors:
                continue
            pair = (min(int(qubits[0]), int(qubits[1])),
                    max(int(qubits[0]), int(qubits[1])))
            value = max(0.0, min(1.0, 1.0 - min(errors)))
            reliability[pair] = max(reliability.get(pair, 0.0), value)
    except Exception:
        # BackendV2 targets do not necessarily expose BackendV1 properties.
        pass
    return reliability


def coupling_graph_from_backend(backend: Any) -> nx.Graph:
    """Return an undirected NetworkX graph for a Qiskit backend."""
    n = _backend_num_qubits(backend)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    reliability = _backend_edge_reliability(backend)
    for u, v in _backend_coupling_edges(backend):
        graph.add_edge(u, v, reliability=reliability.get((u, v), 1.0))
    return graph


def _path_search(graph: nx.Graph, target_length: int,
                 max_expansions: int = 750_000) -> Optional[List[int]]:
    """Find a simple path of exactly target_length nodes using ordered DFS.

    The degree-aware neighbor ordering is a heuristic intended for sparse
    heavy-hex graphs.  It is not a longest-path solver; the expansion cap
    prevents pathological backtracking on arbitrary coupling graphs.
    """
    if target_length < 1:
        raise ValueError("target_length must be positive")
    if target_length > graph.number_of_nodes():
        return None
    if target_length == 1:
        return [min(graph.nodes)] if graph.number_of_nodes() else None

    expansions = 0
    node_quality = {}
    for node in graph.nodes:
        node_quality[node] = float(
            graph.nodes[node].get("reliability", graph.nodes[node].get("weight", 1.0))
        )

    # Low-degree starts are more likely to serve as path endpoints.  Reliability
    # breaks ties in favour of better calibrated regions.
    starts = sorted(
        graph.nodes,
        key=lambda u: (graph.degree(u), -node_quality[u], u),
    )

    for start in starts:
        path = [start]
        used = {start}

        def dfs(u: int) -> Optional[List[int]]:
            nonlocal expansions
            if len(path) == target_length:
                return list(path)
            if expansions >= max_expansions:
                return None

            candidates = [v for v in graph.neighbors(u) if v not in used]
            # Prefer continuing through vertices with more unused options, then
            # prefer higher-reliability edges/nodes. Deterministic final tie-break.
            def candidate_key(v: int) -> Tuple[float, float, float, int]:
                onward = sum(1 for w in graph.neighbors(v) if w not in used and w != u)
                edge_rel = float(
                    graph[u][v].get("reliability", graph[u][v].get("weight", 1.0))
                )
                return (-onward, -edge_rel, -node_quality[v], int(v))

            candidates.sort(key=candidate_key)
            for v in candidates:
                expansions += 1
                if expansions > max_expansions:
                    return None
                used.add(v)
                path.append(v)
                result = dfs(v)
                if result is not None:
                    return result
                path.pop()
                used.remove(v)
            return None

        result = dfs(start)
        if result is not None:
            return result
        if expansions >= max_expansions:
            break

    return None


def extract_hardware_path(
    backend: Any,
    target_length: int,
    *,
    coupling_graph: Optional[nx.Graph] = None,
    max_expansions: int = 750_000,
) -> List[int]:
    """Extract a target_length-node path subgraph from a hardware coupling map.

    This is the 1D LNN fallback for heavy-hex. Consecutive entries in the
    returned list are guaranteed to be adjacent in the undirected coupling map.
    Extra heavy-hex edges are intentionally not used by this 1D implementation.
    """
    graph = coupling_graph.copy() if coupling_graph is not None else coupling_graph_from_backend(backend)
    nphysical = _backend_num_qubits(backend)
    if target_length > nphysical:
        raise ValueError(
            f"The circuit has {target_length} logical qubits, but the backend has "
            f"only {nphysical} physical qubits."
        )
    if graph.number_of_nodes() == 0:
        raise ValueError("The backend coupling graph is empty.")

    path = _path_search(graph, target_length, max_expansions=max_expansions)
    if path is None:
        raise RuntimeError(
            f"Could not find a simple hardware path containing {target_length} qubits "
            f"within {max_expansions:,} DFS expansions. Try supplying `path=` "
            "explicitly, increasing max_expansions, or using a smaller logical circuit."
        )
    return path


def _twoq_layer_data(
    remaining_twoq: Sequence[Tuple[int, int, int]],
    nlogical: int,
) -> Tuple[Dict[int, int], Dict[int, int], int, List[int]]:
    """Compute paper-style forward/reverse layers for remaining 2Q operations.

    remaining_twoq entries are (instruction_index, logical_q0, logical_q1),
    in their original circuit order. Direct blockers are the latest previous
    two-qubit operation on either involved logical qubit.
    """
    last_on_qubit: Dict[int, int] = {}
    predecessors: Dict[int, set] = {}
    pair_by_index: Dict[int, Tuple[int, int]] = {}
    ordered_ids: List[int] = []

    for instruction_index, q0, q1 in remaining_twoq:
        blockers = set()
        if q0 in last_on_qubit:
            blockers.add(last_on_qubit[q0])
        if q1 in last_on_qubit:
            blockers.add(last_on_qubit[q1])
        predecessors[instruction_index] = blockers
        pair_by_index[instruction_index] = (q0, q1)
        ordered_ids.append(instruction_index)
        last_on_qubit[q0] = instruction_index
        last_on_qubit[q1] = instruction_index

    forward: Dict[int, int] = {}
    for instruction_index in ordered_ids:
        blockers = predecessors[instruction_index]
        forward[instruction_index] = (
            0 if not blockers else 1 + max(forward[b] for b in blockers)
        )

    # Reverse layering is the same greedy layering on the reversed 2Q list.
    next_on_qubit: Dict[int, int] = {}
    reverse: Dict[int, int] = {}
    for instruction_index in reversed(ordered_ids):
        q0, q1 = pair_by_index[instruction_index]
        successors = set()
        if q0 in next_on_qubit:
            successors.add(next_on_qubit[q0])
        if q1 in next_on_qubit:
            successors.add(next_on_qubit[q1])
        reverse[instruction_index] = (
            0 if not successors else 1 + max(reverse[s] for s in successors)
        )
        next_on_qubit[q0] = instruction_index
        next_on_qubit[q1] = instruction_index

    total_layers = max(forward.values(), default=0)
    front = [idx for idx in ordered_ids if forward[idx] == 0]
    return forward, reverse, total_layers, front


def _interaction_graph(
    nlogical: int,
    remaining_twoq: Sequence[Tuple[int, int, int]],
    current_order: Optional[Sequence[int]],
    alpha: float,
    beta: float,
    tau: int,
) -> Tuple[nx.Graph, Dict[int, int], Dict[int, int], int, List[int]]:
    """Build the weighted interaction graph from Eq. (14) of the paper."""
    graph = nx.Graph()
    graph.add_nodes_from(range(nlogical))

    forward, reverse, total_layers, front = _twoq_layer_data(remaining_twoq, nlogical)
    pairs = {idx: (q0, q1) for idx, q0, q1 in remaining_twoq}

    for idx, q0, q1 in remaining_twoq:
        if forward[idx] <= tau:
            # The paper discounts gates that can be moved later. A gate with a
            # long reverse dependency chain receives a larger weight.
            weight = float(alpha) ** max(0, total_layers - reverse[idx])
            if graph.has_edge(q0, q1):
                graph[q0][q1]["weight"] += weight
            else:
                graph.add_edge(q0, q1, weight=weight)

    # Retain the previous LNN adjacency with weight beta (paper's memory term).
    if current_order is not None:
        position = {logical: pos for pos, logical in enumerate(current_order)}
        for q0 in range(nlogical):
            for q1 in range(q0 + 1, nlogical):
                if abs(position[q0] - position[q1]) == 1:
                    if graph.has_edge(q0, q1):
                        graph[q0][q1]["weight"] += float(beta)
                    else:
                        graph.add_edge(q0, q1, weight=float(beta))

    return graph, forward, reverse, total_layers, front


def _fiedler_order(graph: nx.Graph, groups: Sequence[Tuple[int, ...]],
                   current_order: Optional[Sequence[int]]) -> List[int]:
    """Sort fused/singleton groups by the Fiedler vector of their quotient graph."""
    if len(groups) <= 1:
        return list(range(len(groups)))

    logical_to_group = {}
    for group_index, group in enumerate(groups):
        for logical in group:
            logical_to_group[logical] = group_index

    quotient = nx.Graph()
    quotient.add_nodes_from(range(len(groups)))
    for q0, q1, data in graph.edges(data=True):
        g0, g1 = logical_to_group[q0], logical_to_group[q1]
        if g0 == g1:
            continue
        w = float(data.get("weight", 1.0))
        if quotient.has_edge(g0, g1):
            quotient[g0][g1]["weight"] += w
        else:
            quotient.add_edge(g0, g1, weight=w)

    # An edgeless or one-node quotient has no meaningful Fiedler direction.
    if quotient.number_of_edges() == 0:
        group_order = list(range(len(groups)))
    else:
        adjacency = nx.to_numpy_array(
            quotient, nodelist=list(range(len(groups))), weight="weight", dtype=float
        )
        lap = np.diag(adjacency.sum(axis=1)) - adjacency
        eigenvalues, eigenvectors = eigh(lap, check_finite=True)
        vector = np.asarray(eigenvectors[:, 1], dtype=float)

        if current_order is None:
            tie_key = {i: min(groups[i]) for i in range(len(groups))}
        else:
            old_pos = {logical: pos for pos, logical in enumerate(current_order)}
            tie_key = {
                i: sum(old_pos[q] for q in groups[i]) / len(groups[i])
                for i in range(len(groups))
            }
        group_order = sorted(
            range(len(groups)), key=lambda i: (float(vector[i]), tie_key[i], i)
        )

    return group_order


def _expand_group_order(
    groups: Sequence[Tuple[int, ...]],
    group_order: Sequence[int],
    current_order: Optional[Sequence[int]],
) -> List[int]:
    old_pos = ({logical: pos for pos, logical in enumerate(current_order)}
               if current_order is not None else {})
    result: List[int] = []
    for group_index in group_order:
        group = list(groups[group_index])
        # Preserve the previous relative order inside a fused pair whenever
        # possible; this avoids paying an unnecessary within-pair transposition.
        group.sort(key=lambda q: (old_pos.get(q, q), q))
        result.extend(group)
    return result


def _choose_spectral_order(
    nlogical: int,
    remaining_twoq: Sequence[Tuple[int, int, int]],
    current_order: Optional[Sequence[int]],
    alpha: float,
    beta: float,
    tau: int,
) -> List[int]:
    """Choose the next logical ordering using spectral drawing + forced coupling."""
    if not remaining_twoq:
        return list(range(nlogical)) if current_order is None else list(current_order)

    graph, forward, reverse, total_layers, front = _interaction_graph(
        nlogical, remaining_twoq, current_order, alpha, beta, tau
    )

    singleton_groups = [(q,) for q in range(nlogical)]
    order_ids = _fiedler_order(graph, singleton_groups, current_order)
    candidate = _expand_group_order(singleton_groups, order_ids, current_order)

    def front_compliant(order: Sequence[int]) -> bool:
        pos = {logical: i for i, logical in enumerate(order)}
        for idx in front:
            q0, q1 = next((a, b) for gate_idx, a, b in remaining_twoq if gate_idx == idx)
            if abs(pos[q0] - pos[q1]) == 1:
                return True
        return False

    # Lin et al. choose the reflection (y or -y) with lower displacement from
    # the previous mapping because an LNN path is reflection-symmetric.
    if current_order is not None:
        reverse_candidate = list(reversed(candidate))
        old_pos = {logical: pos for pos, logical in enumerate(current_order)}
        def displacement(order: Sequence[int]) -> int:
            return sum(abs(old_pos[q] - pos) for pos, q in enumerate(order))
        if displacement(reverse_candidate) < displacement(candidate):
            candidate = reverse_candidate

    if front_compliant(candidate):
        return candidate

    # Forced coupling fallback: fuse the highest-priority front-layer pairs
    # (those with the largest reverse-layer value), spectral-order the fused
    # graph, then expand each fused pair into adjacent path positions.
    if not front:
        return candidate

    max_reverse = max(reverse[idx] for idx in front)
    forced_pairs = []
    for idx, q0, q1 in remaining_twoq:
        if idx in front and reverse[idx] == max_reverse:
            forced_pairs.append((q0, q1))

    # Front-layer gates are pairwise disjoint by construction. Guard against
    # duplicate pairs just in case a nonstandard gate stream is supplied.
    used = set()
    groups: List[Tuple[int, ...]] = []
    for q0, q1 in forced_pairs:
        if q0 not in used and q1 not in used:
            groups.append((q0, q1))
            used.add(q0)
            used.add(q1)
    groups.extend((q,) for q in range(nlogical) if q not in used)

    # The paper's selected configuration uses a wider look-ahead for the
    # forced-pair calculation (tau = 4M), while the ordinary mapper uses tau = M.
    forced_tau = max(int(tau), 4 * nlogical)
    forced_graph, _, _, _, _ = _interaction_graph(
        nlogical, remaining_twoq, current_order, alpha, beta, forced_tau
    )
    group_order = _fiedler_order(forced_graph, groups, current_order)
    forced_candidate = _expand_group_order(groups, group_order, current_order)

    # Try the reflected fused ordering as well. Keep each forced pair intact.
    reflected = _expand_group_order(groups, list(reversed(group_order)), current_order)
    if current_order is not None:
        old_pos = {logical: pos for pos, logical in enumerate(current_order)}
        d0 = sum(abs(old_pos[q] - p) for p, q in enumerate(forced_candidate))
        d1 = sum(abs(old_pos[q] - p) for p, q in enumerate(reflected))
        if d1 < d0:
            forced_candidate = reflected

    if not front_compliant(forced_candidate):
        # Defensive fallback for unusual/degenerate interaction graphs.
        # Force the first front-layer pair into adjacent positions.
        q0, q1 = next((a, b) for idx, a, b in remaining_twoq if idx in front)
        rest = [q for q in range(nlogical) if q not in (q0, q1)]
        forced_candidate = [q0, q1] + rest
        if current_order is not None:
            old_pos = {logical: pos for pos, logical in enumerate(current_order)}
            reversed_candidate = list(reversed(forced_candidate))
            d0 = sum(abs(old_pos[q] - p) for p, q in enumerate(forced_candidate))
            d1 = sum(abs(old_pos[q] - p) for p, q in enumerate(reversed_candidate))
            if d1 < d0:
                forced_candidate = reversed_candidate

    return forced_candidate


def _adjacent_token_swaps(current_order: Sequence[int],
                          target_order: Sequence[int]) -> List[Tuple[int, int]]:
    """Minimum adjacent swaps on a line to transform one token order to another."""
    if sorted(current_order) != sorted(target_order):
        raise ValueError("Current and target mappings must be permutations of the same tokens.")
    tokens = list(current_order)
    swaps: List[Tuple[int, int]] = []
    for target_pos, wanted in enumerate(target_order):
        source_pos = tokens.index(wanted)
        while source_pos > target_pos:
            left = source_pos - 1
            swaps.append((left, source_pos))
            tokens[left], tokens[source_pos] = tokens[source_pos], tokens[left]
            source_pos -= 1
    return swaps


def _build_schedule(
    nlogical: int,
    op_qargs: Sequence[Tuple[int, ...]],
    op_cargs: Sequence[Tuple[int, ...]],
    op_kinds: Sequence[str],
    path: Sequence[int],
    alpha: float,
    beta: float,
    tau: Optional[int],
) -> _Schedule:
    """Create a mapping/routing schedule for a circuit's 0/1/2Q instructions."""
    if len(path) != nlogical:
        raise ValueError("The selected hardware path must contain exactly nlogical nodes.")
    if len(op_qargs) != len(op_kinds) or len(op_qargs) != len(op_cargs):
        raise ValueError("op_qargs, op_cargs, and op_kinds must have the same length.")

    # Circuit dependency DAG: operations sharing a quantum or classical wire
    # retain their original order. Disjoint operations may commute, as in the
    # paper. Classical-wire dependencies matter for repeated writes/reads of
    # measurement bits.
    predecessor_count = [0] * len(op_qargs)
    successors: List[List[int]] = [[] for _ in op_qargs]
    last_on_resource: Dict[Tuple[str, int], int] = {}
    for idx, (qargs, cargs) in enumerate(zip(op_qargs, op_cargs)):
        blockers = set()
        for q in qargs:
            if q < 0 or q >= nlogical:
                raise ValueError(f"Instruction {idx} references invalid logical qubit {q}.")
            resource = ("q", q)
            if resource in last_on_resource:
                blockers.add(last_on_resource[resource])
        for c in cargs:
            resource = ("c", c)
            if resource in last_on_resource:
                blockers.add(last_on_resource[resource])
        predecessor_count[idx] = len(blockers)
        for blocker in blockers:
            successors[blocker].append(idx)
        for q in qargs:
            last_on_resource[("q", q)] = idx
        for c in cargs:
            last_on_resource[("c", c)] = idx

    ready = {i for i, count in enumerate(predecessor_count) if count == 0}
    completed = set()
    events: List[Tuple[Any, ...]] = []
    swap_positions: List[Tuple[int, int]] = []
    mapping_orders: List[List[int]] = []
    current_order: Optional[List[int]] = None
    tau_value = int(tau if tau is not None else nlogical)

    twoq_ids_all = [
        i for i, (qargs, kind) in enumerate(zip(op_qargs, op_kinds))
        if len(qargs) == 2 and kind != "barrier"
    ]

    while len(completed) < len(op_qargs):
        remaining_twoq = [
            (idx, op_qargs[idx][0], op_qargs[idx][1])
            for idx in twoq_ids_all if idx not in completed
        ]

        # The first mapping is selected before any logical operation is issued.
        if current_order is None:
            current_order = _choose_spectral_order(
                nlogical, remaining_twoq, None, alpha, beta, tau_value
            ) if remaining_twoq else list(range(nlogical))
            mapping_orders.append(list(current_order))

        position = {logical: pos for pos, logical in enumerate(current_order)}
        executable = []
        for idx in sorted(ready):
            qargs = op_qargs[idx]
            kind = op_kinds[idx]
            if kind == "barrier" or len(qargs) <= 1:
                executable.append(idx)
            elif len(qargs) == 2:
                q0, q1 = qargs
                if abs(position[q0] - position[q1]) == 1:
                    executable.append(idx)
            else:
                raise NotImplementedError(
                    f"Instruction {idx} acts on {len(qargs)} qubits. This implementation "
                    "supports one- and two-qubit operations (plus barriers). Decompose "
                    "multi-controlled/multi-qubit gates before routing."
                )

        if executable:
            # The ready set is pairwise non-overlapping on quantum wires, so
            # emitting these operations in index order is semantics-preserving.
            for idx in executable:
                qargs = op_qargs[idx]
                positions = tuple(position[q] for q in qargs)
                events.append(("gate", idx, positions))
                ready.remove(idx)
                completed.add(idx)
                for child in successors[idx]:
                    predecessor_count[child] -= 1
                    if predecessor_count[child] == 0:
                        ready.add(child)
            continue

        if not ready:
            raise RuntimeError("Circuit dependency scheduler stalled; check the instruction DAG.")
        if not remaining_twoq:
            raise RuntimeError("Scheduler stalled although no two-qubit operations remain.")

        # No currently-ready 2Q operation is adjacent. Generate the next
        # spectral permutation and bridge it with exact minimum LNN token swaps.
        target_order = _choose_spectral_order(
            nlogical, remaining_twoq, current_order, alpha, beta, tau_value
        )
        if target_order == current_order:
            raise RuntimeError(
                "The spectral mapper failed to produce a different mapping even though "
                "no ready two-qubit gate is adjacent."
            )

        transitions = _adjacent_token_swaps(current_order, target_order)
        for left, right in transitions:
            events.append(("swap", left, right))
            swap_positions.append((left, right))
        current_order = list(target_order)
        mapping_orders.append(list(current_order))

    if current_order is None:
        current_order = list(range(nlogical))
    return _Schedule(
        events=events,
        initial_order=mapping_orders[0] if mapping_orders else list(range(nlogical)),
        final_order=list(current_order),
        swap_positions=swap_positions,
        mapping_orders=mapping_orders,
    )


def _run_one_configuration(
    circuit: Any,
    path: Sequence[int],
    alpha: float,
    beta: float,
    tau: Optional[int],
) -> Tuple[_Schedule, List[Tuple[int, ...]], List[str]]:
    nlogical = int(circuit.num_qubits)
    op_qargs: List[Tuple[int, ...]] = []
    op_cargs: List[Tuple[int, ...]] = []
    op_kinds: List[str] = []

    for idx, instruction in enumerate(circuit.data):
        operation = instruction.operation
        name = str(getattr(operation, "name", "")).lower()
        qargs = tuple(circuit.find_bit(q).index for q in instruction.qubits)
        cargs = tuple(circuit.find_bit(c).index for c in instruction.clbits)
        op_qargs.append(qargs)
        op_cargs.append(cargs)
        op_kinds.append("barrier" if name == "barrier" else name)

        condition = getattr(operation, "condition", None)
        if condition is not None:
            raise NotImplementedError(
                "Classically conditioned operations are not supported by this mapper yet."
            )
        if len(qargs) > 2 and name != "barrier":
            raise NotImplementedError(
                f"Instruction '{name}' acts on {len(qargs)} qubits. Decompose it into "
                "one- and two-qubit gates before calling this mapper."
            )

    schedule = _build_schedule(
        nlogical=nlogical,
        op_qargs=op_qargs,
        op_cargs=op_cargs,
        op_kinds=op_kinds,
        path=path,
        alpha=alpha,
        beta=beta,
        tau=tau,
    )
    return schedule, op_qargs, op_kinds


def route_circuit_on_heavyhex_path(
    circuit: Any,
    backend: Any,
    *,
    path: Optional[Sequence[int]] = None,
    alpha: float = 0.8,
    beta: float = 0.2,
    tau: Optional[int] = None,
    paper_meta: bool = False,
    max_path_expansions: int = 750_000,
) -> SpectralMappingResult:
    """Route a Qiskit circuit using Lin et al.'s spectral mapper on a heavy-hex path.

    Parameters
    ----------
    circuit:
        Qiskit QuantumCircuit. One- and two-qubit operations are supported,
        including measurements and barriers. Multi-qubit gates must be
        decomposed first.
    backend:
        Qiskit backend with a coupling map / target.
    path:
        Optional explicit ordered list of physical qubits forming a simple path.
        If omitted, a path of length circuit.num_qubits is extracted from the
        backend coupling graph, preferring edges with lower reported gate error.
    alpha, beta, tau:
        Paper parameters. tau defaults to the number of logical qubits for the
        ordinary mapper; the forced-pair fallback uses max(tau, 4M), matching
        the paper's selected experimental setting.
    paper_meta:
        If True, run the ten (alpha, beta) configurations selected by the paper
        and keep the result with the fewest inserted SWAPs. Otherwise use the
        supplied alpha and beta (defaults 0.8 and 0.2).
    max_path_expansions:
        DFS expansion cap for automatic path extraction.

    Notes
    -----
    The returned circuit has one wire per physical qubit of the backend. The
    logical input is placed according to ``initial_layout``; measurements are
    redirected so their classical-bit results retain their logical meaning.
    The final logical-to-physical correspondence is ``final_layout``.
    """
    try:
        from qiskit import QuantumCircuit
    except Exception as exc:
        raise ImportError(
            "Qiskit is required for route_circuit_on_heavyhex_path(). "
            "Install qiskit in the environment where you run the module."
        ) from exc

    nlogical = int(circuit.num_qubits)
    nphysical = _backend_num_qubits(backend)
    if nlogical > nphysical:
        raise ValueError(
            f"Circuit width ({nlogical}) exceeds backend width ({nphysical})."
        )

    hardware_graph = coupling_graph_from_backend(backend)
    if path is None:
        selected_path = extract_hardware_path(
            backend, nlogical, coupling_graph=hardware_graph,
            max_expansions=max_path_expansions,
        )
    else:
        selected_path = [int(q) for q in path]
        if len(selected_path) != nlogical:
            raise ValueError(
                f"Explicit path has {len(selected_path)} nodes, but circuit has "
                f"{nlogical} logical qubits."
            )
        if len(set(selected_path)) != len(selected_path):
            raise ValueError("The explicit path contains duplicate physical qubits.")
        for q in selected_path:
            if q not in hardware_graph:
                raise ValueError(f"Physical qubit {q} is not present in the backend graph.")
        for u, v in zip(selected_path, selected_path[1:]):
            if not hardware_graph.has_edge(u, v):
                raise ValueError(
                    f"Explicit path is not a hardware path: ({u}, {v}) is not a coupling edge."
                )

    configurations = PAPER_META_CONFIGS if paper_meta else ((float(alpha), float(beta)),)
    if any(not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0) for a, b in configurations):
        raise ValueError("alpha and beta must lie in [0, 1].")

    candidates = []
    for a, b in configurations:
        schedule, op_qargs, op_kinds = _run_one_configuration(
            circuit, selected_path, a, b, tau
        )
        candidates.append((len(schedule.swap_positions), a, b, schedule, op_qargs, op_kinds))

    # The paper's meta-algorithm selects the configuration with minimum added
    # SWAP count. The alpha/beta tuple provides deterministic tie-breaking.
    swap_count, chosen_alpha, chosen_beta, schedule, op_qargs, op_kinds = min(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    chosen_tau = int(tau if tau is not None else nlogical)

    # Materialize the abstract schedule as a physical-width Qiskit circuit.
    physical = QuantumCircuit(nphysical, circuit.num_clbits, name=f"{circuit.name}_spectral")
    try:
        physical.global_phase = circuit.global_phase
    except Exception:
        pass
    if getattr(circuit, "metadata", None) is not None:
        physical.metadata = dict(circuit.metadata)

    swap_sequence: List[Tuple[int, int]] = []
    for event in schedule.events:
        if event[0] == "swap":
            _, left_pos, right_pos = event
            qleft, qright = selected_path[left_pos], selected_path[right_pos]
            physical.swap(qleft, qright)
            swap_sequence.append((qleft, qright))
        elif event[0] == "gate":
            _, instruction_index, positions = event
            instruction = circuit.data[instruction_index]
            operation = instruction.operation
            physical_qargs = [physical.qubits[selected_path[pos]] for pos in positions]
            physical_cargs = [
                physical.clbits[circuit.find_bit(cbit).index]
                for cbit in instruction.clbits
            ]
            try:
                operation = operation.copy()
            except Exception:
                pass
            physical.append(operation, physical_qargs, physical_cargs)
        else:
            raise RuntimeError(f"Unknown scheduling event: {event[0]}")

    initial_layout = {
        logical: selected_path[pos]
        for pos, logical in enumerate(schedule.initial_order)
    }
    final_layout = {
        logical: selected_path[pos]
        for pos, logical in enumerate(schedule.final_order)
    }
    mapping_history = [
        {
            logical: selected_path[pos]
            for pos, logical in enumerate(order)
        }
        for order in schedule.mapping_orders
    ]

    return SpectralMappingResult(
        circuit=physical,
        path=list(selected_path),
        initial_layout=initial_layout,
        final_layout=final_layout,
        swap_count=swap_count,
        swap_sequence=swap_sequence,
        mapping_history=mapping_history,
        alpha=chosen_alpha,
        beta=chosen_beta,
        tau=chosen_tau,
    )


def transpile_mapped_result(
    result: SpectralMappingResult,
    backend: Any,
    *,
    optimization_level: int = 1,
    seed_transpiler: int = 0,
) -> Any:
    """Translate/optimize the routed circuit without invoking another router.

    The spectral mapper has already inserted connectivity-compliant SWAPs.
    ``routing_method='none'`` is intentional: silently running SABRE here would
    obscure whether the spectral method itself produced a compliant circuit.
    """
    try:
        from qiskit import transpile
    except Exception as exc:
        raise ImportError("Qiskit is required for transpile_mapped_result().") from exc

    return transpile(
        result.circuit,
        backend=backend,
        initial_layout=list(range(result.circuit.num_qubits)),
        routing_method="none",
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )


# ---------------------------------------------------------------------------
# Small pure-Python checks for the path mapper core (no Qiskit required).
# ---------------------------------------------------------------------------

def validate_path(path: Sequence[int], coupling_graph: nx.Graph) -> bool:
    """Return True iff path is simple and every consecutive pair is coupled."""
    return (
        len(set(path)) == len(path)
        and all(coupling_graph.has_edge(u, v) for u, v in zip(path, path[1:]))
    )


def build_path_subgraph(path: Sequence[int], coupling_graph: nx.Graph) -> nx.Graph:
    """Build the strict 1D path graph induced by an ordered hardware path.

    Only consecutive path edges are retained; extra heavy-hex couplers between
    nonconsecutive selected vertices are deliberately discarded by the LNN
    model. Edge/node attributes from the hardware graph are copied.
    """
    if not validate_path(path, coupling_graph):
        raise ValueError("The supplied sequence is not a valid simple path in the coupling graph.")
    result = nx.Graph()
    for q in path:
        result.add_node(q, **dict(coupling_graph.nodes[q]))
    for u, v in zip(path, path[1:]):
        result.add_edge(u, v, **dict(coupling_graph[u][v]))
    return result

