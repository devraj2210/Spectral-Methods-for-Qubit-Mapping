import numpy as np
import random
import networkx as nx

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RZZGate
from collections import deque
from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import PassManager, Layout
from qiskit.transpiler.passes import (
    SetLayout,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    ApplyLayout,
    SabreSwap,
    SabreLayout
)

from scipy.sparse.csgraph import laplacian
from scipy.linalg import eig, eigh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def dfs_spanning_tree(G, root=0):
    """
    Constructs a DFS spanning tree of G rooted at `root`.

    Parameters:
        G : networkx.Graph
            Input undirected graph.
        root : int
            Root node to start DFS.

    Returns:
        T : networkx.DiGraph
            DFS tree (no dummy nodes, no degree bounds).
    """
    T = nx.Graph()
    visited = set()

    def dfs(current):
        visited.add(current)
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                T.add_edge(current, neighbor)
                dfs(neighbor)

    dfs(root)
    return T

def build_tree_cluster(G, source, n):
    """
    Grow a connected cluster (tree) from `source` by BFS until it has `n` nodes.
    Returns the set of nodes in the cluster and the tree edges.
    """
    visited = set([source])
    tree_edges = []
    queue = deque([source])

    while len(visited) < n and queue:
        current = queue.popleft()
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                tree_edges.append((current, neighbor))
                queue.append(neighbor)
                if len(visited) == n:
                    break

    return G.subgraph(visited)

def get_coupling_graph(backend, weighted=False):
    edges = backend.configuration().coupling_map
    G = nx.Graph()
    if weighted:

        props = backend.properties()
        conf = backend.configuration()
        coupling = conf.coupling_map

        alpha=0.75
        beta_readout=0.7
        beta_t12=0.2
        beta_gate=0.1

        G = nx.Graph()

        # ---- Node (1Q) noise ----
        node_noise = {}
        for q in range(conf.num_qubits):
            qb = props.qubits[q]
            for p in qb:
                if p.name=='T1':
                    if p.value>0:
                        t1 = p.value
                    else:
                        t1 = 100e6
                if p.name=='T2':
                    if p.value>0:
                        t2 = p.value
                    else:
                        t2 = 100e6
                if p.name=='readout_error':
                    readout_err = p.value
            decoh = (1 / t1 + 1 / t2) if t1 > 0 and t2 > 0 else 0
            gate_errs = [
                param.value
                for g in props.gates
                if len(g.qubits) == 1 and g.qubits[0] == q
                for param in g.parameters
                if param.name == "gate_error"
            ]
            # print(gate_errs)
            max_gate_err = np.mean(gate_errs) if gate_errs else 0
            node_noise[q] = (
                beta_readout * readout_err
                + beta_t12 * decoh
                + beta_gate * max_gate_err
            )
            # print(q, node_noise[q])
            G.add_node(q, weight=max(1 - node_noise[q], 0))
            # print(q, decoh, max_gate_err, readout_err, 1-node_noise[q])

        # ---- Edge (2Q) noise ----
        for g in props.gates:
            if len(g.qubits) == 2:
                q1, q2 = g.qubits
                err = next(param.value for param in g.parameters if param.name == "gate_error")
                node_term = (node_noise[q1] + node_noise[q2]) / 2
                # print(node_term)
                combined_noise = alpha * err + (1 - alpha) * node_term
                reliability = 1 - combined_noise
                G.add_edge(q1, q2, weight=max(reliability, 0))
    else:
        G.add_edges_from(edges)

    # ---- Identify Source Node with Highest Average Edge Reliability ----
    best_node = None
    max_avg_reliability = -1

    for node in G.nodes():
        degree = G.degree(node)
        if degree > 0:
            # Sum the weights of all edges connected to the node. 
            # If the graph is unweighted, it defaults the edge weight to 1.0.
            total_reliability = sum(weight for _, _, weight in G.edges(node, data='weight', default=1.0))
            avg_reliability = total_reliability / degree
            
            if avg_reliability > max_avg_reliability:
                max_avg_reliability = avg_reliability
                best_node = node
        elif best_node is None:
            # Fallback for an isolated node, if any exist in the topology
            best_node = node 

    return G, best_node
    # return G

# import networkx as nx
# import numpy as np

# def get_coupling_graph(backend, weighted=False):
#     edges = backend.configuration().coupling_map
#     G = nx.Graph()
    
#     if weighted:
#         props = backend.properties()
#         conf = backend.configuration()

#         alpha = 0.75
#         beta_readout = 0.7
#         beta_t12 = 0.2
#         beta_gate = 0.1

#         # ---- Node (1Q) noise ----
#         node_noise = {}
#         for q in range(conf.num_qubits):
#             qb = props.qubits[q]
#             t1 = 100e6
#             t2 = 100e6
#             readout_err = 0
            
#             for p in qb:
#                 if p.name == 'T1' and p.value > 0:
#                     t1 = p.value
#                 elif p.name == 'T2' and p.value > 0:
#                     t2 = p.value
#                 elif p.name == 'readout_error':
#                     readout_err = p.value
                    
#             decoh = (1 / t1 + 1 / t2) if t1 > 0 and t2 > 0 else 0
#             gate_errs = [
#                 param.value
#                 for g in props.gates
#                 if len(g.qubits) == 1 and g.qubits[0] == q
#                 for param in g.parameters
#                 if param.name == "gate_error"
#             ]
            
#             max_gate_err = np.mean(gate_errs) if gate_errs else 0
#             node_noise[q] = (
#                 beta_readout * readout_err
#                 + beta_t12 * decoh
#                 + beta_gate * max_gate_err
#             )
            
#             G.add_node(q, weight=max(1 - node_noise[q], 0))

#         # ---- Edge (2Q) noise ----
#         for g in props.gates:
#             if len(g.qubits) == 2:
#                 q1, q2 = g.qubits
#                 err = next(param.value for param in g.parameters if param.name == "gate_error")
#                 node_term = (node_noise[q1] + node_noise[q2]) / 2
#                 combined_noise = alpha * err + (1 - alpha) * node_term
#                 reliability = 1 - combined_noise
                
#                 G.add_edge(q1, q2, weight=max(reliability, 0))
#     else:
#         G.add_edges_from(edges)

#     # ---- Identify Source Node with Highest Average Edge Reliability ----
#     best_node = None
#     max_avg_reliability = -1

#     for node in G.nodes():
#         degree = G.degree(node)
#         if degree > 0:
#             # Sum the weights of all edges connected to the node. 
#             # If the graph is unweighted, it defaults the edge weight to 1.0.
#             total_reliability = sum(weight for _, _, weight in G.edges(node, data='weight', default=1.0))
#             avg_reliability = total_reliability / degree
            
#             if avg_reliability > max_avg_reliability:
#                 max_avg_reliability = avg_reliability
#                 best_node = node
#         elif best_node is None:
#             # Fallback for an isolated node, if any exist in the topology
#             best_node = node 

#     return G, best_node

def find_linear_chain(backend, target_length):
    '''For small target_length, find a linear chain of that length in the coupling graph.'''
    coupling_map = backend.coupling_map

    # adjacency as a dict: {qubit: [connected_qubits]}
    adj = {i: [] for i in range(backend.configuration().num_qubits)}
    for (i, j) in coupling_map:
        adj[i].append(j)
        adj[j].append(i)

    def dfs(node, visited, path):
        if len(path) == target_length:
            return path
        for neighbor in adj[node]:
            if neighbor not in visited:
                result = dfs(neighbor, visited | {neighbor}, path + [neighbor])
                if result:
                    return result
        return None

    for start in adj:
        result = dfs(start, {start}, [start])
        if result:
            return result
    return None

def graph_fiedler_sort(graph):
    '''Sort nodes by their Fiedler vector values.'''
    adj_matrix = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes())).todense()
    L = laplacian(adj_matrix)
    vals, vecs = eigh(L)
    fiedler = vecs[:,1]
    sorted_nodes = np.argsort(fiedler)
    return sorted_nodes

def build_bfs_tree(G, source):
    """
    Build a BFS tree from a graph starting at the given source node.
    
    Parameters:
    - G: networkx.Graph or networkx.DiGraph
    - source: node in G from which BFS starts

    Returns:
    - T: networkx.DiGraph representing the BFS tree
    """
    bfs_edges = list(nx.bfs_edges(G, source))
    
    T = nx.Graph()  # BFS tree is rooted and directed
    T.add_nodes_from(G.nodes())
    T.add_edges_from(bfs_edges)

    return T

def graph_laplacian_embedding(G, k=2, alpha=0.95, normalized=False):
    '''Compute a spectral embedding of the graph (PageRank-like approach for directed graphs.)'''
    # Step 0: adjacency matrix
    if isinstance(G, nx.DiGraph):
        W = nx.to_numpy_array(G, nodelist=sorted(G.nodes()), weight=None)
        n = W.shape[0]
        e = np.ones((n, 1))

        # Out-degree diagonal matrix
        out_degrees = W.sum(axis=1)
        D0 = np.diag(out_degrees)

        # Step 1: Transition probability matrix P with perturbation
        mu = np.where(out_degrees == 0, 1, 0).reshape(-1, 1)  # dangling nodes
        D0_inv = np.linalg.pinv(D0)  # use pseudo-inverse to handle zeros
        P = alpha * (D0_inv @ W + (1/n) * mu @ e.T) + (1 - alpha) * (1/n) * e @ e.T

        # Step 2: Solve eigenproblem π^T P = π^T, i.e., find stationary distribution
        # Equivalent to solving P^T π = π
        eigvals, eigvecs = eig(P.T)
        idx = np.argmin(np.abs(eigvals - 1))  # eigenvalue closest to 1
        pi = np.real(eigvecs[:, idx])
        pi = pi / pi.sum()  # normalize so sum=1

        # Step 3: Construct Laplacian
        Phi = np.diag(pi)
        L = Phi - (Phi @ P + P.T @ Phi) / 2

        # Step 4: Generalized eigenvalue problem L v = λ Φ v
        eigvals, eigvecs = eig(np.linalg.pinv(Phi) @ L)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

        # Sort eigenvectors by eigenvalues
        idx = np.argsort(eigvals)
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        # Skip the first trivial eigenvector (corresponding to eigenvalue ~0)
        Y = eigvecs[:, 1:k+1]
        # Y=[]
        # for i in range(n):
        #     Y.append([np.mean(eigvecs[i,1:k+1]), np.std(eigvecs[i,1:k+1])])
        # Y = np.array(Y)
        return Y
    elif isinstance(G, nx.Graph):
        if normalized:
            A = adj_matrix = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).todense()
            degrees = np.sum(A, axis=1)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-12)))

            # --- Normalized Laplacian ---
            L_sym = np.eye(G.number_of_nodes()) - D_inv_sqrt @ A @ D_inv_sqrt

            # --- Compute eigenvalues and eigenvectors ---
            vals, vecs = eigh(L_sym)
            # Sort eigenpairs
            idx = np.argsort(vals)
            vals, vecs = vals[idx], vecs[:, idx]

            # Skip the first eigenvector (constant)
            X = vecs[:, 1:k+1]
        else:
            adj_matrix = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).todense()
            L = laplacian(adj_matrix)
            vals, vecs = eigh(L)
            X = vecs[:, 1:k+1]  # skip the first (trivial) eigenvector
        return X
    else:
        return None
    
def central_node(G):
    # Compute eccentricity of each node
    ecc = nx.eccentricity(G)
    
    # Find center nodes (min eccentricity)
    min_ecc = min(ecc.values())
    center_nodes = [n for n, e in ecc.items() if e == min_ecc]
    
    # If multiple centers exist, break ties using closeness centrality
    if len(center_nodes) > 1:
        closeness = nx.closeness_centrality(G)
        center_nodes.sort(key=lambda n: closeness[n], reverse=True)
    
    return center_nodes[0]  # Best root

def selective_bfs(G, source, n, weight='weight', threshold=0.9, relax_factor=0.5, min_threshold=0.0):
    '''Perform a BFS that only includes neighbors with edge weight above a certain threshold.'''
    if source not in G:
        raise ValueError(f"Source node {source} not in graph")

    visited = set([source])
    q = deque([source])
    sub_nodes = [source]
    current_threshold = threshold

    while len(sub_nodes) < n:
        progress = False
        temp_q = deque(q)

        while temp_q and len(sub_nodes) < n:
            u = temp_q.popleft()
            for v in G.neighbors(u):
                if v not in visited:
                    w = G[u][v].get(weight, 1.0)
                    if w >= current_threshold:
                        visited.add(v)
                        q.append(v)
                        sub_nodes.append(v)
                        progress = True
                        if len(sub_nodes) >= n:
                            break

        # If no new nodes were added, relax the threshold
        if not progress:
            current_threshold *= relax_factor
            if current_threshold < min_threshold:
                # As a last resort, include any remaining nodes
                remaining = [v for v in G.nodes if v not in visited]
                sub_nodes.extend(remaining[: n - len(sub_nodes)])
                break

    return G.subgraph(sub_nodes).copy()

def initial_layout(graph, backend, how='fiedler', topology='tree_dfs', root_on_backend=None, noise_aware=False):
    '''Compute an initial layout mapping logical qubits to physical qubits.'''
    num_nodes = graph.number_of_nodes()
    n = int(np.sqrt(num_nodes))
    if how=='fiedler':
        sorted_nodes = graph_fiedler_sort(graph)

        if topology=='chain':
            chain = find_linear_chain(backend, num_nodes)
        else:
            if noise_aware:
                G, best_node = get_coupling_graph(backend, weighted=True)
                root_on_backend = best_node
            else:
                G = get_coupling_graph(backend)
                root_on_backend = central_node(G)
            # if not root_on_backend:

            if noise_aware:
                tree_layout = selective_bfs(G, source=root_on_backend, n=num_nodes)
            else:
                tree_layout = build_tree_cluster(G, source=root_on_backend, n=num_nodes)

            order = list(nx.dfs_preorder_nodes(tree_layout, source=root_on_backend))

    if how not in ['trivial', 'random']:
        if topology=='chain':
            logical_to_physical = {sorted_nodes[i]: chain[i] for i in range(num_nodes)}
        else:
            logical_to_physical = {sorted_nodes[i]: order[i] for i in range(num_nodes)}
    elif how=='trivial':
        logical_to_physical = {i: i for i in range(num_nodes)}
    elif how=='random':
        random_map = list(range(num_nodes))
        random.seed(0)
        random.shuffle(random_map)
        logical_to_physical = {i: random_map[i] for i in range(num_nodes)}
    else:
        print("Invalid starting layout!!!")
        return -1

    return logical_to_physical

def sabre_layout(circuit:QuantumCircuit, backend):
    '''Compute an initial layout using SABRE's layout pass, without routing.'''
    # circuit = transpile(circuit, basis_gates=backend.basis_gates, seed_transpiler=0)
    pm = PassManager()
    pm.append(SabreLayout(backend.coupling_map, skip_routing=True, seed=0))
    pm.run(circuit)

    layout = pm.property_set["layout"]
    mapping_dict = {i: layout[q] for i, q in enumerate(circuit.qubits)}
    return mapping_dict

def build_max_cut_paulis(graph: nx.Graph) -> list[tuple[str, float]]:
    """Convert NetworkX graph to Pauli list for MaxCut."""
    pauli_list = []
    num_nodes = len(graph.nodes())
    
    for edge in graph.edges(data=True):
        u, v, data = edge
        paulis = ["I"] * num_nodes
        
        # NetworkX nodes can be any hashable type, so we use positional indices
        paulis[u] = "Z"
        paulis[v] = "Z"
        
        # Get weight with default 1.0 if not specified
        weight = data.get('weight', 1.0)
        
        # Reverse string to match RustworkX qubit ordering convention
        pauli_list.append(("".join(paulis)[::-1], float(weight)))
    
    return pauli_list

def get_hamiltonian(graph):
    max_cut_paulis = build_max_cut_paulis(graph)
    cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
    return cost_hamiltonian

def sabre_routed_circuit(circuit:QuantumCircuit, backend, swap_heuristic='lookahead', opt_level=3):
    '''Transpile the circuit using a custom SABRE-based routing pipeline.'''
    coupling_map = backend.coupling_map

    pm = PassManager([
        # SetLayout(initial_layout),
        SabreLayout(coupling_map, skip_routing=True),
        EnlargeWithAncilla(),
        FullAncillaAllocation(coupling_map),
        ApplyLayout(),
        SabreSwap(coupling_map, heuristic=swap_heuristic)
    ])

    # Transpile using the custom pipeline
    transpiled = pm.run(circuit)
    return transpiled

def gen_QAOA_circuit(graph:nx.Graph, reps=2):
    cost_hamiltonian = get_hamiltonian(graph)
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    circuit.measure_all()
    return circuit

def custom_routed_circuit(circuit : QuantumCircuit, initial_mapping, backend, swap_heuristic='lookahead', opt_level=3):
    '''Transpile the circuit using a custom SABRE-based routing pipeline with a specified initial layout.'''
    basis_gates = backend.basis_gates
    coupling_map = backend.coupling_map

    mapping_layout = Layout({q: initial_mapping[i] for i,q in enumerate(circuit.qubits)})

    pm = PassManager([
        SetLayout(mapping_layout),
        EnlargeWithAncilla(),
        FullAncillaAllocation(coupling_map),
        ApplyLayout(),
        SabreSwap(coupling_map, heuristic=swap_heuristic),
    ])

    transpiled = pm.run(circuit)

    return transpiled

def final_optimized_circuit(circuit, backend, initial_mapping=None, opt_level=3):
    '''Run the final optimization pass manager with or without an initial layout.'''
    if initial_mapping:
        mapping_layout = Layout({q: initial_mapping[i] for i,q in enumerate(circuit.qubits)})
        pm = generate_preset_pass_manager(optimization_level=opt_level, initial_layout=mapping_layout, backend=backend)
        candidate_circuit = pm.run(circuit)
    else:
        pm = generate_preset_pass_manager(optimization_level=opt_level, backend=backend)
        candidate_circuit = pm.run(circuit)
    return candidate_circuit

def optimal_qubit_assignment(logical_embedding, physical_embedding, nodes=None):
    """
    Find optimal mapping between logical and physical qubits
    using Hungarian algorithm on Euclidean distances.

    Args:
        logical_embedding: np.ndarray, shape (n, k)
        physical_embedding: np.ndarray, shape (n, k)

    Returns:
        mapping: dict {logical_index: physical_index}
    """
    if not nodes:
        nodes = list(range(physical_embedding.shape[0]))
    # cost matrix: pairwise distances between embeddings
    cost_matrix = cdist(logical_embedding, physical_embedding, metric='sqeuclidean')

    # Hungarian (linear sum assignment)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Construct mapping
    mapping = {int(r): int(nodes[c]) for r, c in zip(row_ind, col_ind)}
    return mapping

def normalized_graph(graph:nx.Graph):
    '''Normalize edge weights by node degrees to mitigate hub dominance.'''
    for u,v in list(graph.edges):
        graph[u][v]['weight'] = 1.0/np.sqrt(graph.degree(u)*graph.degree(v))
    return graph

def fiedler_ordered_mapping(graph:nx.Graph, backend, noise_aware=False, normalized=True):
    '''Compute a layout based on Fiedler ordering and a BFS/DFS tree.'''
    num_nodes = graph.number_of_nodes()
    if noise_aware:
        G, best_node = get_coupling_graph(backend, weighted=True)
        # high_degree_nodes = [u for u in G.nodes if G.degree(u) >= 3]
        # root_on_backend = max(high_degree_nodes, key=lambda q: G.nodes[q]["weight"])
        root_on_backend = best_node
    else:
        G = get_coupling_graph(backend)
        root_on_backend = backend.number_of_qubits // 2

    if noise_aware:
        tree_layout = selective_bfs(G, source=root_on_backend, n=num_nodes)
    else:
        tree_layout = build_tree_cluster(G, source=root_on_backend, n=num_nodes)
    order = list(nx.dfs_preorder_nodes(tree_layout, source=root_on_backend))

    tree = dfs_spanning_tree(graph, root=0)
    fiedler = graph_laplacian_embedding(tree, k=1, normalized=normalized).flatten()
    sorted_nodes = np.argsort(fiedler)

    return {sorted_nodes[i]: order[i] for i in range(num_nodes)}

def spectral_assignment_mapping(graph:nx.Graph, backend, k=2, noise_aware=False, normalized=True):
    '''Compute a layout by directly matching spectral embeddings of the problem and coupling graphs.'''
    num_nodes = graph.number_of_nodes()
    if noise_aware:
        G, best_node = get_coupling_graph(backend, weighted=True)
        # high_degree_nodes = [u for u in G.nodes if G.degree(u) >= 3]
        # root_on_backend = max(high_degree_nodes, key=lambda q: G.nodes[q]["weight"])
        root_on_backend = best_node
    else:
        G = get_coupling_graph(backend)
        root_on_backend = backend.number_of_qubits // 2

    if noise_aware:
        S = selective_bfs(G, source=root_on_backend, n=num_nodes)
    else:
        S = build_tree_cluster(G, source=root_on_backend, n=num_nodes)

    logical_embedding = graph_laplacian_embedding(graph, k=k, normalized=normalized)
    physical_embedding = graph_laplacian_embedding(S, k=k)

    return optimal_qubit_assignment(logical_embedding, physical_embedding, nodes=sorted(S.nodes()))

def custom_qaoa_ansatz(cost_graph: nx.Graph, layout: dict, coupling_map: list[tuple[int, int]], reps: int) -> QuantumCircuit:
    '''Build a custom QAOA ansatz that prioritizes edges based on physical adjacency.'''
    num_qubits = len(cost_graph.nodes)
    circuit = QuantumCircuit(num_qubits)

    # Parameters
    gammas = ParameterVector('γ', reps)
    betas = ParameterVector('β', reps)

    # Hadamard initialization
    circuit.h(range(num_qubits))

    # Coupling graph for distance calculation
    coupling_graph = nx.Graph()
    coupling_graph.add_edges_from(coupling_map)

    def physical_distance(q1, q2):
        try:
            return nx.shortest_path_length(coupling_graph, q1, q2)
        except nx.NetworkXNoPath:
            return float('inf')

    # Precompute sorted edges (by physical adjacency/distance)
    edge_info = []
    for i, j in cost_graph.edges():
        p_i = layout[i]
        p_j = layout[j]
        dist = physical_distance(p_i, p_j)
        edge_info.append((i, j, dist))
    edge_info.sort(key=lambda x: x[2])

    # Build QAOA layers
    for rep in range(reps):
        # Cost Layer
        for i, j, _ in edge_info:
            circuit.append(RZZGate(2*gammas[rep]), [i, j])
        # circuit.barrier()

        # Mixer Layer (RX(2*β))
        for q in range(num_qubits):
            circuit.rx(2 * betas[rep], q)
        # circuit.barrier()
    circuit.measure_all()

    return circuit