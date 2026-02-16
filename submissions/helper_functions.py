import numpy as np
import random
import math
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler

from scipy.optimize import minimize

from implementation import (
    graph_laplacian_embedding,
    fiedler_ordered_mapping,
    spectral_assignment_mapping,
    custom_qaoa_ansatz,
    get_hamiltonian,
    gen_QAOA_circuit,
    custom_routed_circuit,
    sabre_routed_circuit,
    final_optimized_circuit
)

def barabasi_albert_graph(n, m, seed=None):
    if seed is not None:
        random.seed(seed)

    if m < 1 or m >= n:
        raise ValueError("Require 1 ≤ m < n")

    G = nx.Graph()

    # --- Seed graph ---
    if m == 1:
        G.add_node(0)
        node_list = [0]
        start = 1
    else:
        G.add_nodes_from(range(m))
        for i in range(m):
            for j in range(i + 1, m):
                G.add_edge(i, j)

        node_list = []
        for i in range(m):
            node_list.extend([i] * (m - 1))
        start = m

    # --- Preferential attachment ---
    for new_node in range(start, n):
        targets = set()
        while len(targets) < m:
            targets.add(random.choice(node_list))

        for t in targets:
            G.add_edge(new_node, t)
            node_list.append(t)
            node_list.append(new_node)

    return G

def draw_graph_custom(G, node_labels=True, edge_labels=False, layout='kamada_kawai',
                      node_size=100, font_size=6, highlight_nodes=None, title=None):
    '''Custom graph drawing function with various layout options and node highlighting.'''
    graph = nx.Graph()
    graph.add_nodes_from(G.nodes())
    graph.add_edges_from(G.edges())
    G=graph

    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    elif layout == 'grid':
        # Create a grid layout with bottom-to-top rows
        n = G.number_of_nodes()
        side = math.ceil(math.sqrt(n))
        rows = math.ceil(n / side)
        pos = {
            node: (i % side, (i // side))  # bottom-to-top layout
            for i, node in enumerate(list(range(n)))
        }
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    
    # Node colors
    if highlight_nodes is not None:
        node_colors = [
            'orange' if node in highlight_nodes else 'skyblue'
            for node in G.nodes()
        ]
    else:
        node_colors = 'skyblue'

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=node_labels, node_size=node_size,
            font_size=font_size, edge_color='gray', node_color=node_colors)

    # Draw edge labels if requested
    if edge_labels:
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def gradient_graph(graph : nx.Graph, normalized=False, tree=None, title=None, flip_sign=False):
    '''Visualize the graph with nodes colored by their Fiedler vector values.'''
    if not tree:
        tree = graph
    G = nx.Graph()
    G.add_nodes_from(sorted(graph.nodes()))
    G.add_edges_from(graph.edges())
    graph=G
    fiedler = graph_laplacian_embedding(tree, k=1, normalized=normalized).flatten()
    # sorted_nodes = np.argsort(fiedler)

    fiedler_dict = {}
    fiedler_dict = {list(graph.nodes)[i] : (1+2*flip_sign*(-1))*fiedler[i] for i in range(graph.number_of_nodes())}

    values = [fiedler_dict[node] for node in graph.nodes()]
    pos = nx.kamada_kawai_layout(graph)
    nx.draw(graph, pos, node_color=values, cmap=plt.cm.viridis, with_labels=True)
    # plt.colorbar(label="Fiedler Vector Value")
    if title:
        plt.title(title)
    plt.show()

def return_ops(backend, graph_type='reg', param=4, average_over=5, node_list=list(range(10,101,2)), reps=2, tech='1D', k=2, opt_level=3, op_name='ecr', stop_after_routing=False, normalize_laplacian=True, noise_aware=False):
    swaps = {'means': {'SABRE':[], 'Proposed':[], 'Trivial':[], 'Random':[]},
             'stddevs': {'SABRE':[], 'Proposed':[], 'Trivial':[], 'Random':[]}}
    for n in tqdm(node_list, desc=f'Iterating for number of nodes', leave=False):
        swap_avg = {'sabre':[], 'fied':[], 'triv':[], 'rand':[]}

        # choose graph type
        if graph_type=='reg':
            graph = nx.random_regular_graph(n=n, d=param, seed=0)
        else:
            graph = barabasi_albert_graph(n=n, m=param, seed=0)

        # choose proposed mapping technique
        if tech=='1D':
            proposed_mapping = fiedler_ordered_mapping(graph, backend, noise_aware=noise_aware, normalized=normalize_laplacian)
        elif tech=='2D':
            proposed_mapping = spectral_assignment_mapping(graph, backend, k=k, noise_aware=noise_aware, normalized=normalize_laplacian)

        gen = gen_QAOA_circuit(graph, reps=reps) # for sabre
        prop = custom_qaoa_ansatz(graph, proposed_mapping, backend.coupling_map, reps=reps) # for proposed mapping
        
        if stop_after_routing:
            func_with_mapping = custom_routed_circuit
            func_sabre = sabre_routed_circuit
        else:
            func_with_mapping = final_optimized_circuit
            func_sabre = final_optimized_circuit

        for _ in range(average_over):
            swap_avg['fied'].append(func_with_mapping(circuit=prop, backend=backend, initial_mapping=proposed_mapping, opt_level=opt_level).count_ops().get(op_name, 0))
            swap_avg['triv'].append(func_with_mapping(circuit=gen, backend=backend, initial_mapping={i: i for i in range(n)}, opt_level=opt_level).count_ops().get(op_name, 0))
            swap_avg['sabre'].append(func_sabre(circuit=gen, backend=backend, opt_level=opt_level).count_ops().get(op_name, 0))

        swaps['means']['SABRE'].append(np.mean(swap_avg['sabre']))
        swaps['stddevs']['SABRE'].append(np.std(swap_avg['sabre']))

        swaps['means']['Proposed'].append(np.mean(swap_avg['fied']))
        swaps['stddevs']['Proposed'].append(np.std(swap_avg['fied']))

        swaps['means']['Trivial'].append(np.mean(swap_avg['triv']))
        swaps['stddevs']['Trivial'].append(np.std(swap_avg['triv']))

    return swaps

def return_ops_all_methods(
    backend,
    graph_type='reg',
    param=4,
    average_over=5,
    node_list=list(range(10,101,2)),
    k_list=[2,3,5],
    reps=2,
    opt_level=1,
    op_name='ecr',
    stop_after_routing=False,
    normalize_laplacian=True,
    noise_aware=False,
):
    methods = ["SABRE", "Trivial", "FOM"]
    for k in k_list:
        methods.append(f"SAM{k}")
    
    swaps = {
        'means': {m: [] for m in methods},
        'stddevs': {m: [] for m in methods}
    }

    for n in tqdm(node_list, desc=f'Iterating for number of nodes', leave=False):
        swap_avg = {m: [] for m in methods}

        # Generate problem graph
        if graph_type == 'reg':
            graph = nx.random_regular_graph(n=n, d=param, seed=0)
        else:
            graph = barabasi_albert_graph(n=n, m=param, seed=0)

        # Create QAOA circuit (unmapped)
        gen = gen_QAOA_circuit(graph, reps=reps)

        # proposed circuit
        prop = custom_qaoa_ansatz(graph, fiedler_mapping, backend.coupling_map, reps=reps)

        fiedler_mapping = fiedler_ordered_mapping(graph, backend, noise_aware=noise_aware, normalized=normalize_laplacian)
        trivial_mapping = {i: i for i in range(n)}
        assign_mappings = {
            f"SAM{k}": spectral_assignment_mapping(graph, backend, k=k, noise_aware=noise_aware, normalized=normalize_laplacian)
            for k in k_list
        }

        # Circuit Execution & Counting
        if stop_after_routing:
            exec_map_func = custom_routed_circuit
            exec_sabre_func = sabre_routed_circuit
        else:
            exec_map_func = final_optimized_circuit
            exec_sabre_func = final_optimized_circuit

        for i in range(average_over):
            # Fiedler
            swap_avg['FOM'].append(
                exec_map_func(circuit=prop, 
                              backend=backend, initial_mapping=fiedler_mapping, opt_level=opt_level).count_ops().get(op_name, 0)
            )
        
            # SAM with different k
            for k in k_list:
                key = f"SAM{k}"
                swap_avg[key].append(
                    exec_map_func(circuit=prop, 
                                  backend=backend, initial_mapping=assign_mappings[f"SAM{k}"], opt_level=opt_level).count_ops().get(op_name, 0)
                )

            # Trivial
            swap_avg['Trivial'].append(
                exec_map_func(circuit=gen, backend=backend, initial_mapping=trivial_mapping, opt_level=opt_level).count_ops().get(op_name, 0)
            )

            # SABRE
            swap_avg['SABRE'].append(
                exec_sabre_func(circuit=gen, backend=backend, opt_level=opt_level).count_ops().get(op_name, 0)
            )

        # Statistical Results
        for m in methods:
            key = m
            swaps['means'][m].append(np.mean(swap_avg[key]))
            swaps['stddevs'][m].append(np.std(swap_avg[key]))

    return swaps

# Colors for consistency across plots
colors = {
    "SABRE":   "#E69F00",
    "Trivial": "#56B4E9",
    "FOM": "#009E73",
    "SAM2": "#0072B2",
    "SAM3": "#D55E00",
    'SAM5': "#F0E442",
    "SAM4": "#CC79A7"
}

def plot_swap_results(swaps_dict, node_list, k_list, params, zoom=None, title_prefix="", ylabel="SWAP Count"):
    """
    swaps_dict: swap statistics returned from return_ops_all_methods()
    node_list: list of graph sizes tested (x-axis)
    params: list of values used for graph param (ex: degrees or m attachments)
    """
    methods = ["SABRE", "Trivial", "FOM"]
    for k in k_list:
        methods.append(f"SAM{k}")
    if zoom:
        node_list=node_list[:zoom]
    for param in params:
        plt.figure(figsize=(9, 6))
        
        for method in methods:
            means=swaps_dict[param]['means'][method]
            stds=swaps_dict[param]['stddevs'][method]
            if zoom:
                means=means[:zoom]
                stds=stds[:zoom]
            mean_vals = np.array(means)
            std_vals  = np.array(stds)
            
            plt.plot(node_list, mean_vals,
                     label=method,
                     color=colors.get(method, None),
                     linewidth=2)

            plt.fill_between(node_list,
                             mean_vals - std_vals,
                             mean_vals + std_vals,
                             alpha=0.25,
                             color=colors.get(method, None))

        plt.xlabel("Number of Nodes", fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        if title_prefix:
            plt.title(f"{title_prefix} (param = {param})", fontsize=18)

        plt.grid(alpha=0.3)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.show()

def compute_percentage_improvement(swaps_dict, methods, baseline="SABRE"):
    """
    Compute percentage improvement over a baseline method.

    Parameters
    ----------
    swaps_dict : dict
        Output of return_ops_all_methods() for a single param
    methods : list
        Methods to compare against baseline (e.g. ["FOM", "SAM2", "SAM3"])
    baseline : str
        Baseline method name (default: "SABRE")

    Returns
    -------
    improvements : dict
        {
            method: {
                'means': [...],
                'stddevs': [...]
            }
        }
    """
    improvements = {}

    sabre_means = np.array(swaps_dict['means'][baseline])
    sabre_stds  = np.array(swaps_dict['stddevs'][baseline])

    for method in methods:
        m_means = np.array(swaps_dict['means'][method])
        m_stds  = np.array(swaps_dict['stddevs'][method])

        # Percentage improvement
        perc_mean = 100.0 * (sabre_means - m_means) / sabre_means

        # Error propagation (first-order approximation)
        perc_std = 100.0 * np.sqrt(
            (m_stds / sabre_means)**2 +
            (m_means * sabre_stds / sabre_means**2)**2
        )

        improvements[method] = {
            'means': perc_mean.tolist(),
            'stddevs': perc_std.tolist()
        }

    return improvements

def compute_all_improvements(swaps_all_params, k_list, baseline="SABRE"):
    """
    Compute improvements over SABRE for all params.

    Returns
    -------
    improvements_all : dict
        {
            param: {
                method: {'means': [...], 'stddevs': [...]}
            }
        }
    """
    methods = ["FOM"] + [f"SAM{k}" for k in k_list]

    improvements_all = {}
    for param, swaps_dict in swaps_all_params.items():
        improvements_all[param] = compute_percentage_improvement(
            swaps_dict,
            methods=methods,
            baseline=baseline
        )

    return improvements_all

def plot_percentage_improvements(
    improvements_all,
    node_list,
    k_list,
    params,
    zoom=None,
    ylabel="Improvement over SABRE (%)",
    title_prefix=""
):
    methods = ["FOM"] + [f"SAM{k}" for k in k_list]

    if zoom:
        node_list = node_list[:zoom]

    for param in params:
        plt.figure(figsize=(9, 6))

        for method in methods:
            means = improvements_all[param][method]['means']
            stds  = improvements_all[param][method]['stddevs']

            if zoom:
                means = means[:zoom]
                stds  = stds[:zoom]

            means = np.array(means)
            stds  = np.array(stds)

            plt.plot(
                node_list,
                means,
                label=method,
                color=colors.get(method, None),
                linewidth=2
            )

            plt.fill_between(
                node_list,
                means - stds,
                means + stds,
                alpha=0.25,
                color=colors.get(method, None)
            )

        plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)

        plt.xlabel("Number of Nodes", fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        if title_prefix:
            plt.title(f"{title_prefix} (param = {param})", fontsize=18)

        plt.grid(alpha=0.3)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.show()

def QAOA_max_cut(candidate_circuit:QuantumCircuit, cost_hamiltonian:SparsePauliOp, backend, reps=2, estimator_shots=1000, sampler_shots=1000):
    initial_gamma = np.pi
    initial_beta = np.pi/2
    init_params = [initial_gamma, initial_beta] * reps

    objective_func_vals = [] # Global variable

    def cost_func_estimator(params, ansatz, hamiltonian, estimator):
        isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

        pub = (ansatz, isa_hamiltonian, params)
        job = estimator.run([pub])

        results = job.result()[0]
        cost = results.data.evs

        objective_func_vals.append(cost)

        return cost

    with Session(backend=backend) as session:
        estimator = Estimator(mode=session)
        estimator.options.default_shots = estimator_shots

        estimator.options.dynamical_decoupling.enable = True
        estimator.options.dynamical_decoupling.sequence_type = "XY4"
        estimator.options.twirling.enable_gates = True
        estimator.options.twirling.num_randomizations = "auto"

        result = minimize(
            cost_func_estimator,
            init_params,
            args=(candidate_circuit, cost_hamiltonian, estimator),
            method="COBYLA",
            tol=1e-2,
        )
    optimized_circuit = candidate_circuit.assign_parameters(result.x)

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = sampler_shots

    # Set simple error suppression/mitigation options
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.num_randomizations = "auto"

    pub= (optimized_circuit, )
    job = sampler.run([pub], shots=int(1e4))
    counts_int = job.result()[0].data.meas.get_int_counts()
    counts_bin = job.result()[0].data.meas.get_counts()
    shots = sum(counts_bin.values())
    final_distribution_int = {key: val/shots for key, val in counts_int.items()}
    final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}

    return final_distribution_bin

def optimized_parameters(candidate_circuit: QuantumCircuit, cost_hamiltonian:SparsePauliOp, backend, reps=2, shots=100):
    initial_gamma = np.pi
    initial_beta = np.pi/2
    init_params = [initial_gamma, initial_beta] * reps

    objective_func_vals = [] # Global variable

    def cost_func_estimator(params, ansatz, hamiltonian, estimator):
        isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

        pub = (ansatz, isa_hamiltonian, params)
        job = estimator.run([pub])

        results = job.result()[0]
        cost = results.data.evs

        objective_func_vals.append(cost)

        return cost

    with Session(backend=backend) as session:
        estimator = Estimator(mode=session)
        estimator.options.default_shots = shots

        estimator.options.dynamical_decoupling.enable = True
        estimator.options.dynamical_decoupling.sequence_type = "XY4"
        estimator.options.twirling.enable_gates = True
        estimator.options.twirling.num_randomizations = "auto"

        result = minimize(
            cost_func_estimator,
            init_params,
            args=(candidate_circuit, cost_hamiltonian, estimator),
            method="COBYLA",
            tol=1e-2,
        )
        return result.x
    
def sample_distribution(candidate_circuit:QuantumCircuit, optimized_params, backend, shots=1000):
    optimized_circuit = candidate_circuit.assign_parameters(optimized_params)

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = shots

    # Set simple error suppression/mitigation options
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.num_randomizations = "auto"

    pub= (optimized_circuit, )
    job = sampler.run([pub], shots=int(1e4))
    return job.job_id()

def dist_from_job_id(job_id, backend):
    job = backend.retrieve_job(job_id)
    result = job.result()
    counts = result[0].data.meas.get_counts()
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}

def evaluate_cut(graph: nx.Graph, bitstring: str) -> float:
    cut_value = 0.0
    for u, v in list(graph.edges()):
        if bitstring[u] != bitstring[v]:  # Nodes are in different partitions
            cut_value += 1
    return cut_value

def maxcut_dict(graph, cut_dict):
    mc_dict = defaultdict(int)
    for bitstr, prob in cut_dict.items():
        mc_dict[evaluate_cut(graph, bitstr[::-1])] += prob
    return mc_dict
    
def maxcut_dist(sim, backend, num_nodes, graph_type='reg', param=4, reps=1, opt_level=3, estimator_shots=1000, sampler_shots=1000, normalize_laplacian=True, tech='1D', k=2, noise_aware=False, default=False, gate='cz'):
    if graph_type=='reg':
        graph = nx.random_regular_graph(n=num_nodes, d=param, seed=0)
    else:
        graph = nx.barabasi_albert_graph(n=num_nodes, m=param, seed=0)

    hamiltonian = get_hamiltonian(graph)

    if default:
        opt_gen = final_optimized_circuit(gen_QAOA_circuit(graph, reps=reps), backend, initial_mapping=None, opt_level=opt_level)
        print('Default=> ', gate,': ', opt_gen.count_ops().get(gate, 0), 'Depth: ',opt_gen.depth())
        return QAOA_max_cut(opt_gen, hamiltonian, backend=sim, reps=reps, estimator_shots=estimator_shots, sampler_shots=sampler_shots)
    else:
        if tech=='1D':
                proposed_mapping = fiedler_ordered_mapping(graph, backend, noise_aware=noise_aware, normalized=normalize_laplacian)
        elif tech=='2D':
                proposed_mapping = spectral_assignment_mapping(graph, backend, k=k, noise_aware=noise_aware, normalized=normalize_laplacian)

        # Distribution with Proposed Initial Mapping
        prop = custom_qaoa_ansatz(graph, proposed_mapping, backend.coupling_map, reps=reps)
        opt_prop = final_optimized_circuit(prop, backend, proposed_mapping, opt_level=opt_level)

        print('Prop=> ', gate,': ', opt_prop.count_ops().get(gate, 0), 'Depth: ',opt_prop.depth())
        
        return QAOA_max_cut(opt_prop, hamiltonian, backend=sim, reps=reps, estimator_shots=estimator_shots, sampler_shots=sampler_shots)

def most_likely_cut(dist: dict):
    keys = list(dist.keys())
    values = list(dist.values())
    most_likely = keys[np.argmax(np.abs(values))]

    return most_likely[::-1]

def expected_size(graph, dist: dict):
    keys = list(dist.keys())
    values = list(dist.values())
    sumValue = 0.0
    
    for key in keys:
        sumValue += dist[key] * evaluate_cut(graph, key[::-1]) # qiskit circuit measurements are returned in little-endian format, so we have to reverse the bitstrings

    return sumValue

def random_size(graph, dist: dict):
    keys = list(dist.keys())
    numKey = float(len(keys))
    sumValue = 0.0
    
    for key in keys:
        sumValue += evaluate_cut(graph, key[::-1]) / numKey

    return sumValue

import matplotlib.pyplot as plt

def plot_histogram(dist, keep_top=50, title=None, ax=None, y_lim=None):
    # Step 1: Get top `keep_top` items by value
    if not keep_top or keep_top>len(dist):
        keep_top = len(dist)
    top_items = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:keep_top]

    # Step 2: Sort by bitstring key (lexicographically)
    top_items_sorted = sorted(top_items, key=lambda x: x[0])

    keys, values = zip(*top_items_sorted)

    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # ax.bar(range(len(values)), values, tick_label=keys)
    ax.bar(range(len(values)), values, tick_label=keys, width=0.5)
    ax.set_xticks([])
    # ax.set_xticklabels(keys, rotation=90)
    ax.set_xlabel(f'Bitstrings')
    ax.set_ylabel('Probability')
    if y_lim:
        ax.set_ylim(0, y_lim)
    if title:
        ax.set_title(title)

    if ax is None:
        plt.tight_layout()
        plt.show()

def show_histograms(dist, y_lim=None, keep_top=None):
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))  # (rows, cols)

    plot_histogram(dist['def'], ax=axs[0], keep_top=keep_top)
    plot_histogram(dist['prop'], ax=axs[1], keep_top=keep_top)

    axs[0].set_title("Default Qiskit")
    axs[1].set_title("With proposed method")

    # uncomment if want to see with consistent scale
    if y_lim:
        axs[0].set_ylim(0, y_lim)
        axs[1].set_ylim(0, y_lim)

    plt.tight_layout()
    plt.show()

def average_distributions(dists):
    """
    Average a list of measurement distributions.
    Each distribution is a dict: bitstring -> probability.
    """
    avg = defaultdict(float)
    for dist in dists:
        for bitstring, prob in dist.items():
            avg[bitstring] += prob

    n = len(dists)
    for bitstring in avg:
        avg[bitstring] /= n

    return dict(avg)

def maxcut_results(
    sim,
    backend,
    num_nodes,
    graph_type='reg',
    param=4,
    reps=1,
    opt_level=1,
    estimator_shots=1000,
    sampler_shots=1000,
    normalize_laplacian=True,
    tech='1D',
    k=2,
    noise_aware=True,
    default=True,
    gate='cz',
    average_over=5,
):
    # ---- Graph generation ----
    if graph_type == 'reg':
        graph = nx.random_regular_graph(n=num_nodes, d=param, seed=0)
    else:
        graph = barabasi_albert_graph(n=num_nodes, m=param, seed=0)

    hamiltonian = get_hamiltonian(graph)

    all_dists = []
    expected_cuts = []

    # ---- Monte Carlo loop ----
    for _ in tqdm(range(average_over), desc=f'Running {average_over} iterations', leave=True):
        if default:
            circuit = gen_QAOA_circuit(graph, reps=reps)
            opt_circ = final_optimized_circuit(
                circuit, backend, initial_mapping=None, opt_level=opt_level
            )
        else:
            if tech == '1D':
                proposed_mapping = fiedler_ordered_mapping(graph, backend, noise_aware=noise_aware, normalized=normalize_laplacian)

            elif tech == '2D':
                proposed_mapping = spectral_assignment_mapping(graph, backend, k=k, noise_aware=noise_aware, normalized=normalize_laplacian)

            circuit = custom_qaoa_ansatz(
                graph, proposed_mapping, backend.coupling_map, reps=reps
            )
            opt_circ = final_optimized_circuit(
                circuit, backend, proposed_mapping, opt_level=opt_level
            )

        print(f'{gate} Count: {opt_circ.count_ops().get(gate, 0)}, Depth: {opt_circ.depth()}')
        # ---- Execute QAOA ----
        dist = QAOA_max_cut(
            opt_circ,
            hamiltonian,
            backend=sim,
            reps=reps,
            estimator_shots=estimator_shots,
            sampler_shots=sampler_shots,
        )

        all_dists.append(dist)

        exp_cut = expected_size(graph, dist)
        expected_cuts.append(exp_cut)

    # ---- Aggregate results ----
    mean_dist = average_distributions(all_dists)
    mean_cut = float(np.mean(expected_cuts))
    std_cut = float(np.std(expected_cuts))

    return mean_dist, mean_cut, std_cut
