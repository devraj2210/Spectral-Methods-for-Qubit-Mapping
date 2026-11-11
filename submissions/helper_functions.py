import numpy as np
import random
import math
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from time import perf_counter

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler

from scipy.optimize import minimize

from implementation import (
    graph_laplacian_embedding,
    fiedler_ordered_mapping,
    spectral_assignment_mapping,
    sabre_layout,
    custom_qaoa_ansatz,
    get_hamiltonian,
    gen_QAOA_circuit,
    custom_routed_circuit,
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

from collections import defaultdict
import rustworkx as rx

from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from sat_mapping import SATMapper


def extract_longest_chain(backend):
    """
    Extracts an approximate longest simple path from the backend coupling graph.
    Works well for heavy-hex architectures.
    """

    coupling = backend.coupling_map.get_edges()

    G = nx.Graph()
    G.add_edges_from(coupling)

    # two-sweep heuristic
    start = list(G.nodes())[0]

    lengths = nx.single_source_shortest_path_length(G, start)
    u = max(lengths, key=lengths.get)

    lengths = nx.single_source_shortest_path_length(G, u)
    v = max(lengths, key=lengths.get)

    path = nx.shortest_path(G, u, v)

    return path


def build_line_swap_strategy(chain):
    """
    Creates a line swap strategy over a chain of qubits.
    """

    line_edges = [(chain[i], chain[i+1]) for i in range(len(chain)-1)]

    cmap = CouplingMap(line_edges)

    return SwapStrategy.from_line([q for q in chain])


def sat_line_mapping(graph, backend, timeout=30):
    """
    SAT-based initial mapping using longest heavy-hex chain.
    """

    chain = extract_longest_chain(backend)

    strategy = build_line_swap_strategy(chain)

    mapper = SATMapper(timeout=timeout)

    _, edge_map, layers = mapper.remap_graph_with_sat(
        graph,
        strategy
    )

    return edge_map, layers

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

def get_metric(circuit, metric="swap_count", op_name="swap"):

    if metric == "swap_count":
        return circuit.count_ops().get(op_name, 0)

    elif metric == "depth":
        return circuit.depth()

    elif metric == "two_qubit_depth":
        return circuit.depth(
            filter_function=lambda inst:
                inst.operation.num_qubits == 2
        )

    elif metric == "two_qubit_gate_count":
        return sum(
            instruction.operation.num_qubits == 2
            for instruction in circuit.data
        )

    elif metric == "swap_depth":
        return circuit.depth(
            filter_function=lambda inst:
                inst.operation.name == op_name
        )

    else:
        raise ValueError(f"Unknown metric: {metric}")

def return_ops_all_methods(
    backend,
    graph_type='reg',
    param=4,
    average_over=5,
    node_list=list(range(10,101,2)),
    k_list=[2,3,5],
    reps=2,
    op_name='ecr',
    stop_after_routing=False,
    noise_aware=False,
    metric="swap_count",
    opt_levels=None,
):
    """Collect routed SWAP counts and optimized circuit metrics only.

    SWAPs are counted on custom_routed_circuit. Two-qubit gate counts and
    depths are counted on final_optimized_circuit at each requested level.
    ``opt_level`` and ``stop_after_routing`` remain accepted for compatibility;
    pass ``opt_levels`` to collect multiple optimization levels.
    """
    methods = ["SABRE", "Trivial", "FOM"] + [f"SAM{k}" for k in k_list]
    multiple_levels_requested = opt_levels is not None
    opt_levels = list(dict.fromkeys(opt_levels if multiple_levels_requested else [1]))
    if not opt_levels:
        raise ValueError("opt_levels must contain at least one optimization level")
    if average_over < 1:
        raise ValueError("average_over must be at least 1")
    if graph_type not in {'reg', 'ba'}:
        raise ValueError("graph_type must be 'reg' or 'ba'")

    swaps_samples = {method: [] for method in methods}
    two_qubit_samples = {
        str(level): {method: [] for method in methods} for level in opt_levels
    }
    depth_samples = {
        str(level): {method: [] for method in methods} for level in opt_levels
    }

    def summarize(per_method):
        return {
            'means': {
                method: [float(np.mean(values)) for values in node_values]
                for method, node_values in per_method.items()
            },
            'stddevs': {
                method: [float(np.std(values)) for values in node_values]
                for method, node_values in per_method.items()
            },
            'medians': {
                method: [float(np.median(values)) for values in node_values]
                for method, node_values in per_method.items()
            },
        }

    for n in tqdm(node_list, desc=f'Collecting circuit ops (param={param})', leave=False):
        graph = (
            nx.random_regular_graph(n=n, d=param, seed=0)
            if graph_type == 'reg'
            else barabasi_albert_graph(n=n, m=param, seed=0)
        )
        fiedler_mapping = fiedler_ordered_mapping(
            graph, backend, noise_aware=noise_aware, normalized=False,
        )
        trivial_mapping = {i: i for i in range(n)}
        assign_mappings = {
            f"SAM{k}": spectral_assignment_mapping(
                graph, backend, k=k, noise_aware=noise_aware, normalized=True,
            )
            for k in k_list
        }

        generic = transpile(
            gen_QAOA_circuit(graph, reps=reps),
            basis_gates=backend.basis_gates, seed_transpiler=0,
        )
        fiedler_circuit = transpile(
            custom_qaoa_ansatz(graph, fiedler_mapping, backend.coupling_map, reps=reps),
            basis_gates=backend.basis_gates, seed_transpiler=0,
        )
        sam_circuits = {
            f"SAM{k}": transpile(
                custom_qaoa_ansatz(
                    graph, assign_mappings[f"SAM{k}"], backend.coupling_map, reps=reps,
                ),
                basis_gates=backend.basis_gates, seed_transpiler=0,
            )
            for k in k_list
        }
        method_inputs = {
            'SABRE': (generic, sabre_layout(generic.copy(), backend)),
            'Trivial': (generic, trivial_mapping),
            'FOM': (fiedler_circuit, fiedler_mapping),
        }
        method_inputs.update({
            method: (sam_circuits[method], assign_mappings[method])
            for method in sam_circuits
        })

        node_swaps = {method: [] for method in methods}
        node_two_qubit = {
            str(level): {method: [] for method in methods} for level in opt_levels
        }
        node_depth = {
            str(level): {method: [] for method in methods} for level in opt_levels
        }
        for _ in range(average_over):
            for method, (circuit, mapping) in method_inputs.items():
                routed = custom_routed_circuit(circuit.copy(), mapping, backend)
                node_swaps[method].append(
                    get_metric(routed, metric='swap_count', op_name='swap')
                )
                for level in opt_levels:
                    key = str(level)
                    optimized = final_optimized_circuit(
                        circuit.copy(), backend, initial_mapping=mapping,
                        opt_level=level,
                    )
                    node_two_qubit[key][method].append(
                        get_metric(optimized, metric='two_qubit_gate_count')
                    )
                    node_depth[key][method].append(
                        get_metric(optimized, metric='depth')
                    )

        for method in methods:
            swaps_samples[method].append(node_swaps[method])
            for level in opt_levels:
                key = str(level)
                two_qubit_samples[key][method].append(node_two_qubit[key][method])
                depth_samples[key][method].append(node_depth[key][method])

    first_level = str(opt_levels[0])
    results = {
        'swaps': summarize(swaps_samples),
        'two_qubit_gates': {
            level: summarize(two_qubit_samples[level]) for level in two_qubit_samples
        },
        'circuit_depth': {
            level: summarize(depth_samples[level]) for level in depth_samples
        },
    }
    first_level_two_qubit = results['two_qubit_gates'][first_level]
    first_level_depth = results['circuit_depth'][first_level]
    if not multiple_levels_requested:
        results['two_qubit_gates'] = first_level_two_qubit
        results['circuit_depth'] = first_level_depth
    if metric == 'depth':
        legacy_view = first_level_depth
    elif metric in {'two_qubit_gate_count', 'two_qubit_gates'}:
        legacy_view = first_level_two_qubit
    elif stop_after_routing or op_name == 'swap':
        legacy_view = results['swaps']
    else:
        legacy_view = results['two_qubit_gates'][str(opt_levels[0])]
    results['means'] = legacy_view['means']
    results['stddevs'] = legacy_view['stddevs']
    return results


def return_times_all_methods(
    backend, graph_type='reg', param=4, average_over=5,
    node_list=list(range(10, 101, 2)), k_list=[2, 3, 5], reps=2,
    noise_aware=False, opt_levels=None,
):
    """Measure mapping and final compilation time without collecting ops."""
    k_list = list(dict.fromkeys(k_list))
    multiple_levels_requested = opt_levels is not None
    opt_levels = list(dict.fromkeys(opt_levels if multiple_levels_requested else [1]))
    if not opt_levels:
        raise ValueError("opt_levels must contain at least one optimization level")
    if average_over < 1:
        raise ValueError("average_over must be at least 1")
    if graph_type not in {'reg', 'ba'}:
        raise ValueError("graph_type must be 'reg' or 'ba'")

    methods = ["SABRE", "Trivial", "FOM"] + [f"SAM{k}" for k in k_list]
    level_keys = [str(level) for level in opt_levels]
    samples = {
        'mapping_time': {method: [] for method in methods},
        'routing_optimization_time': {
            level: {method: [] for method in methods} for level in level_keys
        },
        'total_compilation_time': {
            level: {method: [] for method in methods} for level in level_keys
        },
    }

    def timed(fn, *args, **kwargs):
        start = perf_counter()
        result = fn(*args, **kwargs)
        return result, perf_counter() - start

    def summarize(per_method):
        summary = {
            'means': {}, 'stddevs': {}, 'medians': {}, 'trimmed_means': {},
            'q25': {}, 'q75': {}, 'iqrs': {},
        }
        for method, node_samples in per_method.items():
            def trimmed_mean(values):
                ordered = np.sort(np.asarray(values, dtype=float))
                return float(np.mean(ordered[1:-1] if len(ordered) >= 3 else ordered))
            summary['means'][method] = [float(np.mean(x)) for x in node_samples]
            summary['stddevs'][method] = [float(np.std(x)) for x in node_samples]
            summary['medians'][method] = [float(np.median(x)) for x in node_samples]
            summary['trimmed_means'][method] = [trimmed_mean(x) for x in node_samples]
            summary['q25'][method] = [float(np.quantile(x, .25)) for x in node_samples]
            summary['q75'][method] = [float(np.quantile(x, .75)) for x in node_samples]
            summary['iqrs'][method] = [
                float(np.quantile(x, .75) - np.quantile(x, .25))
                for x in node_samples
            ]
        return summary

    for n in tqdm(node_list, desc=f'Timing compilation (param={param})', leave=False):
        graph = (
            nx.random_regular_graph(n=n, d=param, seed=0)
            if graph_type == 'reg'
            else barabasi_albert_graph(n=n, m=param, seed=0)
        )
        node_samples = {
            'mapping_time': {method: [] for method in methods},
            'routing_optimization_time': {
                level: {method: [] for method in methods} for level in level_keys
            },
            'total_compilation_time': {
                level: {method: [] for method in methods} for level in level_keys
            },
        }

        for _ in range(average_over):
            # Circuit generation and basis transpilation are outside the timers.
            generic = transpile(
                gen_QAOA_circuit(graph, reps=reps),
                basis_gates=backend.basis_gates, seed_transpiler=0,
            )
            trivial_mapping, trivial_time = timed(lambda: {i: i for i in range(n)})
            fom_mapping, fom_time = timed(
                fiedler_ordered_mapping, graph, backend,
                noise_aware=noise_aware, normalized=False,
            )
            fom_circuit = transpile(
                custom_qaoa_ansatz(graph, fom_mapping, backend.coupling_map, reps=reps),
                basis_gates=backend.basis_gates, seed_transpiler=0,
            )
            method_inputs = {
                'Trivial': (generic, trivial_mapping, trivial_time),
                'FOM': (fom_circuit, fom_mapping, fom_time),
            }

            for k in k_list:
                method = f"SAM{k}"
                mapping, mapping_time = timed(
                    spectral_assignment_mapping, graph, backend, k=k,
                    noise_aware=noise_aware, normalized=True,
                )
                circuit = transpile(
                    custom_qaoa_ansatz(graph, mapping, backend.coupling_map, reps=reps),
                    basis_gates=backend.basis_gates, seed_transpiler=0,
                )
                method_inputs[method] = (circuit, mapping, mapping_time)

            sabre_circuit = transpile(
                gen_QAOA_circuit(graph, reps=reps),
                basis_gates=backend.basis_gates, seed_transpiler=0,
            )
            sabre_mapping, sabre_time = timed(sabre_layout, sabre_circuit, backend)
            method_inputs['SABRE'] = (sabre_circuit, sabre_mapping, sabre_time)

            for method, (circuit, mapping, mapping_time) in method_inputs.items():
                node_samples['mapping_time'][method].append(mapping_time)
                for level in opt_levels:
                    level_key = str(level)
                    _, route_opt_time = timed(
                        final_optimized_circuit,
                        circuit.copy(), backend, initial_mapping=mapping,
                        opt_level=level,
                    )
                    node_samples['routing_optimization_time'][level_key][method].append(
                        route_opt_time
                    )
                    node_samples['total_compilation_time'][level_key][method].append(
                        mapping_time + route_opt_time
                    )

        for method in methods:
            samples['mapping_time'][method].append(node_samples['mapping_time'][method])
            for level in level_keys:
                for stage in ('routing_optimization_time', 'total_compilation_time'):
                    samples[stage][level][method].append(
                        node_samples[stage][level][method]
                    )

    mapping_summary = summarize(samples['mapping_time'])
    route_opt_summaries = {
        level: summarize(samples['routing_optimization_time'][level])
        for level in level_keys
    }
    total_summaries = {
        level: summarize(samples['total_compilation_time'][level])
        for level in level_keys
    }
    if not multiple_levels_requested:
        return {
            'mapping_time': mapping_summary,
            'routing_optimization_time': route_opt_summaries[level_keys[0]],
            'total_compilation_time': total_summaries[level_keys[0]],
        }
    return {
        'mapping_time': mapping_summary,
        'routing_optimization_time': route_opt_summaries,
        'total_compilation_time': total_summaries,
    }


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

def plot_metric_results(results_dict, node_list, k_list, params, metric,
                        zoom=None, title_prefix="", ylabel=None, fill_between=True, plot='mean'):
    """
    results_dict: statistics returned from return_ops_all_methods(), indexed by param
    node_list: list of graph sizes tested (x-axis)
    params: list of values used for graph param (ex: degrees or m attachments)
    metric: a result key or dotted path, e.g. "swaps" or
        "execution_time.stop_after_routing"
    """
    metric_label = metric.split(".")[-1].replace("_", " ").title() if metric else "Metric"
    if ylabel is None:
        ylabel = metric_label

    methods = ["SABRE", "Trivial", "FOM"]
    for k in k_list:
        methods.append(f"SAM{k}")

    if zoom:
        node_list=node_list[:zoom]

    for param in params:
        plt.figure(figsize=(9, 6))

        metric_stats = results_dict[param]
        if metric:
            for path_part in metric.split("."):
                metric_stats = metric_stats[path_part]

        for method in methods:
            means=metric_stats['means'][method]
            stds=metric_stats['stddevs'][method]
            medians=metric_stats['medians'][method]
            if zoom:
                means=means[:zoom]
                stds=stds[:zoom]
            mean_vals = np.array(means)
            std_vals  = np.array(stds)
            median_vals = np.array(medians)

            
            plt.plot(node_list, median_vals if plot == 'median' else mean_vals,
                     label=method,
                     color=colors.get(method, None),
                     linewidth=2)

            if fill_between:
                plt.fill_between(node_list,
                                 mean_vals - std_vals,
                                 mean_vals + std_vals,
                                 alpha=0.25,
                                 color=colors.get(method, None))

        plt.xlabel("Number of Nodes", fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        # All collected metrics are non-negative. Starting both axes at zero
        # makes the chart origin explicit and avoids a misleading offset.
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # plot_title = title_prefix or metric_label
        # plt.title(f"{plot_title} (param = {param})", fontsize=18)

        plt.grid(alpha=0.3)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.show()


def plot_swap_results(swaps_dict, node_list, k_list, params, zoom=None,
                      title_prefix="", ylabel="SWAP Count"):
    """Plot the legacy top-level means/stddevs view from the result object."""
    plot_metric_results(
        swaps_dict, node_list, k_list, params, metric="",
        zoom=zoom, title_prefix=title_prefix or ylabel, ylabel=ylabel,
    )

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
        # Node counts start at zero; percentage improvements can be negative,
        # so only constrain the horizontal axis for these plots.
        plt.xlim(left=0)
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
                proposed_mapping = fiedler_ordered_mapping(graph, backend, noise_aware=noise_aware, normalized=False)

            elif tech == '2D':
                proposed_mapping = spectral_assignment_mapping(graph, backend, k=k, noise_aware=noise_aware, normalized=True)

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
