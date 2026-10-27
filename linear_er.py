import numpy as np
import networkx as nx
import torch as tc
from circuit_solver import circuit as ct
from circuit_solver import elements as es
from circuit_solver import utils
from seeding import seeded_model
import argparse
import random


def default_params():
    # data params
    input_gain = 1.0
    output_gain = 1.0

    # input-output edges
    k_min = 0
    k_max = 1E1
    resistor_param_range = [(k_min, k_max)]
    resistor_init_mode = 'normal'
    resistor_init_params = (1.,0.)

    include_ground_edges = False
    g_min = 0
    g_max = 1E2
    ground_param_range = [(g_min, g_max)]
    ground_init_mode = 'constant'
    ground_init_params = g_min

    # Architecture params
    N = 5       # input nodes
    M = 3       # output nodes
    N_h = 5     # hidden nodes
    p = 0.5     # ER edge probability
    dtype = tc.float32
    seed = 0    # RNG seed for graph generation and parameter initialization

    return argparse.Namespace(**locals())


def er_network(n, p, seed, directed=True, max_connected_attempts=100):
    """Seeded version of utils.er_network: resamples until the graph is (weakly) connected"""
    rng = random.Random(seed)
    for _ in range(max_connected_attempts):
        graph = nx.erdos_renyi_graph(n, p, seed=rng, directed=directed)
        check_graph = graph.to_undirected() if directed else graph
        if nx.is_connected(check_graph):
            break
    else:
        raise RuntimeError(f'no connected ER graph found in {max_connected_attempts} attempts (n={n}, p={p}, seed={seed})')

    for i, node in enumerate(graph.nodes):
        theta = i * 2 * np.pi / n
        graph.nodes[node]['pos'] = np.array([np.cos(theta), np.sin(theta)])
    return graph


def add_module(graph, new_module, module_nodes):
    """Merges new_module onto graph according to nodes"""
    node_mapping = {i:n for i, n in enumerate(module_nodes)}
    new_module = nx.relabel_nodes(new_module, node_mapping, copy=True)
    graph = nx.compose(graph, new_module)
    return graph

def new(**params):
    v = default_params()
    vars(v).update(params)

    if not v.include_ground_edges:
        v.g_min = 0.
        v.g_max = 0.
        v.ground_param_range = [(v.g_min, v.g_max)]
        v.ground_init_mode = 'constant'
        v.ground_init_params = v.g_min

    out_element = es.Resistor(name='out', param_ranges=v.resistor_param_range,
                              init_mode=v.resistor_init_mode, init_params=v.resistor_init_params)
    ground_element = es.Resistor(name='ground', param_ranges=v.ground_param_range,
                                 init_mode=v.ground_init_mode, init_params=v.ground_init_params)

    absolute_high = max(v.input_gain, v.output_gain)
    model_config = ct.CircuitModelConfig(
        HIGH_voltage=absolute_high,
        LOW_voltage=-absolute_high,
        dtype=v.dtype
    )

    # Build circuit: node 0 = GROUND, nodes 1..N = x, N+1..N+N_h = hidden, N+N_h+1..N+N_h+M = y
    N_internal = v.N + v.N_h + v.M
    N_nodes = 1 + N_internal
    graph = nx.DiGraph()
    graph.add_nodes_from(range(N_nodes))

    segment_sizes = (1, 0, 0, v.N, v.N_h, v.M)
    GROUND, HIGH, LOW, x_nodes, hidden_nodes, y_nodes = utils.segment_array(graph.nodes, segment_sizes)

    fixed_nodes = GROUND + HIGH + LOW
    internal_nodes = x_nodes + hidden_nodes + y_nodes
    
    # bulk edges
    er_graph = er_network(N_internal, v.p, seed=v.seed, directed=True)
    graph = add_module(graph, er_graph, internal_nodes)
    xx_edges = [(u, v) for (u, v) in graph.edges() if u in x_nodes and v in x_nodes]
    graph.remove_edges_from(xx_edges)

    # Add y -> GROUND edges
    if v.include_ground_edges:
        yf_edges = utils.all_pairs_between(y_nodes, fixed_nodes)
        graph.add_edges_from(yf_edges)
    
    # edge lists
    bulk_edges = utils.get_edges_between(internal_nodes, internal_nodes, graph)
    ground_edges = utils.get_edges_between(internal_nodes, GROUND, graph)

    # Classify each ER edge by the types of its endpoint nodes
    x_set, h_set, y_set = set(x_nodes), set(hidden_nodes), set(y_nodes)
    def _node_type(n):
        if n in x_set: return 'x'
        if n in h_set: return 'h'
        if n in y_set: return 'y'

    edge_type_inds = {}
    for i, (u, w) in enumerate(bulk_edges):
        t = (_node_type(u), _node_type(w))
        edge_type_inds.setdefault(t, []).append(i)

    # bulk_edges = utils.get_edges_between(internal_nodes, internal_nodes, graph)
    # ground_edges = utils.get_edges_between(internal_nodes, GROUND, graph)
    if v.include_ground_edges:
        element_dict = {
            out_element: bulk_edges,
            ground_element: ground_edges,
        }
    else:
        element_dict = {
            out_element: bulk_edges,
        }

    node_type_dict = {
        'GROUND': GROUND,
        'HIGH': HIGH,
        'LOW': LOW,
        'x': x_nodes,
        'hidden': hidden_nodes,
        'y': y_nodes,
    }

    circuit, model = seeded_model(graph, element_dict, node_type_dict, model_config, v.seed)

    return argparse.Namespace(**locals())
