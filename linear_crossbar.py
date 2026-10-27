import numpy as np
import networkx as nx
import torch as tc
from circuit_solver import circuit as ct
from circuit_solver import elements as es
from circuit_solver import utils
from seeding import seeded_model
import argparse

def default_params():
    # data params
    input_gain = 1.0
    output_gain = 1.0

    # input-output edges
    k_min = 0
    k_max = 1E1
    resistor_param_range = [(k_min, k_max)]
    resistor_init_mode = 'constant'
    resistor_init_params = 1.

    include_ground_edges = False
    g_min = 0
    g_max = 1E2
    ground_param_range = [(g_min, g_max)]
    ground_init_mode = 'constant'
    ground_init_params = g_min
    
    # Architecture params 
    N = 100
    M = 10
    dtype = tc.float32
    seed = 0    # RNG seed for parameter initialization

    return argparse.Namespace(**locals())

def new(**params):
    # update default_params
    v = default_params()
    vars(v).update(params)

    if not v.include_ground_edges:
        v.g_min = 0.
        v.g_max = 0.
        v.ground_param_range = [(v.g_min, v.g_max)]
        v.ground_init_mode = 'constant'
        v.ground_init_params = v.g_min

    out_element = es.Resistor(name='out', param_ranges=v.resistor_param_range, init_mode=v.resistor_init_mode, init_params=v.resistor_init_params)
    ground_element = es.Resistor(name='ground', param_ranges=v.ground_param_range, init_mode=v.ground_init_mode,init_params=v.ground_init_params)

    absolute_high = max(v.input_gain, v.output_gain) 
    model_config = ct.CircuitModelConfig(
        HIGH_voltage=absolute_high,
        LOW_voltage=-absolute_high,
        dtype=v.dtype
    )

    segment_sizes = (1, 0, 0, v.N, v.M)
    N_nodes = sum(segment_sizes)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(N_nodes)) 

    # define indices for different node types
    GROUND, HIGH, LOW, h_nodes, y_nodes = utils.segment_array(graph.nodes, segment_sizes)   # HIGH = [], LOW = []

    fixed_nodes = GROUND + HIGH + LOW

    # add edges
    hy_edges = utils.all_pairs_between(y_nodes, h_nodes)
    yf_edges = utils.all_pairs_between(y_nodes, fixed_nodes)
    graph.add_edges_from(hy_edges)
    if v.include_ground_edges:
        graph.add_edges_from(yf_edges)
        ground_edges = utils.get_edges_between(y_nodes, GROUND, graph)

    # Define edge types
    out_edges = utils.get_edges_between(h_nodes, y_nodes, graph)

    # initialize Circuit
    if v.include_ground_edges:
        element_dict = {
            out_element: out_edges,
            ground_element: ground_edges
        }
    else:
        element_dict = {
            out_element: out_edges,
        }


    node_type_dict = {
        'GROUND': GROUND,
        'HIGH': HIGH,
        'LOW': LOW,
        'x': h_nodes,
        'y': y_nodes,
    }

    circuit, model = seeded_model(graph, element_dict, node_type_dict, model_config, v.seed)


    return argparse.Namespace(**locals())

