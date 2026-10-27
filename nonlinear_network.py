import numpy as np
import networkx as nx
import torch as tc
from circuit_solver import circuit as ct
from circuit_solver import elements as es
from circuit_solver import utils
from seeding import seeded_model
import argparse


def default_params():
    input_gain = 1.0
    output_gain = 1.0

    # Learned resistors (input→hidden and hidden→output)
    k_min = 0
    k_max = 1E1
    resistor_param_range = [(k_min, k_max)]
    resistor_init_mode = 'normal'
    resistor_init_params = (1.,0.1)

    # Fixed grounding ideal diodes on hidden nodes
    # param_range min == max so clipping keeps conductance fixed
    g_diode = 1E3
    diode_param_range = [(0, 1E4)]
    diode_lr = 0.

    # Architecture
    N = 10   # input units
    H = 5    # hidden units
    M = 3    # output units

    # dtype = tc.float32
    dtype = tc.float64
    seed = 0    # RNG seed for parameter initialization

    return argparse.Namespace(**locals())


def new(**params):
    v = default_params()
    vars(v).update(params)

    # Elements
    in_element = es.Resistor(
        name='in',
        param_ranges=v.resistor_param_range,
        init_mode=v.resistor_init_mode,
        init_params=v.resistor_init_params,
    )
    out_element = es.Resistor(
        name='out',
        param_ranges=v.resistor_param_range,
        init_mode=v.resistor_init_mode,
        init_params=v.resistor_init_params,
    )
    # Ideal diode: fixed large conductance, not learned.
    # Direction GROUND → h_node so the diode conducts when V_h < 0,
    # clamping hidden voltages to be non-negative (ReLU-like).
    diode_element = es.IdealDiode(
        name='diode',
        param_ranges=v.diode_param_range,
        init_mode='constant',
        init_params=v.g_diode,
        learning_rates = [v.diode_lr]
    )
    # Reversed diode: direction h_node → GROUND so the diode conducts when V_h > 0,
    # clamping hidden voltages to be non-positive (negative ReLU-like).
    diode_neg_element = es.IdealDiode(
        name='diode_neg',
        param_ranges=v.diode_param_range,
        init_mode='constant',
        init_params=v.g_diode,
        learning_rates = [v.diode_lr]
    )

    absolute_high = max(v.input_gain, v.output_gain)
    model_config = ct.CircuitModelConfig(
        HIGH_voltage=absolute_high,
        LOW_voltage=-absolute_high,
        dtype=v.dtype,
    )

    # Node layout: (GROUND, HIGH, LOW, x_nodes, h_nodes, h_neg_nodes, y_nodes)
    # h_nodes:     diodes from GROUND → h  (clamp V_h >= 0, ReLU-like)
    # h_neg_nodes: diodes from h → GROUND  (clamp V_h <= 0, negative ReLU-like)
    segment_sizes = (1, 0, 0, v.N, v.H, v.H, v.M)
    N_nodes = sum(segment_sizes)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(N_nodes))

    GROUND, HIGH, LOW, x_nodes, h_nodes, h_neg_nodes, y_nodes = utils.segment_array(
        graph.nodes, segment_sizes
    )

    # Edges
    xh_edges     = utils.all_pairs_between(x_nodes,     h_nodes)      # input → hidden+ (learned)
    xh_neg_edges = utils.all_pairs_between(x_nodes,     h_neg_nodes)  # input → hidden- (learned)
    hy_edges     = utils.all_pairs_between(h_nodes,     y_nodes)      # hidden+ → output (learned)
    hy_neg_edges = utils.all_pairs_between(h_neg_nodes, y_nodes)      # hidden- → output (learned)
    hg_edges     = [(GROUND[0], h) for h in h_nodes]                  # ground → hidden+ (fixed diode)
    hg_neg_edges = [(h, GROUND[0]) for h in h_neg_nodes]              # hidden- → ground (fixed diode, reversed)

    graph.add_edges_from(xh_edges + xh_neg_edges + hy_edges + hy_neg_edges + hg_edges + hg_neg_edges)

    in_edges         = utils.get_edges_between(x_nodes,     h_nodes,     graph)
    in_neg_edges     = utils.get_edges_between(x_nodes,     h_neg_nodes, graph)
    out_edges        = utils.get_edges_between(h_nodes,     y_nodes,     graph)
    out_neg_edges    = utils.get_edges_between(h_neg_nodes, y_nodes,     graph)
    diode_edges      = utils.get_edges_between(GROUND,      h_nodes,     graph)
    diode_neg_edges  = utils.get_edges_between(h_neg_nodes, GROUND,      graph)

    element_dict = {
        in_element:        in_edges + in_neg_edges,
        out_element:       out_edges + out_neg_edges,
        diode_element:     diode_edges,
        diode_neg_element: diode_neg_edges,
    }

    node_type_dict = {
        'GROUND': GROUND,
        'HIGH':   HIGH,
        'LOW':    LOW,
        'x':      x_nodes,
        'h':      h_nodes,
        'h_neg':  h_neg_nodes,
        'y':      y_nodes,
    }

    circuit, model = seeded_model(graph, element_dict, node_type_dict, model_config, v.seed)

    return argparse.Namespace(**locals())
