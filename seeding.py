import torch as tc
from circuit_solver import circuit as ct


def seeded_model(graph, element_dict, node_type_dict, model_config, seed):
    """Build Circuit and CircuitModel with reproducible parameter initialization.

    circuit_solver creates element layers in set order (which depends on object ids),
    so initial values are redrawn here in a fixed order (sorted by element name).
    The global torch RNG state is left untouched.
    """
    with tc.random.fork_rng(devices=[]):
        circuit = ct.Circuit(graph, element_dict)
        model = ct.CircuitModel(circuit, node_type_dict, model_config)
        tc.manual_seed(seed)
        with tc.no_grad():
            for name in sorted(model.element_layers):
                layer = model.element_layers[name]
                layer.theta_init.copy_(layer._initialize_parameters(layer.shape))
    model.reset()
    return circuit, model
