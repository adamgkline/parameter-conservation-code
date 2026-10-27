# parameter-conservation-code

Accompanying code for the paper "A Conservation Law for Equilibrium Propagation and Coupled Learning" by Joshua A McGinnis, Adam G Kline, and Yoichiro Mori. The preprint is available at [arXiv:2606.15444](https://arxiv.org/abs/2606.15444).

## Setup

```bash
pip install -r requirements.txt
```

This installs all dependencies, including the [`circuit-solver`](https://github.com/adamgkline/circuit-solver) package that the simulations are built on.

## Contents

The Jupyter notebooks are named after the figure in the paper that they produce:

| Notebook | Paper figure |
| --- | --- |
| `Figure 3 - voltage divider flow plots.ipynb` | Fig. 3 |
| `Figure 4 - crossbar trajectories.ipynb` | Fig. 4 |
| `Figure 5 - ER graph trajectories.ipynb` | Fig. 5 |
| `Figure 6 - nonlinear network.ipynb` | Fig. 6 |
| `large ER graph trajectories.ipynb` | Not in the paper. The ER graph experiment run on a larger graph. |

Supporting modules:

- `linear_crossbar.py`, `linear_er.py`, `nonlinear_network.py`: network construction and default parameters for each experiment
- `seeding.py`: reproducible parameter initialization for `circuit-solver` models

## Note on the sign of η

`circuit-solver` and the paper use **opposite sign conventions** for the clamping parameter η. In the code, a positive η pushes the outputs toward the target state. In the paper, a negative η does. That's why some plot labels in the notebooks show η with the opposite sign to the variable used in the code.
