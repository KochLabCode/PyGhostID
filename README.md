# Overview
PyGhostID is a Python package for identifying generalized saddle-node ghosts in the vicinity of saddle-node bifurcations and their composite ghost structures such as ghost channels and ghost cycles in dynamical systems. PyGhostID's main function is the implementation of GhostID, a trajectory-based algorithm to identify saddle-node ghosts in dynamical systems.

<p align="center">
  <img src="https://raw.githubusercontent.com/KochLabCode/PyGhostID/refs/heads/main/PyGhostID.png" alt="PyGhostID_scheme">
</p>
<p align="center"><em>a: GhostID algorithm. b: data recorded about identified ghosts. c: Features of PyGhostID.</em></p>

## Quick Start

Call `ghostID`as follows:

```python
import pyghostid as gid

# Minimal working example
result = ghostID(model, params, dt, trajectory)
```

where `model`is the Python function describing the system dynamics, `parameters`are the model parameters to be given as argument to `model`, `dt`is the stepsize and `trajectory`is a trajectory of the system. It returns `ghostSeq`, a Python list of identified ghost states (Python dictionary) and, if `return_ctrl_figs = True`, the figures for the control plots requested by the user. 

## Documentation

Please refer to our [readthedocs](https://pyghostid.readthedocs.io/en/latest/) page for information about PyGhostID's functions, tutorials and more. 

## Reproducing figures from the paper

The folder "paper" of this repository contains the code and data to reproduce the results and figures from the study:

[Daniel Koch, Akhilesh Nandan (2026). Generalized saddle-node ghosts and their composite structures in dynamical systems. arxiv: 2604.05194.](http://arxiv.org/abs/2604.05194)

If you use PyGhostID or its repository in your research, please cite our paper.

#### Dependencies

To recreate the figures and simulations from the study, you’ll need:

- Python 3.x
- PyGhostID (available at https://pypi.org/project/PyGhostID/)
- Other required Python packages (installable via the environment `PyGhostID.yaml`)

#### Running Code

- Main Figures:
Run any of the `FigureX.ipynb` scripts to generate the main figures from the paper.

Supplementary Figures & Videos:
- Use `Supp Figure X.ipynb` to reproduce additional analyses and visuals.

## Contact

For questions or additional information, please contact Daniel Koch: daniel.koch@umanitoba.ca
