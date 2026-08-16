# Welcome to PyGhostID's documentation!

## Background
PyGhostID is a Python package for identifying generalized saddle-node ghosts in the vicinity of saddle-node bifurcations and their composite ghost structures such as ghost channels and ghost cycles in dynamical systems. PyGhostID's main function is the implementation of **GhostID**, a trajectory-based algorithm to identify saddle-node ghosts in dynamical systems. Besides the main algorithm, additional functions allow users to identify composite structures of ghosts ([ghost cycles, channels and networks](https://doi.org/10.1103/PhysRevLett.133.047202)) and track ghosts versus changing parameters.

![Figure1](_static/PyGhostID.png)
**Figure 1**: (a) flow-diagram of ghostID, (b) data associated with ghost states, (c) functionalities of the PyGhostID package.

## Installation

Install the latest version of PyGhostID via pip:

```bash
pip install PyGhostID
```
## Contact

For questions or additional information, please write to [daniel.koch@umanitoba.ca](mailto:daniel.koch@umanitoba.ca).

## Related Publication

The theory behind PyGhostID and some applications can be found in our research article:

[Daniel Koch, Akhilesh Nandan (2026). Generalized saddle-node ghosts and their composite structures in dynamical systems. arxiv: 2604.05194.](http://arxiv.org/abs/2604.05194)

All results from the paper can be reproduced using the code available on [github](https://github.com/KochLabCode/PyGhostID).

If you use PyGhostID or its repository in your research, please cite our paper!


## Overview

```{toctree}
:maxdepth: 2
:caption: Contents:

quickstart
tutorials
changelog
API Reference <api/modules>
```