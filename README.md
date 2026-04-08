# Overview
PyGhostID is a Python package for identifying generalized saddle-node ghosts in the vicinity of saddle-node bifurcations and their composite ghost structures such as ghost channels and ghost cycles in dynamical systems.
PyGhostID's main function is the implementation of GhostID, a trajectory-based algorithm to identify saddle-node ghosts in dynamical systems.

# Quick Start

Call `ghostID`as follows:

```python
import pyghostid as gid

# Minimal working example
result = ghostID(model, params, dt, trajectory, epsilon_gid = 0.05, delta_gid=0.05, **kwargs)
```

where `model`is the Python function describing the system dynamics, `parameters`are the model parameters to be given as argument to `model`, `dt`is the stepsize and `trajectory`is a trajectory simulated by the system. It returns `ghostSeq`, a Python list of identified ghost states (Python dictionary) and, if `return$\_$ctrl$\_$figs = True`, the figures for the control plots requested by the user. The hyper-parameters of the algorithm are as follows:

- `epsilon_gid`(float): Radius of the ε-sphere around Q-minima determining the trajectory segments along which eigenvalues are evaluated. For many models considered here we found values in the range of 0.01 to 0.1 a reasonable choice, hence the default value is set to 0.05. However, suitable values strongly depend on the model and the typical ranges of the phase space variables. Generally, the value should be chosen big enough such that trajectory segments consist of enough points to allow a reliable identification of the eigenvalue spectrum along the segment, but small enough so that the eigenvalues are still representative of the local phase-space topology around potential ghosts. A good strategy for choosing `epsilon_Qmin` is to start with small values and increase it until eigenvalue control plots look reasonably smooth.
- `delta_gid` (kwarg, float): Distance in phase space between two ghosts identified by GhostID above which they are considered distinct and given different identifiers. Default value is 0.1.
- `peak_kwargs` (kwarg, dictionary): Can contain any kwarg for SciPy's `find_peaks` function to improve the detection of Q-minima from peaks in the pQ-timeseries.
- `evLimit` (kwarg, float): Default value is 0. Enables indirect method of identifying ghosts if `evLimit` > 0. A trajectory segment is considered to pass nearby a ghost if the following holds true: the absolute value of the median of the eigenvalues along the trajectory segment is below `evLimit`, the linear fit of the eigenvalues along the segment has an R²≥0.99 and a slope that lies within the range given by the kwarg `slopeLimits` (two-element array of floats, default is [0,∞]).
- `batchModel` (kwarg, function): Vectorized version of the model able to handle batch inputs. Either manually coded model function or made via `make_batch_model`. While the algorithm calls `make_batch_model` itself at its initialization, providing a pre-calculated batch model can be useful to improve performance when `ghostID` is called many times using the same model. If at least one control plot is enabled, control plots will either be shown inline or returned if `return_ctrl_figs` is set to `True`.

Eigenvalues are calculated by `ghostID` using the `jax.numpy.linalg.eigvals` function. However, there is no guarantee that the indexing of the eigenvalues along a trajectory segment remains the same for consecutive points along the segment, i.e. λ₁ at t might be λ₂ at step t+1, thereby potentially interfering with the identification of ghosts. While we found the indexing to remain the same for the majority of cases, we also found cases in which eigenvalues were jumbled. `ghostID` has two complementary methods to deal with such cases, outlier removal and sorting of eigenvalues across time, that can be controlled as follows:

- `ev_outlier_removal` (kwarg, boolean): Removes eigenvalues along a trajectory based on outlier detection, where eigenvalues are removed if, along a sliding window, they fall above q₃ + k(q₃ - q₁) or below q₁ - k(q₃ - q₁), where k is a positive float, q₁ and q₃ the 25th and 75th percentile, respectively. Default value is `False`.
- `ev_outlier_removal_k` (kwarg, float): Determines the size of the range outside which eigenvalues are considered outliers. Default value is 1.5.
- `ev_outlier_removal_ws` (kwarg, float): Determines the size of the sliding window in which eigenvalues are evaluated for outliers. Default value is 7.
- `eigval_NN_sorting` (kwarg, boolean): Sorts eigenvalues λᵢ(t) across time for a given index i by making a linear prediction λₚ of the next real value of Re(λᵢ(t+1)) and assigns λᵢ(t+1) = λⱼ(t+1) for 1≤j≤n for which |λₚ - λⱼ(t+1)| is minimal. Default value is `False`.

`ghostID` also features several control outputs that are helpful for trouble-shooting and selecting hyper-parameters:

- `display_warnings` (kwarg, boolean): show/hide warning messages from GhostID.
- `ctrlOutputs` (kwarg, dictionary): Several keys can be used to plot the algorithm's two core quantities, Q- and eigenvalues, and customize the plots.
  - `ctrl_qplot` (boolean): enables plot of pQ-values and Q-minima along trajectory if set to `True`.
  - `qplot_xscale` (string): set scale of x-axis to `"linear"` (default) or `"log"`.
  - `qplot_yscale` (string): set scale of y-axis to `"linear"` (default) or `"log"`.
  - `ctrl_evplot` (boolean): if set to `True`, a plot of eigenvalues along each trajectory segment around identified Q-minima is shown including information about the evaluation criteria for ghosts which are listed in the plot's heading.
  - `evplot_xscale` (string): set scale of x-axis to `"linear"` (default) or `"log"`.
  - `evplot_yscale` (string): set scale of y-axis to `"linear"` (default) or `"log"`.
- `return_ctrl_figs` (kwarg, boolean): Returns control plots for manual customization of plot settings if set to `True`. Default value is `False`.

For a full documentation of PyGhostID's capabilities including its other functions (tracking ghosts versus parameter changes, identifying ghost channels and ghost cycles), please refer to:

[Daniel Koch, Akhilesh Nandan (2026). Generalized saddle-node ghosts and their composite structures in dynamical systems. arxiv: 2604.05194.](http://arxiv.org/abs/2604.05194)

If you use PyGhostID in your research, please cite our paper.