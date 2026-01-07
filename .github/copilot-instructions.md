# PyGhostID - AI Coding Guidelines

## Project Overview
PyGhostID identifies "ghosts" (near-saddle-node bifurcations) and composite ghost structures in dynamical systems. The core algorithm analyzes trajectories to detect regions where systems linger near ghost attractors.

## Architecture
- **`src/PyGhostID/core.py`**: Main `ghostID()` function and supporting algorithms
- **`src/PyGhostID/_utils.py`**: Utility functions for Jacobian computation, batch processing, plotting
- **`test/`**: Integration tests with real dynamical models (FHN, GRN networks)
- **`paper/`**: Research validation code and figure generation

## Key Patterns
### Function Naming
- Use camelCase for main functions: `ghostID()`, `track_ghost_branch()`, `unify_IDs()`
- Internal helpers use snake_case: `make_jacfun()`, `make_batch_model()`

### Parameter Conventions
- Model functions: `def model(t, x, params)` where `params` is a dict
- Trajectories: 2D numpy arrays `(time_steps, dimensions)`
- Hyperparameters prefixed with `epsilon_`: `epsilon_Qmin`, `epsilon_SN_ghosts`

### JAX Integration
- Always JIT-compile Jacobians: `J_fun = jax.jit(jax.jacfwd(lambda x: model(0, x, params)))`
- Use `jnp` for JAX arrays, `np` for NumPy
- Batch models for trajectory processing: `Xs = model_batch(trajectory)`

### Error Handling
- Validate parameter shapes/types early with descriptive `ValueError` messages
- Use assertions for internal consistency checks

### Plotting
- Control plots via `ctrlOutputs` dict with keys like `"qplot"`, `"evplot"`
- Figure sizes in cm: `fig.set_size_inches(17/(2*2.54), 17/(3*2.54))`
- Return figures when `return_ctrl_figs=True`

## Development Workflow
### Setup
```bash
pip install -e .
# Or for development
pip install -e ".[dev]"  # if dev dependencies added
```

### Testing
- Run specific test: `python test/251101_pkg_test.py`
- Tests use real models; expect long runtimes for trajectory integration
- Validate with `assert len(ghosts) > 0` after calling `ghostID()`

### Building
```bash
python -m build
# Or
pip install build && python -m build
```

### Debugging
- Enable control plots: `ghostID(..., ctrlOutputs={"qplot": True, "evplot": True})`
- Check eigenvalue convergence with `evLimit` parameter
- Use `tqdm` progress bars for long computations

## Common Pitfalls
- JAX functions require static shapes; avoid dynamic slicing in JIT-compiled code
- Trajectory arrays must be C-contiguous for KD-tree operations
- Eigenvalue sorting assumes real parts dominate; check `jnp.real(eig[0])`
- NetworkX graphs use node attributes for ghost metadata

## Dependencies
- **JAX**: Core computations, JIT compilation
- **SciPy**: Optimization, peak finding, spatial queries
- **NetworkX**: Ghost connectivity graphs
- **Matplotlib**: Diagnostic plotting

## File Organization
- Place new models in `test/` with descriptive names (e.g., `260101_GRNghost.py`)
- Add utility functions to `_utils.py` if reusable across models
- Update `__init__.py` exports for new public functions