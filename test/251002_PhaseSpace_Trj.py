import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp
import random

# ============================================================
#  Lotka–Volterra predator–prey system (JAX-based)
# ============================================================
def lotka_volterra(y, t, params):
    """Predator–prey ODE for JAX: y = [prey, predator]."""
    alpha, beta, delta, gamma = params
    x, z = y
    dxdt = alpha*x - beta*x*z
    dzdt = delta*x*z - gamma*z
    return jnp.array([dxdt, dzdt])

# ============================================================
#  Helpers for grid <-> point
# ============================================================
def index_from_point(x, grid_bounds, grid_shape):
    """Map continuous point x to integer grid index tuple. None if outside grid."""
    idx = []
    for xi, (bmin,bmax), Ni in zip(x, grid_bounds, grid_shape):
        if xi < bmin or xi > bmax:
            return None
        u = (xi - bmin) / (bmax - bmin)
        ii = int(np.floor(u * Ni))
        if ii == Ni:  # handle xi == bmax
            ii = Ni - 1
        idx.append(ii)
    return tuple(idx)

def cell_center_from_index(idx, grid_bounds, grid_shape):
    coords = []
    for ii, (bmin,bmax), Ni in zip(idx, grid_bounds, grid_shape):
        cell_width = (bmax - bmin) / Ni
        center = bmin + (ii + 0.5) * cell_width
        coords.append(center)
    return np.array(coords)

def neighboring_indices(idx, grid_shape, radius=1):
    ranges = [range(max(0,i-radius), min(Ni, i+radius+1)) for i, Ni in zip(idx, grid_shape)]
    for offset in np.ndindex(*[len(r) for r in ranges]):
        neighbor = tuple(ranges[d][offset[d]] for d in range(len(ranges)))
        yield neighbor

# ============================================================
#  JAX wrapper for solve_ivp
# ============================================================
def make_numpy_rhs_from_jax(model_jax, params):
    @jax.jit
    def f_jax(y, t):
        return model_jax(y, t, params)
    def rhs(t, y):
        return np.asarray(f_jax(jnp.asarray(y), t))
    return rhs

# ============================================================
#  Simulation routine with chunked integration
# ============================================================
def simulate_until_enter_or_exit(ic, T_total, dt_chunk, dt, rhs_numpy,
                                 index_from_point_fn, consec_thresh,
                                 visited_set, grid_bounds, grid_shape):
    t0 = 0.0
    processed_times, processed_states = [], []
    entered_visited, exit_boundary = False, False
    y0 = np.array(ic)

    while t0 < T_total and not entered_visited and not exit_boundary:
        t1 = min(T_total, t0 + dt_chunk)
        chunk_points = int(np.ceil((t1 - t0) / dt))
        t_eval = np.linspace(t0, t1, chunk_points)

        sol = solve_ivp(fun=rhs_numpy, t_span=(t0, t1), y0=y0, t_eval=t_eval, rtol=1e-6)

        consec = 0
        for ti, yi in zip(sol.t, sol.y.T):
            idx = index_from_point_fn(yi, grid_bounds, grid_shape)
            if idx is None:
                exit_boundary = True
                break

            if idx in visited_set:
                consec += 1
            else:
                consec = 0

            processed_times.append(ti)
            processed_states.append(yi.copy())

            if consec >= consec_thresh:
                entered_visited = True
                for _ in range(consec):
                    processed_times.pop()
                    processed_states.pop()
                break

        if not entered_visited and not exit_boundary:
            y0 = sol.y.T[-1].copy()
            t0 = sol.t[-1]

    return np.array(processed_times), np.array(processed_states), entered_visited, exit_boundary

def update_visited_from_states(states, visited_set, index_from_point_fn, grid_bounds, grid_shape):
    new = 0
    for y in states:
        idx = index_from_point_fn(y, grid_bounds, grid_shape)
        if idx is None:
            continue
        if idx not in visited_set:
            visited_set.add(idx)
            new += 1
    return new

# ============================================================
#  Frontier selection (boundary-based seeding)
# ============================================================
def boundary_indices(grid_shape):
    """Return all cells on the boundary of the grid."""
    boundary = set()
    for idx in np.ndindex(*grid_shape):
        if any(i==0 or i==Ni-1 for i,Ni in zip(idx,grid_shape)):
            boundary.add(idx)
    return boundary

def compute_frontier(visited_set, grid_shape):
    """Return neighbors of visited cells that are unvisited."""
    frontier = set()
    if not visited_set:
        return boundary_indices(grid_shape)
    for idx in visited_set:
        for neigh in neighboring_indices(idx, grid_shape, radius=1):
            if neigh not in visited_set:
                frontier.add(neigh)
    return frontier

def sample_initial_conditions_from_frontier(frontier, batch_size, grid_bounds, grid_shape):
    picked = random.sample(list(frontier), min(len(frontier), batch_size))
    return [cell_center_from_index(idx, grid_bounds, grid_shape) for idx in picked], picked

# ============================================================
#  Coverage algorithm
# ============================================================
def coverage_algorithm(model_jax, params, grid_bounds, grid_shape,
                       batch_size=10, T_total=15.0, dt_chunk=3.0, dt=0.05,
                       consec_thresh=3, max_iters=10000):
    rhs = make_numpy_rhs_from_jax(model_jax, params)
    visited_set = set()
    total_cells = np.prod(grid_shape)
    iters = 0
    all_iterations = []
    coverage_history = []

    while len(visited_set) < total_cells and iters < max_iters:
        iters += 1
        iteration_trajectories = []
        frontier = compute_frontier(visited_set, grid_shape)
        IC_coords, IC_indices = sample_initial_conditions_from_frontier(frontier, batch_size, grid_bounds, grid_shape)

        for ic, ic_idx in zip(IC_coords, IC_indices):
            if ic_idx in visited_set:
                continue
            times, states, entered_visited, exit_boundary = simulate_until_enter_or_exit(
                ic, T_total, dt_chunk, dt, rhs,
                index_from_point, consec_thresh,
                visited_set, grid_bounds, grid_shape
            )
            new_adds = update_visited_from_states(states, visited_set, index_from_point, grid_bounds, grid_shape)
            iteration_trajectories.append({
                "ic": ic, 
                "times": times, 
                "states": states,
                "entered_visited": entered_visited,
                "exit_boundary": exit_boundary,
                "new_cells": new_adds
            })

        visited_now = len(visited_set)
        coverage_history.append(visited_now)
        all_iterations.append({
            "iteration": iters,
            "visited_now": visited_now,
            "trajectories": iteration_trajectories
        })
        print(f"Iteration {iters}: visited {visited_now}/{total_cells} cells")

        # ============ Plot for this iteration ============
        fig, ax = plt.subplots(figsize=(5,5))
        grid_mask = np.zeros(grid_shape, dtype=int)
        for idx in visited_set:
            grid_mask[idx] = 1
        ax.imshow(grid_mask.T, origin="lower", extent=(
            grid_bounds[0][0], grid_bounds[0][1],
            grid_bounds[1][0], grid_bounds[1][1]),
            cmap="Greens", alpha=0.6, aspect="auto"
        )
        for traj in iteration_trajectories:
            states = traj["states"]
            if len(states) > 0:
                ax.plot(states[:,0], states[:,1], lw=1, color="black")
        ax.set_title(f"Iteration {iters} - Visited {visited_now}")
        plt.show()

    return visited_set, all_iterations, coverage_history

# ============================================================
#  Run example with Lotka–Volterra
# ============================================================

grid_bounds = [(0.0, 5.0), (0.0, 5.0)]
grid_shape = (25, 25)
params = (2.0, 1.0, 0.5, 1.0)

visited, all_iterations, coverage_history = coverage_algorithm(
    lotka_volterra, params, grid_bounds, grid_shape,
    batch_size=8, T_total=15.0, dt_chunk=3.0, dt=0.05,
    consec_thresh=3
)

print(f"Final coverage: {len(visited)}/{np.prod(grid_shape)} cells")

# ========================================================
#  Plot convergence curve
# ========================================================
plt.figure(figsize=(6,4))
plt.plot(range(1,len(coverage_history)+1), coverage_history, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Visited cells")
plt.title("Coverage convergence")
plt.grid(True)
plt.show()
