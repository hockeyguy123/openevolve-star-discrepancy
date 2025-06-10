import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

"""Finding optimal configuration of 14 points in a square that mimizes the L-infinity star discrepancy."""

def construct_star() -> np.ndarray:
    """
    Find the optimal configuration of points in a square [0, 1] x [0, 1]
    to minimize the L-infinity star discrepancy.
    This function constructs 14 points that are evenly distributed.
    The points are represented in coordinates (x, y) in the square
    x and y are in the range [0, 1].
    Returns:
        A: np.array of shape (14, 2) with coordinates of points in the square.
        x and y coordinates are in the range [0, 1].
    """
    n_points = 14

    # Improved initialization using a Sobol sequence (approximation)
    sobol_x = [(2*i - 1) / (2 * n_points) for i in range(1, n_points + 1)]
    sobol_y = []
    for i in range(1, n_points + 1):
        index = i
        fraction = 0
        base = 3  # Using base 3 for the second dimension
        k = 1
        while index > 0:
            fraction += (index % base) / (base**k)
            index //= base
            k += 1
        sobol_y.append(fraction)
    initial_points = np.array(list(zip(sobol_x, sobol_y)))
    
    # Objective function (negative star discrepancy score)
    def objective_function(x):
        points = x.reshape(n_points, 2)
        return -score_star(points)

    # Bounds for point coordinates
    bounds = [(0.0, 1.0)] * (n_points * 2)

    # Optimization parameters
    num_restarts = 25  # Increased restarts to explore more local minima
    max_iter = 17500   # Increased iterations for better convergence
    tolerance = 1e-12  # Tightened tolerance for higher precision

    best_points = None
    best_score = -np.inf

    for i in range(num_restarts):
        # Perturb initial points with a smaller standard deviation
        initial_points_perturbed = initial_points + np.random.normal(0, 0.00075, size=(n_points, 2))  # Even smaller noise
        initial_points_perturbed = np.clip(initial_points_perturbed, 0.0, 1.0)

        # SLSQP optimization
        result = minimize(objective_function, initial_points_perturbed.flatten(), method='SLSQP',
                          bounds=bounds, options={'maxiter': max_iter, 'ftol': tolerance, 'disp': False})

        # Evaluate and update best result
        optimized_points = result.x.reshape(n_points, 2)
        current_score = score_star(optimized_points)

        if current_score > best_score:
            best_score = current_score
            best_points = optimized_points

    return best_points

# EVOLVE-BLOCK-END

import numpy as np
import itertools
from numba import njit
# This part remains fixed (not evolved)
def run_star() -> np.ndarray:
    """Run the star constructor for n=14"""
    A = construct_star()
    return A
# Numba helper function for calculating discrepancy for a single box corner
@njit(cache=True)
def _calculate_single_box_discrepancy_numba(points_X_arg: np.ndarray, 
                                           N_arg: int, 
                                           D_arg: int, 
                                           y_corner_arg: np.ndarray) -> float:
    """
    Calculates the local discrepancy for a single d-dimensional anchored box.
    Box is defined by [0, y_corner_arg[0]] x ... x [0, y_corner_arg[D-1]].
    """
    # Calculate volume of the box
    volume = 1.0
    for k_dim in range(D_arg):
        volume *= y_corner_arg[k_dim]

    # Count points within the box [0, y_corner_arg]
    # The original logic `points_X[None, :] <= y_corners[:, None, :]`
    # effectively means a point is counted if point_coord <= corner_coord for all dimensions.
    count_in_box = 0
    count_on_line = 0
    for i_point in range(N_arg): # Iterate through each point
        point_is_in_box = True
        point_is_on_line = False
        for k_dim in range(D_arg): # Iterate through each dimension for the current point
            if points_X_arg[i_point, k_dim] > y_corner_arg[k_dim]: # Point is outside this dimension
                point_is_in_box = False
                break
            elif points_X_arg[i_point, k_dim] == y_corner_arg[k_dim]:
                point_is_on_line = True
            
        if point_is_in_box:
            count_in_box += 1
            if point_is_on_line:
                count_on_line += 1

    return max(abs(count_in_box / N_arg - volume), abs((count_in_box - count_on_line) / N_arg - volume))

def star_discrepancy(points_X: np.ndarray) -> float:
    """
    Calculates the L-infinity star discrepancy of the point set P.
    Optimized using Numba for the core calculation loop.
    Args:
        points_X (np.ndarray): An array of points to evaluate, shape (N, D) for D-dimensional points.
    Returns:
        float: The maximum star discrepancy value.
    """
    # Input validation and preparation
    if not isinstance(points_X, np.ndarray):
        points_X_np = np.array(points_X, dtype=np.float64)
    elif points_X.dtype != np.float64: # Ensure float64 for Numba compatibility and precision
        points_X_np = points_X.astype(np.float64)
    else:
        points_X_np = points_X

    if points_X_np.ndim == 1:
        points_X_np = points_X_np.reshape(-1, 1)
    
    N, D = points_X_np.shape

    if N == 0:
        return 1.0

    points_X_clipped = np.clip(points_X_np, 0.0, 1.0)
    
    if not points_X_clipped.flags.c_contiguous:
        points_X_clipped = np.ascontiguousarray(points_X_clipped)

    grid_lines_per_dim = []
    for j in range(D):
        unique_coords_dim_j = np.unique(points_X_clipped[:, j])
        current_dim_grid_lines = np.union1d(unique_coords_dim_j, 
                                            np.array([1.0], dtype=points_X_clipped.dtype))
        grid_lines_per_dim.append(current_dim_grid_lines)

    max_discrepancy_val = 0.0
    
    y_corner_for_numba = np.empty(D, dtype=points_X_clipped.dtype)

    if not all(len(gl) > 0 for gl in grid_lines_per_dim):
        max_discrepancy_val = 0.0
    else:
        for y_corner_tuple in itertools.product(*grid_lines_per_dim):
            for i_val in range(D):
                y_corner_for_numba[i_val] = y_corner_tuple[i_val]
            
            local_discrepancy = _calculate_single_box_discrepancy_numba(
                points_X_clipped, N, D, y_corner_for_numba
            )
            
            if local_discrepancy > max_discrepancy_val:
                max_discrepancy_val = local_discrepancy

    return max_discrepancy_val


def score_star(X_points: np.ndarray) -> float:
    """ Calculates the score based on the star discrepancy of the given points.
    Args:
        X_points (np.ndarray): An array of points to evaluate, shape (N, 2) for 2D points.
    Returns:
        float: The score based on the star discrepancy, defined as 1 / (1 + max_discrepancy_val).
    """
    discrepancy = star_discrepancy(X_points)
    return 1 / (1 + discrepancy)  # Return the score as per the definition of star discrepancy



if __name__ == "__main__":
    # Example usage
    points = run_star()
    score = score_star(points)
    print("Score:", score)