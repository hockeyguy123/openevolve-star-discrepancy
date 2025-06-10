import numpy as np
import math

# EVOLVE-BLOCK-START

"""Finding optimal configuration of 820 points in a square that mimizes the L-infinity star discrepancy."""

def construct_star() -> np.ndarray:
    """
    Find the optimal configuration of points in a square [0, 1] x [0, 1]
    to minimize the L-infinity star discrepancy.
    This function constructs 820 points that are evenly distributed.
    The points are represented in coordinates (x, y) in the square
    x and y are in the range [0, 1].
    Returns:
        A: np.array of shape (820, 2) with coordinates of points in the square.
        x and y coordinates are in the range [0, 1].
    """
    N = 820
    A = np.zeros((N, 2))

    # Optimized stratified sampling and golden ratio sequence
    x = (np.arange(N) + 0.5) / N
    golden_ratio = (math.sqrt(5) - 1) / 2
    y = (np.arange(N) * golden_ratio) % 1

    # Apply a smaller perturbation to avoid exact alignment on grid lines.
    rng = np.random.default_rng(seed=42)  # For reproducibility
    perturbation = rng.uniform(-0.02 / N, 0.02 / N, size=N)  # Reduced perturbation

    # Tent map transformation with adjusted parameters.
    y_transformed = np.where(y < 0.5, 2 * y, 2 * (1 - y))

    # Combine perturbation and tent map, and clip
    A[:, 0] = np.clip(x + perturbation, 0.0, 1.0)
    A[:, 1] = np.clip(y_transformed + perturbation, 0.0, 1.0)

    return A

# EVOLVE-BLOCK-END

import numpy as np
import itertools
from numba import njit
# This part remains fixed (not evolved)
def run_star() -> np.ndarray:
    """Run the star constructor for n=820"""
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

    return max(discrepancy_val, 0.0)


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