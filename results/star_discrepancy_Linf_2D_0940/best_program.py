import numpy as np
from scipy.stats import qmc

# EVOLVE-BLOCK-START

"""Finding optimal configuration of 940 points in a square that mimizes the L-infinity star discrepancy."""

def construct_star() -> np.ndarray:
    """
    Find the optimal configuration of points in a square [0, 1] x [0, 1]
    to minimize the L-infinity star discrepancy.
    This function constructs 940 points that are evenly distributed.
    The points are represented in coordinates (x, y) in the square
    x and y are in the range [0, 1].
    Returns:
        A: np.array of shape (940, 2) with coordinates of points in the square.
        x and y coordinates are in the range [0, 1].
    """
    N = 940
    A = np.zeros((N, 2))

    # Optimized Golden Ratio sequence using vectorization
    n = np.arange(N)
    golden_ratio = 0.6180339887498949  # Pre-calculated golden ratio

    # Stratified sampling with a shift and golden ratio
    A[:, 0] = ((n + golden_ratio) / N) % 1  # Shifted stratified sampling in x
    A[:, 1] = ((n * golden_ratio) % 1 + (0.5 / N)) % 1  # Golden ratio with offset in y

    # Refinement using a different perturbation approach based on sin and cos, with reduced strength
    perturbation_strength = 0.0015 / N  # Further reduced perturbation strength, tuned down
    A[:, 0] += perturbation_strength * np.cos(2 * np.pi * n * golden_ratio)
    A[:, 1] += perturbation_strength * np.sin(2 * np.pi * n * golden_ratio)

    # Apply a different perturbation based on a power of n
    power_perturbation_strength = 0.0004 / N  # Strength for the power perturbation, tuned down
    A[:, 0] += power_perturbation_strength * (n / N)**2 * np.cos(2 * np.pi * n * golden_ratio)
    A[:, 1] += power_perturbation_strength * (n / N)**2 * np.sin(2 * np.pi * n * golden_ratio)

    # Clip values to ensure they remain within [0, 1]
    A = np.clip(A, 0.0, 1.0)

    # Additional perturbation using square root, reduced strength and different frequency
    sqrt_perturbation_strength = 0.00025 / N
    A[:, 0] += sqrt_perturbation_strength * np.sqrt(n / N) * np.cos(6 * np.pi * n * golden_ratio)
    A[:, 1] += sqrt_perturbation_strength * np.sqrt(n / N) * np.sin(6 * np.pi * n * golden_ratio)

    # Clip values to ensure they remain within [0, 1]
    A = np.clip(A, 0.0, 1.0)

    # Further refinement with a smaller perturbation and different frequency
    final_perturbation_strength = 0.00008 / N  # Tuned down
    A[:, 0] += final_perturbation_strength * np.cos(8 * np.pi * n * golden_ratio)
    A[:, 1] += final_perturbation_strength * np.sin(8 * np.pi * n * golden_ratio)

    A = np.clip(A, 0.0, 1.0)

    # Optimized Halton sequence generation and combination
    halton_engine = qmc.Halton(d=2, seed=12345)
    halton_points = halton_engine.random(n=N)
    
    halton_influence = 0.000055  # Reduced Halton influence
    A = np.clip(A + halton_influence * halton_points, 0.0, 1.0)

    # Add Sobol sequence with a very small influence
    sobol_engine = qmc.Sobol(d=2, seed=54321)
    sobol_points = sobol_engine.random(n=N)
    sobol_influence = 0.000035
    A = np.clip(A + sobol_influence * sobol_points, 0.0, 1.0)

    return A

# EVOLVE-BLOCK-END

import numpy as np
import itertools
from numba import njit
# This part remains fixed (not evolved)
def run_star() -> np.ndarray:
    """Run the star constructor for n=940"""
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
    count_in_box = 0
    for i_point in range(N_arg): # Iterate through each point
        point_is_in_box = True
        for k_dim in range(D_arg): # Iterate through each dimension for the current point
            if points_X_arg[i_point, k_dim] > y_corner_arg[k_dim]: # Point is outside this dimension
                point_is_in_box = False
                break
            
        if point_is_in_box:
            count_in_box += 1

    return abs(count_in_box / N_arg - volume)

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

    # Using a fixed grid size for faster computation
    num_grid_lines = 30  # Reduced number of grid lines
    grid_lines_per_dim = [np.linspace(0.0, 1.0, num_grid_lines) for _ in range(D)]

    max_discrepancy_val = 0.0
    
    y_corner_for_numba = np.empty(D, dtype=points_X_clipped.dtype)

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
