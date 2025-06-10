import numpy as np
import math
from scipy.stats import qmc

# EVOLVE-BLOCK-START

"""Finding optimal configuration of 540 points in a square that mimizes the L-infinity star discrepancy."""

def construct_star() -> np.ndarray:
    """
    Find the optimal configuration of points in a square [0, 1] x [0, 1]
    to minimize the L-infinity star discrepancy.
    This function constructs 540 points that are evenly distributed.
    The points are represented in coordinates (x, y) in the square
    x and y are in the range [0, 1].
    Returns:
        A: np.array of shape (540, 2) with coordinates of points in the square.
        x and y coordinates are in the range [0, 1].
    """
    N = 540
    A = np.zeros((N, 2))
    
    # Optimized LDS construction using a different irrational number and offset
    phi = 0.6180339887498949  # Pre-calculated golden ratio conjugate

    # Optimized vectorization for faster computation
    x = (np.arange(N) + 0.5) / N
    y = (phi * (np.arange(N) + 0.5)) % 1.0

    # Introduce a small perturbation to y coordinates using a Halton sequence
    # This can help to further reduce discrepancy in some cases
    def halton(index, base):
        result = 0
        fractional_part = 1 / base
        i = index
        while i > 0:
            result += fractional_part * (i % base)
            i //= base
            fractional_part /= base
        return result

    # Base 3 Halton sequence with a smaller scaling factor for perturbation
    halton_sequence = np.array([halton(i, 3) for i in range(N)])
    perturbation = 0.00005 * (halton_sequence - 0.5)  # Further reduced perturbation scale
    y = (y + perturbation) % 1.0 # Keep values within [0, 1]
    
    # Further refinement: Apply a tent map transformation to y to improve distribution
    y = np.where(y < 0.5, 2 * y, 2 - 2 * y)
    
    A[:, 0] = x
    A[:, 1] = y
    
    return A

# EVOLVE-BLOCK-END

import numpy as np
import itertools
from numba import njit
# This part remains fixed (not evolved)
def run_star() -> np.ndarray:
    """Run the star constructor for n=540"""
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
    for i_point in range(N_arg):
        point_is_in_box = True
        for k_dim in range(D_arg):
            if points_X_arg[i_point, k_dim] > y_corner_arg[k_dim]:
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
    N, D = points_X.shape
    max_discrepancy_val = 0.0

    # Use more refined grid lines for better accuracy
    num_grid_lines = int(np.sqrt(N))  # Scale grid lines with N
    grid_lines_x = np.linspace(0, 1, num_grid_lines + 1)
    grid_lines_y = np.linspace(0, 1, num_grid_lines + 1)
    
    # Iterate through all grid line combinations
    for x_corner in grid_lines_x:
        for y_corner in grid_lines_y:
            y_corner_arg = np.array([x_corner, y_corner], dtype=np.float64)
            local_discrepancy = _calculate_single_box_discrepancy_numba(
                points_X.astype(np.float64), N, D, y_corner_arg
            )
            max_discrepancy_val = max(max_discrepancy_val, local_discrepancy)

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
    np.random.seed(42) # Set seed for reproducibility
    points = run_star()
    score = score_star(points)
    print("Score:", score)