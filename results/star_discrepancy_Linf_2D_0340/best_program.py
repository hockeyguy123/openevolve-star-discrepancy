import numpy as np
import math
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

"""Finding optimal configuration of 340 points in a square that mimizes the L-infinity star discrepancy."""

def construct_star() -> np.ndarray:
    """
    Find the optimal configuration of points in a square [0, 1] x [0, 1]
    to minimize the L-infinity star discrepancy.
    This function constructs 340 points that are evenly distributed.
    The points are represented in coordinates (x, y) in the square
    x and y are in the range [0, 1].
    Returns:
        A: np.array of shape (340, 2) with coordinates of points in the square.
        x and y coordinates are in the range [0, 1].
    """
    N = 340
    
    def discrepancy_loss(x):
        points = x.reshape((N, 2))
        return star_discrepancy(points)

    # Initial guess using a combination of stratified sampling and golden ratio
    initial_points = np.zeros((N, 2))
    golden_ratio = (1 + math.sqrt(5)) / 2
    for i in range(N):
        initial_points[i, 0] = (i + 0.5) / N
        initial_points[i, 1] = ((i * golden_ratio) % 1 + (0.5/N)) % 1

    initial_guess = initial_points.flatten()
    
    # Define bounds for each coordinate (0 <= x, y <= 1)
    bounds = [(0.0, 1.0) for _ in range(2 * N)]

    # Optimization using SLSQP
    result = minimize(discrepancy_loss, initial_guess, method='SLSQP', bounds=bounds, options={'maxiter': 500})  # Increased maxiter

    A = result.x.reshape((N, 2))
    return A

# EVOLVE-BLOCK-END

import numpy as np
import itertools
from numba import njit
# This part remains fixed (not evolved)
def run_star() -> np.ndarray:
    """Run the star constructor for n=340"""
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
    max_discrepancy = 0.0

    # Create grid lines from the point coordinates
    grid_lines = [np.unique(points_X[:, i]) for i in range(D)]
    grid_lines = [np.concatenate((g, [1.0])) for g in grid_lines]

    # Iterate through all possible box corners
    for corner in itertools.product(*grid_lines):
        corner = np.array(corner)
        discrepancy = _calculate_single_box_discrepancy_numba(points_X, N, D, corner)
        max_discrepancy = max(max_discrepancy, discrepancy)

    return max_discrepancy


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