"""
Evaluator for the star discrepancy problem
"""
import numpy as np
import os
import itertools
from numba import njit
import matplotlib.pyplot as plt


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
    Calculates a score based on the L-infinity star discrepancy of the point set P.
    Optimized using Numba for the core calculation loop.
    The score is 1 / (1 + max_discrepancy_val).
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


def score_star(points_to_evaluate_list: np.ndarray) -> float:
    discrepancy = star_discrepancy(points_to_evaluate_list)
    return 1 / (1 + discrepancy)  # Return the score as per the definition of star discrepancy


def visualize_2D(points: np.ndarray, output_path: str, title: str = "Star Discrepancy Points") -> None:
    """
    Visualizes the points in 2D and saves the plot to the specified output path.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points[:, 0], points[:, 1], s=10, color='blue', marker='*')
    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=300, pad_inches=0.05)
    plt.close(fig)


def visualize_3D(points: np.ndarray, output_path: str, title: str = "Star Discrepancy Points") -> None:
    """
    Visualizes the points in 3D and saves the plot to the specified output path.
    """

    if points.ndim == 1:
        points = points.reshape(-1, 3)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].scatter(points[:, 0], points[:, 1], s=10, color='blue', marker='*')
    ax[0].set_xlabel('X-axis')
    ax[0].set_ylabel('Y-axis')
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].grid(True)
    ax[1].scatter(points[:, 0], points[:, 2], s=10, color='blue', marker='*')
    ax[1].set_title(title)
    ax[1].set_xlabel('X-axis')
    ax[1].set_ylabel('Z-axis')
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[1].grid(True)
    ax[2].scatter(points[:, 1], points[:, 2], s=10, color='blue', marker='*')
    ax[2].set_xlabel('Y-axis')
    ax[2].set_ylabel('Z-axis')
    ax[2].set_xlim(0, 1)
    ax[2].set_ylim(0, 1)
    ax[2].grid(True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=300, pad_inches=0.05)
    plt.close(fig)





if __name__ == "__main__":
    results_path = os.path.join(os.path.dirname(__file__))
    numbers = list(range(2, 21, 1))
    numbers += [30, 40, 50, 60, 80, 100]
    numbers += list(range(140, 1060, 40))


    for i in numbers:
        best_points = np.loadtxt(os.path.join(results_path, f"star_discrepancy_Linf_2D_{i:04d}/best_points.txt"), delimiter=',')
        star_disc = star_discrepancy(best_points)
        print(f"Star Discrepancy for 2D, {i} points: {star_disc}")

        visualize_2D(best_points, 
                   os.path.join(results_path, f"star_discrepancy_Linf_2D_{i:04d}/best_points_visualization.png"),
                   title=f"Openevolve 2D {i} Points Star Discrepancy {star_disc:.4f}")
    
    # numbers = list(range(1, 17, 1))

    # for i in numbers:
    #     best_points = np.loadtxt(os.path.join(results_path, f"star_discrepancy_Linf_3D_{i:04d}/best_points.txt"), delimiter=',')
    #     star_disc = star_discrepancy(best_points)
    #     print(f"Star Discrepancy for 3D, {i} points: {star_disc}")
    #     visualize_3D(best_points, 
    #                os.path.join(results_path, f"star_discrepancy_Linf_3D_{i:03d}/best_points_visualization.png"),
    #                title=f"Openevolve 3D {i} Points Star Discrepancy {star_disc:.4f}")
