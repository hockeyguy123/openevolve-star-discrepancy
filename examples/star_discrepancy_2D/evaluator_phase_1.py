"""
Evaluator for the star discrepancy problem
Has timeout of 60 seconds for the program to run
"""
import numpy as np
import os
import subprocess
import tempfile
import traceback
import sys
import pickle
import itertools
import itertools
from numba import njit


class TimeoutError(Exception):
    pass

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


def run_with_timeout(program_path: str, timeout_seconds: int) -> np.ndarray:
    """
    Run the program in a separate process with timeout
    using a simple subprocess approach

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds

    Returns:
        matrix from the program
    """
    # Create a temporary file to execute
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        script = """
import sys
import numpy as np
import os
import pickle
import traceback

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname('""" + program_path + """'))

# Debugging info
try:
    # Import the program
    spec = __import__('importlib.util').util.spec_from_file_location("program", '""" + program_path + """')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)
    import numpy as np
    np.random.seed(42)  # Set a fixed seed for reproducibility
    # Run the star discrepancy function
    A = program.run_star()

    # Save results to a file
    results = {
        'matrix': A    
    }

    with open('""" + temp_file.name + """.results', 'wb') as f:
        pickle.dump(results, f)
    
except Exception as e:
    # If an error occurs, save the error instead
    print("Error in subprocess:" + str(e))
    traceback.print_exc()
    with open('""" + temp_file.name + """.results', 'wb') as f:
        pickle.dump({'error': str(e)}, f)
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        # Run the script with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            # Always print output for debugging purposes
            print(f"Subprocess stdout: {stdout.decode()}")
            if stderr:
                print(f"Subprocess stderr: {stderr.decode()}")

            # Still raise an error for non-zero exit codes, but only after printing the output
            if exit_code != 0:
                raise RuntimeError(f"Process exited with code {exit_code}")

            # Load the results
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                # Check if an error was returned
                if "error" in results:
                    raise RuntimeError(f"Program execution failed: {results['error']}")

                return results["matrix"]
            else:
                raise RuntimeError("Results file not found")

        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


def evaluate(program_path):
    """
    Evaluate the program by running it once and checking the star discrepancy.

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    try:
        # Use subprocess to run with timeout
        A = run_with_timeout(
            program_path, timeout_seconds=60  # Single timeout
        )

        # Ensure matrix is a numpy array
        if not isinstance(A, np.ndarray):
            A = np.array(A)
        A = A.astype(np.float64)  # Ensure float64 for precision
        if A.shape != (16, 2):
            print("Invalid shape for coordinate matrix: " +  str(A.shape) + ", expected (16, 2)")
            return {
                "score": 0.0,
                "validity": 0.0,
            }
        
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)  # Handle NaNs and Infs
        score = score_star(A)

        return {
            "score": float(score),
            "validity": 1.0
        }

    except Exception as e:
        print("Evaluation failed completely:" + str(e))
        traceback.print_exc()
        return {
            "score": 0.0,
            "validity": 0.0
        }


if __name__ == "__main__":
    # Example usage
    program_path = "openevolve-star-discrepancy/examples/star_discrepancy_2D/initial_program.py"
    if not os.path.exists(program_path):
        print(f"Program file {program_path} does not exist.")
        sys.exit(1)

    result = evaluate(program_path)
    print("Evaluation result:", result)