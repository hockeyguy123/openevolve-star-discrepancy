"""
Evaluator for the star discrepancy problem
"""
import os
import subprocess
import tempfile
import traceback
import sys
import pickle
import math
import time

import ctypes
import shutil
from openevolve.evaluation_result import EvaluationResult

from numba import njit

from typing import List, Dict, Any
import numpy as np
from scipy.stats import norm

np.random.seed(42)  # For reproducibility
class DimensionParameters(ctypes.Structure):
    _fields_ = [
        ("s", ctypes.c_int),
        ("a", ctypes.c_uint32),
        ("m_i", ctypes.c_uint32 * 30) # Array of uint32_t
    ]


def get_sobol_points_cpp(input_params, n_points, n_dimensions, ltm, shifts):
    """
    Generates Sobol points using the C++ shared library.

    Args:
        input_params (List[Dict[str, Any]]): List of dictionaries containing
            Sobol sequence parameters for each dimension. Each dict must contain:
            - 's': int, the degree of the polynomial (1 <= s <= 32)
            - 'a': int, the coefficients of the polynomial (0 <= a < 2^(s-1))
            - 'm_i': List[int], the direction numbers of length s (1 <= m_i < 2^s, odd values only).
            Example: [{'s': 3, 'a': 1, 'm_i': [1, 3, 5]}, ...]
        n_points (int): Number of points to generate.
        n_dimensions (int): Number of dimensions for each point.
        scramble_masks_np (np.ndarray, optional): A 1D NumPy array of uint32
            scrambling masks, one for each dimension. If None, random masks
            will be generated.
    Returns:
        np.ndarray: A 2D NumPy array of shape (n_points, n_dimensions)
                    containing the Sobol points.
    """

    lib_path = "/projects/bdln/asadikov/openevolve/examples/qmc/sobol_generator.so"
    sobol_lib = ctypes.CDLL(lib_path)

    # Define argument types and return type for the C++ function
    sobol_lib.generate_sobol_points.argtypes = [
        ctypes.c_int,                                 # n_points
        ctypes.c_int,                                 # n_dimensions
        ctypes.POINTER(DimensionParameters),          # input_sobol_params
        ctypes.POINTER(ctypes.c_double),              # output_points
        ctypes.POINTER(ctypes.c_uint32), # ltm_elements_flat (raw 0s/1s)
        ctypes.POINTER(ctypes.c_uint32)  # digital_shifts
    ]
    # Define result type (void in C++, so None in Python or not set)
    sobol_lib.generate_sobol_points.restype = None # or ctypes.c_void_p for void functions

    ParamsArrayType = DimensionParameters * (n_dimensions - 1)
    ctypes_params_array = ParamsArrayType()

    LTMElementsFlatTypePy = ctypes.c_uint32 * ltm.size
    ctypes_ltm = LTMElementsFlatTypePy.from_buffer(ltm) # More direct

    DigitalShiftsArrayTypePy = ctypes.c_uint32 * n_dimensions
    ctypes_shift = DigitalShiftsArrayTypePy.from_buffer(shifts)

    for i, py_param in enumerate(input_params): 
        if not isinstance(py_param, dict):
            raise TypeError(f"Parameter for dimension {i+1} must be a dictionary.")
        s_val = py_param.get('s')
        a_val = py_param.get('a')
        m_i_list = py_param.get('m_i')

        ctypes_params_array[i].s = s_val
        ctypes_params_array[i].a = ctypes.c_uint32(a_val)
        for j in range(s_val):
            ctypes_params_array[i].m_i[j] = ctypes.c_uint32(m_i_list[j])
        for j in range(s_val, 30):
             ctypes_params_array[i].m_i[j] = 0


    # 2. Prepare output array (ctypes double array)
    OutputArrayType = ctypes.c_double * (n_points * n_dimensions)
    ctypes_output_points = OutputArrayType()

    sobol_lib.generate_sobol_points(
        n_points,
        n_dimensions,
        ctypes_params_array,
        ctypes_output_points,
        ctypes_ltm,
        ctypes_shift
    )
    np_output_points = np.ctypeslib.as_array(ctypes_output_points)
    np_output_points = np_output_points.reshape((n_points, n_dimensions))
    return np_output_points

def asian_option_pricer(s0, k, t, r, sigma, d, qmc_points):
    """
    Prices an Asian call option using Quasi-Monte Carlo.

    Args:
        s0 (float): Initial asset price.
        k (float): Strike price.
        t (float): Time to expiration (in years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        d (int): Number of observation times (dimensions).
        qmc_points (np.ndarray): Array of QMC points of shape (n_samples, d).
                                 Each point is in [0, 1]^d.

    Returns:
        float: Estimated Asian call option price.
    """
    normal_samples = norm.ppf(qmc_points)
    dt = t / d
    brownian_path_increments = np.sqrt(dt) * normal_samples
    brownian_paths = np.cumsum(brownian_path_increments, axis=1)
    time_steps = np.linspace(dt, t, d)
    asset_paths = s0 * np.exp((r - 0.5 * sigma**2) * time_steps + sigma * brownian_paths)
    average_asset_prices = np.mean(asset_paths, axis=1)
    payoffs = np.maximum(average_asset_prices - k, 0)
    option_price = np.exp(-r * t) * np.mean(payoffs)

    return option_price


def score_integral(A):
    S0 = 50.0       # Initial asset price
    K_strike = 45.0 # Strike price
    T_exp = 1.0     # Time to expiration (1 year)
    R_rate = 0.05   # Risk-free rate
    SIGMA = 0.3     # Volatility
    D_dims = 32     # Number of observation times (dimensions)
    # True value of the option from the paper's SI (page 8)
    C0_true = 7.064526215280412
    # Number of QMC points to test (as in Table 2 of the main paper)
    # N_points_array = [32, 64, 128, 256, 512, 1024, 2048]

    n_points = 8192  # Number of QMC points to use for the estimation
    total_err = 0.0
    powers_of_two = 2 ** np.arange(30, dtype=np.uint32)
    indices = np.arange(30)
    for i in range(1000):
        rng = np.random.default_rng(seed=i)
        digital_shifts_py = rng.integers(2, size=(32, 30), dtype=np.uint32) @ powers_of_two
        ltm_py = np.tril(rng.integers(2, size=(32, 30, 30), dtype=np.uint32))
        ltm_py[:, indices, indices] = 1  # Set diagonal to 1
        # for d_idx_ltm in range(32):
        #     for i_idx_ltm in range(30):
        #         ltm_py[d_idx_ltm, i_idx_ltm, i_idx_ltm] = 1

        optimal_points_set = get_sobol_points_cpp(A, n_points, D_dims, ltm_py, digital_shifts_py)
        estimated_price_optimal = asian_option_pricer(S0, K_strike, T_exp, R_rate, SIGMA, D_dims, optimal_points_set)
        total_err += (estimated_price_optimal - C0_true) ** 2

    return 1 / (1 + total_err * 10)


class TimeoutError(Exception):
    pass


def run_with_timeout(program_path: str, timeout_seconds: int) -> List[Dict[str, Any]]:
    """
    Run the program in a separate process with timeout
    using a simple subprocess approach

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds

    Returns:
        List[Dict[str, Any]]: The result of the program execution, which should be a list of dictionaries
    """
    # Create a temporary file to execute
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        # Write a script that executes the program and saves results

        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname('{program_path}'))

# Debugging info
print(f"Running in subprocess, Python version: {{sys.version}}")
print(f"Program path: {program_path}")

try:
    # Import the program
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    # Run the star discrepancy function
    print("Calling construct_sobol_sequence()...")
    A = program.construct_sobol_sequence()

    # Save results to a file
    results = {{
        'ret': A    
    }}

    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {temp_file.name}.results")
    
except Exception as e:
    # If an error occurs, save the error instead
    print(f"Error in subprocess: {{str(e)}}")
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
    print(f"Error saved to {temp_file.name}.results")
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"
    # get path to temporary file directory

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

                return results["ret"]
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
    Evaluate the program by running it once and checking the sum of radii

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    try:
        # Use subprocess to run with timeout
        a = time.time()
        A = run_with_timeout(
            program_path, timeout_seconds=600  # Single timeout
        )
        b = time.time()
        print(f"Program executed in {b - a:.2f} seconds")
        # artifacts = {"val_report": "Sobol Sequence is valid"}
        metrics = {"score": 0.0, "validity": 0.0}
        if len(A) != 31:
            print(f"Invalid length for Sobol Sequence:" + str(len(A)) + ", expected: 31")
        for entry in A:
            if "s" not in entry:
                print("Missing 's' in Sobol sequence entry:", entry)
                return metrics
            if "a" not in entry:
                print("Missing 'a' in Sobol sequence entry:", entry)
                return metrics
            if "m_i" not in entry:
                print("Missing 'm_i' in Sobol sequence entry:", entry)
                return metrics
            if entry["s"] < 1 or entry["s"] > 30:
                print(f"Invalid 's' value in Sobol sequence entry:" + str(entry['s']) + ", expected: 1-30")
                return metrics
            if entry["a"] < 0 or entry["a"] >= (1 << (entry["s"] - 1)):
                print(f"Invalid 'a' value in Sobol sequence entry:" + str(entry['a']) + ", expected: 0 <= a < 2^(s-1)")
                return metrics
            if len(entry["m_i"]) != entry["s"]:
                print(f"Invalid 'm_i' length in Sobol sequence entry:" + str(len(entry["m_i"])) + ", expected: s")
                return metrics
            for m in entry["m_i"]:
                if m < 1 or m >= (1 << entry["s"]):
                    print(f"Invalid 'm_i' value in Sobol sequence entry:" + str(m) + ", expected: 1 <= m_i < 2^s")
                    return metrics
                if m % 2 == 0:
                    print(f"Invalid 'm_i' value in Sobol sequence entry:" + str(m) + ", expected: odd values only")
                    return metrics
        # Calculate the Sobol scores
        sobol_score = float(score_integral(A))
        return {"score": sobol_score, "validity": 1.0}
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        traceback.print_exc()
        return {"score": 0.0, "validity": 0.0}


if __name__ == "__main__":
    # Example usage
    program_path = "openevolve-old/examples/qmc/initial_program.py"
    if not os.path.exists(program_path):
        print(f"Program file {program_path} does not exist.")
        sys.exit(1)

    import time
    start_time = time.time()
    result = evaluate(program_path)
    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds")
    print("Evaluation result:", result)