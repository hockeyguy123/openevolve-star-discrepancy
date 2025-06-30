import numpy as np
from scipy.stats import norm, qmc
import time

import numba

MACHINE_EPSILON = np.finfo(float).eps
import ctypes
from typing import List, Dict, Any

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


def construct_sobol_sequence():
    """
    Your task is minimize the integration error of an Asian call option price
    using Sobol sequences. The first three dimensions explain 97% of the variance
    and are by far the most important. 
    The Sobol sequence is defined by the parameters s: int, a: int, and m_i: List[int].
    Returns:
        List[Dict[str, Any]]: Sobol sequences. Each dict contains:
            - 's': int, the degree of the polynomial
            - 'a': int, the coefficients of the polynomial
            - 'm_i': List[int], the direction numbers of length s.
    1 <= s <= 32
    0 <= a < 2^(s-1)
    1 <= m_i < 2^i where i = 1, ..., s. Each m_i must be odd.
    """
    params = [
        {'s': 1, 'a': 0, 'm_i': [1]},  # Dimension 2
        {'s': 2, 'a': 1, 'm_i': [1, 3]},  # Dimension 3
        {'s': 3, 'a': 1, 'm_i': [1, 3, 5]},  # Dimension 4
        {'s': 3, 'a': 2, 'm_i': [1, 3, 7]},  # Dimension 5  Changed m_i[2] to 7
        {'s': 4, 'a': 1, 'm_i': [1, 1, 3, 7]},  # Dimension 6
        {'s': 4, 'a': 4, 'm_i': [1, 3, 5, 13]},  # Dimension 7
        {'s': 5, 'a': 2, 'm_i': [1, 1, 5, 5, 17]},  # Dimension 8
        {'s': 5, 'a': 4, 'm_i': [1, 1, 5, 5, 5]},  # Dimension 9
        {'s': 5, 'a': 7, 'm_i': [1, 1, 7, 11, 19]},  # Dimension 10
        {'s': 5, 'a': 11, 'm_i': [1, 1, 5, 1, 1]},  # Dimension 11
        {'s': 5, 'a': 13, 'm_i': [1, 1, 1, 3, 11]},  # Dimension 12
        {'s': 5, 'a': 14, 'm_i': [1, 3, 5, 5, 31]},  # Dimension 13
        {'s': 6, 'a': 1, 'm_i': [1, 3, 3, 9, 7, 49]},  # Dimension 14
        {'s': 6, 'a': 13, 'm_i': [1, 1, 1, 15, 21, 21]},  # Dimension 15
        {'s': 6, 'a': 16, 'm_i': [1, 3, 1, 13, 27, 49]},  # Dimension 16
        {'s': 6, 'a': 19, 'm_i': [1, 1, 1, 15, 7, 5]},  # Dimension 17
        {'s': 6, 'a': 22, 'm_i': [1, 3, 1, 15, 13, 25]},  # Dimension 18
        {'s': 6, 'a': 25, 'm_i': [1, 1, 5, 5, 19, 61]},  # Dimension 19
        {'s': 7, 'a': 1, 'm_i': [1, 3, 7, 11, 23, 15, 103]},  # Dimension 20
        {'s': 7, 'a': 4, 'm_i': [1, 3, 7, 13, 13, 15, 69]},  # Dimension 21
        {'s': 7, 'a': 7, 'm_i': [1, 1, 3, 13, 7, 35, 63]},  # Dimension 22
        {'s': 7, 'a': 8, 'm_i': [1, 3, 5, 9, 1, 25, 53]},  # Dimension 23
        {'s': 7, 'a': 14, 'm_i': [1, 3, 1, 13, 9, 35, 107]},  # Dimension 24
        {'s': 7, 'a': 19, 'm_i': [1, 3, 1, 5, 27, 61, 31]},  # Dimension 25
        {'s': 7, 'a': 21, 'm_i': [1, 1, 5, 11, 19, 41, 61]},  # Dimension 26
        {'s': 7, 'a': 28, 'm_i': [1, 3, 5, 3, 3, 13, 69]},  # Dimension 27
        {'s': 7, 'a': 31, 'm_i': [1, 1, 7, 13, 1, 19, 1]},  # Dimension 28
        {'s': 7, 'a': 32, 'm_i': [1, 3, 7, 5, 13, 19, 59]},  # Dimension 29
        {'s': 7, 'a': 37, 'm_i': [1, 1, 3, 9, 25, 29, 41]},  # Dimension 30
        {'s': 7, 'a': 41, 'm_i': [1, 3, 5, 13, 23, 1, 55]}, # Dimension 31
        {'s': 7, 'a': 42, 'm_i': [1, 3, 7, 3, 13, 59, 17]}  # Dimension 32
    ]
    return params


@numba.jit(nopython=True)
def polyfit_numba(x, y, deg):
    """A simple Numba-compatible implementation of polynomial fitting."""
    n = x.shape[0]
    X = np.ones((n, deg + 1))
    for i in range(1, deg + 1):
        X[:, i] = x**i
    
    # Using pinv for numerical stability, equivalent to np.linalg.lstsq
    coeffs = np.linalg.pinv(X) @ y
    return coeffs

@numba.jit(nopython=True)
def polyval_numba(coeffs, x):
    """A Numba-compatible implementation of polynomial evaluation."""
    deg = coeffs.shape[0] - 1
    y = np.zeros_like(x)
    for i in range(deg + 1):
        y += coeffs[i] * (x**i)
    return y


def bermudan_put_pricer_numba(s0, k, t, r, sigma, d, exercise_dates, qmc_points, poly_deg=3):
    """
    Prices a Bermudan put option using QMC and the Longstaff-Schwartz algorithm.
    Accelerated with Numba.
    """
    dt = t / d
    normal_samples = norm.ppf(qmc_points)
    brownian_path_increments = np.sqrt(dt) * normal_samples
    brownian_paths = np.cumsum(brownian_path_increments, axis=1)
    time_steps = np.linspace(dt, t, d)
    asset_paths = s0 * np.exp((r - 0.5 * sigma**2) * time_steps + sigma * brownian_paths)

    exercise_indices = np.array([int(date / dt) - 1 for date in exercise_dates], dtype=np.int64)
    cash_flows = np.maximum(k - asset_paths[:, -1], 0.0)

    for i in range(len(exercise_indices) - 2, -1, -1):
        idx = exercise_indices[i]
        discount_period = exercise_dates[i+1] - exercise_dates[i]
        cash_flows *= np.exp(-r * discount_period)

        in_the_money_mask = asset_paths[:, idx] < k
        exercise_value = np.maximum(k - asset_paths[:, idx], 0.0)
        
        X = asset_paths[in_the_money_mask, idx]
        Y = cash_flows[in_the_money_mask]

        if X.shape[0] > poly_deg:
            regression_coeffs = polyfit_numba(X, Y, poly_deg)
            continuation_value_all = polyval_numba(regression_coeffs, asset_paths[:, idx])
        else:
            continuation_value_all = np.zeros_like(asset_paths[:, idx])

        should_exercise = (exercise_value > continuation_value_all) & in_the_money_mask
        cash_flows[should_exercise] = exercise_value[should_exercise]

    option_price = np.mean(cash_flows) * np.exp(-r * exercise_dates[0])
    return option_price


# #############################################################################
# 2. Vectorized Pricers for Other Options
# #############################################################################
@numba.jit(nopython=True)
def clip_pts(qmc_points):
    return np.clip(qmc_points, MACHINE_EPSILON, 1 - MACHINE_EPSILON)

def lookback_option_pricer(s0, t, r, sigma, d, qmc_points):
    """Prices a floating strike lookback call option using QMC."""
    dt = t / d
    normal_samples = norm.ppf(qmc_points)
    brownian_path_increments = np.sqrt(dt) * normal_samples
    brownian_paths = np.cumsum(brownian_path_increments, axis=1)
    time_steps = np.linspace(dt, t, d)
    asset_paths = s0 * np.exp((r - 0.5 * sigma**2) * time_steps + sigma * brownian_paths)
    
    final_prices = asset_paths[:, -1]
    min_prices = np.min(asset_paths, axis=1)
    payoffs = final_prices - min_prices
    
    option_price = np.exp(-r * t) * np.mean(payoffs)
    return option_price

def barrier_option_pricer(s0, k, t, r, sigma, d, barrier, qmc_points):
    """Prices a down-and-out call option using QMC."""
    dt = t / d
    normal_samples = norm.ppf(qmc_points)
    brownian_path_increments = np.sqrt(dt) * normal_samples
    brownian_paths = np.cumsum(brownian_path_increments, axis=1)
    time_steps = np.linspace(dt, t, d)
    asset_paths = s0 * np.exp((r - 0.5 * sigma**2) * time_steps + sigma * brownian_paths)

    payoffs = np.maximum(asset_paths[:, -1] - k, 0)
    knock_out_mask = np.any(asset_paths <= barrier, axis=1)
    payoffs[knock_out_mask] = 0

    option_price = np.exp(-r * t) * np.mean(payoffs)
    return option_price

def basket_option_pricer(s0_list, k, t, r, sigma_list, qmc_points, corr_matrix=None):
    """Prices a call option on the arithmetic average of a basket of assets."""
    num_assets = len(s0_list)
    normal_samples = norm.ppf(qmc_points)

    if corr_matrix is not None:
        cholesky_factor = np.linalg.cholesky(corr_matrix)
        normal_samples = normal_samples @ cholesky_factor.T
    
    drift = (r - 0.5 * sigma_list**2) * t
    diffusion = sigma_list * np.sqrt(t) * normal_samples
    asset_prices_at_T = s0_list * np.exp(drift + diffusion)

    average_basket_prices = np.mean(asset_prices_at_T, axis=1)
    payoffs = np.maximum(average_basket_prices - k, 0)
    
    option_price = np.exp(-r * t) * np.mean(payoffs)
    return option_price


# #############################################################################
# 3. Parameters and Main Execution Logic
# #############################################################################

# Define parameters for all test cases
true_prices = {
    "Lookback_Base": 15.314456237747097,
    "Lookback_HighVol": 26.75528839963275,
    "Barrier_Base": 10.151287494731099,
    "Barrier_Close": 7.071902826155028,
    "Basket_32D_LowRho": 5.881124426328102,
    "Basket_32D_HighRho": 9.686894412925659,
    "Basket_32D_MixedVol": 10.347006025966984,
    "Basket_32D_OTM": 1.1746340581833037,
    "Bermudan_ATM": 5.956545670585972,
    "Bermudan_ITM": 11.250153361482433
}

all_options = {
    # Lookback Options (d=32)
    "Lookback_Base": {"s0": 100, "t": 1.0, "r": 0.05, "sigma": 0.2, "d": 32},
    "Lookback_HighVol": {"s0": 100, "t": 1.0, "r": 0.05, "sigma": 0.4, "d": 32},
    
    # Barrier Options (d=32)
    "Barrier_Base": {"s0": 100, "k": 100, "t": 1.0, "r": 0.05, "sigma": 0.2, "barrier": 85, "d": 32},
    "Barrier_Close": {"s0": 100, "k": 100, "t": 1.0, "r": 0.05, "sigma": 0.2, "barrier": 95, "d": 32},
    
    # Basket Options (d = num_assets)
    "Basket_32D_LowRho": {"s0_list": np.full(32, 100), "k": 100, "t": 1.0, "r": 0.05, "sigma_list": np.full(32, 0.2), "d": 32, "corr_matrix": np.full((32, 32), 0.1) + np.diag(np.full(32, 0.9))},
    "Basket_32D_HighRho": {"s0_list": np.full(32, 100), "k": 100, "t": 1.0, "r": 0.05, "sigma_list": np.full(32, 0.2), "d": 32, "corr_matrix": np.full((32, 32), 0.8) + np.diag(np.full(32, 0.2))},
    "Basket_32D_MixedVol": {"s0_list": np.full(32, 100), "k": 100, "t": 1.0, "r": 0.05, "sigma_list": np.linspace(0.15, 0.4, 32), "d": 32, "corr_matrix": np.full((32, 32), 0.5) + np.diag(np.full(32, 0.5))},
    "Basket_32D_OTM": {"s0_list": np.full(32, 100), "k": 110, "t": 1.0, "r": 0.05, "sigma_list": np.full(32, 0.2), "d": 32, "corr_matrix": np.full((32, 32), 0.1) + np.diag(np.full(32, 0.9))},
    
    # Bermudan Options (d=32 sim steps)
    "Bermudan_ATM": {"s0": 100, "k": 100, "t": 1.0, "r": 0.05, "sigma": 0.2, "d": 32, "exercise_dates": [0.25, 0.5, 0.75, 1.0]},
    "Bermudan_ITM": {"s0": 90, "k": 100, "t": 1.0, "r": 0.05, "sigma": 0.2, "d": 32, "exercise_dates": [0.25, 0.5, 0.75, 1.0]}
}

# Map option names to their pricing functions
pricer_map = {
    "Lookback": lookback_option_pricer,
    "Barrier": barrier_option_pricer,
    "Basket": basket_option_pricer,
    "Bermudan": bermudan_put_pricer_numba
}

# High-accuracy simulation settings
N_REPLICATIONS = 10000
# M_BASE_2 = 21  # 2**21 -> ~2 million points

# # Reduced accuracy for slow pricers (Bermudan)
# N_REPLICATIONS_SLOW = 50
# M_BASE_2_SLOW = 15 # 2**15 -> ~32k points

if __name__ == "__main__":
    print("--- Starting High-Accuracy Benchmark Price Calculation ---")
    
    for n_rep in range(5, 14):
        sobol_price = {name: np.zeros(N_REPLICATIONS) for name in all_options.keys()}
        optimal_price = {name: np.zeros(N_REPLICATIONS) for name in all_options.keys()}
        for i in range(N_REPLICATIONS):
            qmc_engine = qmc.Sobol(d=32, scramble=True, seed=i)
            qmc_points = qmc_engine.random_base2(m=n_rep)
            qmc_points = clip_pts(qmc_points)
            
            for name, params in all_options.items():
                pricer_key = name.split('_')[0]
                pricer_func = pricer_map[pricer_key]            
                pricer_params = params.copy()            
                if pricer_key == 'Basket':
                    pricer_params.pop('d', None)

                res = pricer_func(**pricer_params, qmc_points=qmc_points)
                sobol_price[name][i] = res
            
            rng = np.random.default_rng(seed=i)  # Use a different seed for each run
            shift_bits_py = rng.integers(2, size=(32, 30), dtype=np.uint32)
            digital_shifts_py = np.dot(shift_bits_py, 2 ** np.arange(30, dtype=np.uint32))
            ltm_elements_for_tril = rng.integers(2, size=(32, 30, 30), dtype=np.uint32)
            ltm_py = np.tril(ltm_elements_for_tril)
            for d_idx_ltm in range(32):
                for i_idx_ltm in range(30):
                    ltm_py[d_idx_ltm, i_idx_ltm, i_idx_ltm] = 1
            
            sobol_params = construct_sobol_sequence()
            n_pts = 2 ** n_rep
            D_dims = 32
            optimal_points_set = get_sobol_points_cpp(sobol_params, n_pts, D_dims, ltm_py, digital_shifts_py)
            optimal_points_set = clip_pts(optimal_points_set)
            for name, params in all_options.items():
                pricer_key = name.split('_')[0]
                pricer_func = pricer_map[pricer_key]
                pricer_params = params.copy()
                if pricer_key == 'Basket':
                    pricer_params.pop('d', None)

                # Use the optimal points for the current option
                optimal_price[name][i] = pricer_func(**pricer_params, qmc_points=optimal_points_set)
        print(f"\n--- Results for n_rep={n_rep} ---")

        # Calculate mean and std error for each option
        for name in all_options.keys():
            sobol_bias = (np.mean(sobol_price[name]) - true_prices[name]) ** 2
            sobol_var = np.var(sobol_price[name])
            mse = np.mean((sobol_price[name] - true_prices[name]) ** 2)
            print(f"Sobol: n_rep={n_rep}, {name}: Bias={sobol_bias}, Variance={sobol_var}, MSE={mse}")
            optimal_bias = (np.mean(optimal_price[name]) - true_prices[name]) ** 2
            optimal_var = np.var(optimal_price[name])
            optimal_mse = np.mean((optimal_price[name] - true_prices[name]) ** 2)
            print(f"Optimal: n_rep={n_rep}, {name}: Bias={optimal_bias}, Variance={optimal_var}, MSE={optimal_mse}")
            