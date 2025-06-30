import numpy as np
from scipy.stats import qmc
from scipy.stats import norm
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
    dt = t / d
    normal_samples = norm.ppf(qmc_points)
    brownian_path_increments = np.sqrt(dt) * normal_samples
    brownian_paths = np.cumsum(brownian_path_increments, axis=1)
    time_steps = np.linspace(dt, t, d)
    asset_paths = s0 * np.exp((r - 0.5 * sigma**2) * time_steps + sigma * brownian_paths)
    average_asset_prices = np.mean(asset_paths, axis=1)
    payoffs = np.maximum(average_asset_prices - k, 0)
    option_price = np.exp(-r * t) * np.mean(payoffs)
    return option_price

asian_options = {
    "og": {"S0": 50.0, "K_strike": 45.0, "T_exp": 1.0, "R_rate": 0.05, "SIGMA": 0.3, "D_dims": 32, "C0_true": 7.06451424679549},
    "otm": {"S0": 50.0, "K_strike": 60.0, "T_exp": 1.0, "R_rate": 0.05, "SIGMA": 0.3, "D_dims": 32, "C0_true": 1.0161335048477829},
    "atm": {"S0": 50.0, "K_strike": 52.5, "T_exp": 1.0, "R_rate": 0.05, "SIGMA": 0.3, "D_dims": 32, "C0_true": 2.9753224833133496},
    "itm": {"S0": 50.0, "K_strike": 40.0, "T_exp": 1.0, "R_rate": 0.05, "SIGMA": 0.3, "D_dims": 32, "C0_true": 11.015933988360171},
    "hvol": {"S0": 50.0, "K_strike": 52.5, "T_exp": 1.0, "R_rate": 0.05, "SIGMA": 0.6, "D_dims": 32, "C0_true": 6.4274784214688045},
    "lvol": {"S0": 50.0, "K_strike": 52.5, "T_exp": 1.0, "R_rate": 0.05, "SIGMA": 0.1, "D_dims": 32, "C0_true": 0.6931527993078156}
}

sobol_params = construct_sobol_sequence()


for option_name, option_params in asian_options.items():
    S0 = option_params["S0"]
    K_strike = option_params["K_strike"]
    T_exp = option_params["T_exp"]
    R_rate = option_params["R_rate"]
    SIGMA = option_params["SIGMA"]
    D_dims = option_params["D_dims"]
    C0_true = option_params["C0_true"]
    N = 10000
    for n_pts_base in range(5, 14):  # 10 to 14 for 1024 to 16384 points
        n_pts = 2 ** n_pts_base
        print(f"Number of points: {n_pts}")
        est_price = np.zeros(N)
        est_mse = np.zeros(N)
        for i in range(N):
            rng = np.random.default_rng(seed=i)  # Use a different seed for each run
            qmc_engine = qmc.Sobol(d=D_dims, scramble=True, seed=i)
            sobol_points_set = qmc_engine.random_base2(m=n_pts_base)  # Generate 2^10 = 1024 points
            estimated_price_sobol = asian_option_pricer(S0, K_strike, T_exp, R_rate, SIGMA, D_dims, sobol_points_set)
            est_price[i] = estimated_price_sobol
            est_mse[i] = (estimated_price_sobol - C0_true) ** 2

        print(f" {option_name} Sobol Bias: {(np.mean(est_price) - C0_true) ** 2} | Variance: {np.var(est_price)} | MSE: {np.mean(est_mse)}")


        est_price = np.zeros(N)
        est_mse = np.zeros(N)
        for i in range(N):
            rng = np.random.default_rng(seed=i)  # Use a different seed for each run
            shift_bits_py = rng.integers(2, size=(32, 30), dtype=np.uint32)
            digital_shifts_py = np.dot(shift_bits_py, 2 ** np.arange(30, dtype=np.uint32))
            ltm_elements_for_tril = rng.integers(2, size=(32, 30, 30), dtype=np.uint32)
            ltm_py = np.tril(ltm_elements_for_tril)
            for d_idx_ltm in range(32):
                for i_idx_ltm in range(30):
                    ltm_py[d_idx_ltm, i_idx_ltm, i_idx_ltm] = 1

            optimal_points_set = get_sobol_points_cpp(sobol_params, n_pts, D_dims, ltm_py, digital_shifts_py)
            estimated_price_optimal = asian_option_pricer(S0, K_strike, T_exp, R_rate, SIGMA, D_dims, optimal_points_set)
            est_price[i] = estimated_price_optimal
            est_mse[i] = (estimated_price_optimal - C0_true) ** 2


        print(f"{option_name} Optimal Bias: {(np.mean(est_price) - C0_true) **2} | Variance: {np.var(est_price)} | MSE: {np.mean(est_mse)}")
