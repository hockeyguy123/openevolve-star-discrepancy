import numpy as np
from scipy.stats import norm, qmc
import time

import numba

MACHINE_EPSILON = np.finfo(float).eps


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
N_REPLICATIONS = 1000
M_BASE_2 = 21  # 2**21 -> ~2 million points

# # Reduced accuracy for slow pricers (Bermudan)
# N_REPLICATIONS_SLOW = 50
# M_BASE_2_SLOW = 15 # 2**15 -> ~32k points

if __name__ == "__main__":
    print("--- Starting High-Accuracy Benchmark Price Calculation ---")
    
    results = {}
    for name in all_options.keys():
        params = all_options[name]
        d = params.get('d', 32)
        results[name] = np.zeros(N_REPLICATIONS)
    
    for i in range(N_REPLICATIONS):
        qmc_engine = qmc.Sobol(d=d, scramble=True, seed=i)
        qmc_points = qmc_engine.random_base2(m=M_BASE_2)
        qmc_points = clip_pts(qmc_points)
        
        for name, params in all_options.items():
            pricer_key = name.split('_')[0]
            pricer_func = pricer_map[pricer_key]
            
            # Make a copy of params to avoid modifying the original dict
            pricer_params = params.copy()
            
            # Basket pricer doesn't take 'd' as a direct argument, so remove it
            if pricer_key == 'Basket':
                pricer_params.pop('d', None)

            # print(f"\nCalculating price for: {name} (d={params.get('d', 32)})")
            estimated_price = pricer_func(**pricer_params, qmc_points=qmc_points)
            results[name][i] = estimated_price
    
    # Calculate mean and std error for each option
    for name in all_options.keys():
        mean_price = np.mean(results[name])
        std_err = np.std(results[name])
        print(f"{name}: {mean_price} ± {std_err}")