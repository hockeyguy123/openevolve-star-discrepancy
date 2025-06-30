import scipy.stats.qmc as qmc
import numpy as np
from scipy.stats import norm


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

asian_og = {"S0": 50.0, "K_strike": 45.0, "T_exp": 1.0, "R_rate": 0.05, "SIGMA": 0.3, "D_dims": 32}
asian_otm = {"S0": 50.0, "K_strike": 60.0, "T_exp": 1.0, "R_rate": 0.05, "SIGMA": 0.3, "D_dims": 32}
asian_atm = {"S0": 50.0, "K_strike": 52.5, "T_exp": 1.0, "R_rate": 0.05, "SIGMA": 0.3, "D_dims": 32}
asian_itm = {"S0": 50.0, "K_strike": 40.0, "T_exp": 1.0, "R_rate": 0.05, "SIGMA": 0.3, "D_dims": 32}
asian_hvol = {"S0": 50.0, "K_strike": 52.5, "T_exp": 1.0, "R_rate": 0.05, "SIGMA": 0.6, "D_dims": 32}
asian_lvol = {"S0": 50.0, "K_strike": 52.5, "T_exp": 1.0, "R_rate": 0.05, "SIGMA": 0.1, "D_dims": 32}

est_price_og = np.zeros(1000)
est_price_otm = np.zeros(1000)
est_price_atm = np.zeros(1000)
est_price_itm = np.zeros(1000)
est_price_hvol = np.zeros(1000)
est_price_lvol = np.zeros(1000)

for i in range(1000):
    qmc_engine = qmc.Sobol(d=asian_og["D_dims"], scramble=True, seed=i)
    sobol_points_set = qmc_engine.random_base2(m=21)
    estimated_price_sobol = asian_option_pricer(
        asian_og["S0"],
        asian_og["K_strike"],
        asian_og["T_exp"],
        asian_og["R_rate"],
        asian_og["SIGMA"],
        asian_og["D_dims"],
        sobol_points_set
    )
    est_price_og[i] = estimated_price_sobol

    estimated_price_sobol_otm = asian_option_pricer(
        asian_otm["S0"],
        asian_otm["K_strike"],
        asian_otm["T_exp"],
        asian_otm["R_rate"],
        asian_otm["SIGMA"],
        asian_otm["D_dims"],
        sobol_points_set
    )
    est_price_otm[i] = estimated_price_sobol_otm

    estimated_price_sobol_atm = asian_option_pricer(
        asian_atm["S0"],
        asian_atm["K_strike"],
        asian_atm["T_exp"],
        asian_atm["R_rate"],
        asian_atm["SIGMA"],
        asian_atm["D_dims"],
        sobol_points_set
    )
    est_price_atm[i] = estimated_price_sobol_atm

    estimated_price_sobol_itm = asian_option_pricer(
        asian_itm["S0"],
        asian_itm["K_strike"],
        asian_itm["T_exp"],
        asian_itm["R_rate"],
        asian_itm["SIGMA"],
        asian_itm["D_dims"],
        sobol_points_set
    )
    est_price_itm[i] = estimated_price_sobol_itm

    estimated_price_sobol_hvol = asian_option_pricer(
        asian_hvol["S0"],
        asian_hvol["K_strike"],
        asian_hvol["T_exp"],
        asian_hvol["R_rate"],
        asian_hvol["SIGMA"],
        asian_hvol["D_dims"],
        sobol_points_set
    )
    est_price_hvol[i] = estimated_price_sobol_hvol

    estimated_price_sobol_lvol = asian_option_pricer(
        asian_lvol["S0"],
        asian_lvol["K_strike"],
        asian_lvol["T_exp"],
        asian_lvol["R_rate"],
        asian_lvol["SIGMA"],
        asian_lvol["D_dims"],
        sobol_points_set
    )
    est_price_lvol[i] = estimated_price_sobol_lvol
    
print(f"Estimated Asian Call Option Price OG: {np.mean(est_price_og)} ± {np.std(est_price_og)}")
print(f"Estimated Asian Call Option Price OTM: {np.mean(est_price_otm)} ± {np.std(est_price_otm)}")
print(f"Estimated Asian Call Option Price ATM: {np.mean(est_price_atm)} ± {np.std(est_price_atm)}")
print(f"Estimated Asian Call Option Price ITM: {np.mean(est_price_itm)} ± {np.std(est_price_itm)}")
print(f"Estimated Asian Call Option Price HVOL: {np.mean(est_price_hvol)} ± {np.std(est_price_hvol)}")
print(f"Estimated Asian Call Option Price LVOL: {np.mean(est_price_lvol)} ± {np.std(est_price_lvol)}")