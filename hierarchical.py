import numpy as np
import pymc as pm

__all__ = ["fit_hierarchical_runs"]


def fit_hierarchical_runs(run_results, draws=2000, tune=1000, chains=2, random_seed=42):
    """Hierarchical Bayesian inference across multiple runs.

    Parameters
    ----------
    run_results : list of dict
        Each entry must contain ``"half_life"`` and ``"dhalf_life"``. Optional
        keys ``"slope_MeV_per_ch"``, ``"dslope"``, ``"intercept"`` and ``"dintercept"`` add
        calibration parameters to the model.
    draws : int, optional
        Number of MCMC draws after tuning (default 2000).
    tune : int, optional
        Number of tuning steps (default 1000).
    chains : int, optional
        Number of chains to run (default 2).
    random_seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    dict
        Posterior means, standard deviations and 95% HDIs for the global
        parameters ``half_life``, ``slope_MeV_per_ch`` and ``intercept``.
    """

    sanitized = []
    for entry in run_results:
        hl = entry.get("half_life")
        dhl = entry.get("dhalf_life")
        if hl is None or dhl is None:
            continue
        hl = float(hl)
        dhl = float(dhl)
        if not np.isfinite(hl) or not np.isfinite(dhl) or dhl < 0:
            continue

        clean = {"half_life": hl, "dhalf_life": dhl}

        slope = entry.get("slope_MeV_per_ch")
        dslope = entry.get("dslope")
        if (
            slope is not None
            and dslope is not None
            and np.isfinite(slope)
            and np.isfinite(dslope)
        ):
            clean["slope_MeV_per_ch"] = float(slope)
            clean["dslope"] = float(dslope)

        intercept = entry.get("intercept")
        dintercept = entry.get("dintercept")
        if (
            intercept is not None
            and dintercept is not None
            and np.isfinite(intercept)
            and np.isfinite(dintercept)
        ):
            clean["intercept"] = float(intercept)
            clean["dintercept"] = float(dintercept)

        sanitized.append(clean)

    n = len(sanitized)
    if n == 0:
        raise ValueError("run_results must contain finite half-life measurements")

    hl_obs = np.array([r["half_life"] for r in sanitized], dtype=float)
    hl_err = np.array([max(r["dhalf_life"], 1e-6) for r in sanitized], dtype=float)

    use_slope = all("slope_MeV_per_ch" in r and "dslope" in r for r in sanitized)
    if use_slope:
        slope_obs = np.array([r["slope_MeV_per_ch"] for r in sanitized], dtype=float)
        slope_err = np.array([max(r["dslope"], 1e-6) for r in sanitized], dtype=float)

    use_intercept = all("intercept" in r and "dintercept" in r for r in sanitized)
    if use_intercept:
        int_obs = np.array([r["intercept"] for r in sanitized], dtype=float)
        int_err = np.array([max(r["dintercept"], 1e-6) for r in sanitized], dtype=float)

    with pm.Model():
        # Global mean and between-run scatter for half-life
        mu_hl = pm.Normal("mu_hl", mu=hl_obs.mean(), sigma=10 * hl_obs.std() + 1e-6)
        hl_scale = max(hl_obs.std(), hl_obs.mean() * 0.1, 1e-3)
        sigma_hl = pm.HalfNormal("sigma_hl", sigma=hl_scale)
        hl = pm.Normal("hl", mu=mu_hl, sigma=sigma_hl, shape=n)
        pm.Normal("hl_obs", mu=hl, sigma=hl_err, observed=hl_obs)

        if use_slope:
            mu_a = pm.Normal("mu_a", mu=slope_obs.mean(), sigma=10 * slope_obs.std() + 1e-6)
            slope_scale = max(slope_obs.std(), slope_obs.mean() * 0.1, 1e-3)
            sigma_a = pm.HalfNormal("sigma_a", sigma=slope_scale)
            a = pm.Normal("a", mu=mu_a, sigma=sigma_a, shape=n)
            pm.Normal("a_obs", mu=a, sigma=slope_err, observed=slope_obs)

        if use_intercept:
            mu_c = pm.Normal("mu_c", mu=int_obs.mean(), sigma=10 * int_obs.std() + 1e-6)
            int_scale = max(int_obs.std(), int_obs.mean() * 0.1, 1e-3)
            sigma_c = pm.HalfNormal("sigma_c", sigma=int_scale)
            c = pm.Normal("c", mu=mu_c, sigma=sigma_c, shape=n)
            pm.Normal("c_obs", mu=c, sigma=int_err, observed=int_obs)

        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                           random_seed=random_seed, target_accept=0.9, progressbar=False)

    summ = pm.summary(trace, hdi_prob=0.95)
    out = {
        "half_life": {
            "mean": float(summ.loc["mu_hl", "mean"]),
            "sd": float(summ.loc["mu_hl", "sd"]),
            "hdi": [float(summ.loc["mu_hl", "hdi_2.5%"]), float(summ.loc["mu_hl", "hdi_97.5%"])],
        }
    }

    if use_slope:
        out["slope_MeV_per_ch"] = {
            "mean": float(summ.loc["mu_a", "mean"]),
            "sd": float(summ.loc["mu_a", "sd"]),
            "hdi": [float(summ.loc["mu_a", "hdi_2.5%"]), float(summ.loc["mu_a", "hdi_97.5%"])],
        }
    if use_intercept:
        out["intercept"] = {
            "mean": float(summ.loc["mu_c", "mean"]),
            "sd": float(summ.loc["mu_c", "sd"]),
            "hdi": [float(summ.loc["mu_c", "hdi_2.5%"]), float(summ.loc["mu_c", "hdi_97.5%"])],
        }
    return out
