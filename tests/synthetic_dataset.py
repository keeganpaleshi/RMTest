import numpy as np
import pandas as pd


PEAKS = {
    "Po210": 5.3,
    "Po218": 6.0,
    "Po214": 7.7,
}


def synthetic_spectrum(scale=1, *, rng_seed=0):
    """Return energies for three peaks plus background."""
    rng = np.random.default_rng(rng_seed)
    energies = []
    energies.append(rng.normal(PEAKS["Po210"], 0.05, 50 * scale))
    energies.append(rng.normal(PEAKS["Po218"], 0.05, 60 * scale))
    energies.append(rng.normal(PEAKS["Po214"], 0.05, 70 * scale))
    energies.append(rng.uniform(5.0, 8.0, 100 * scale))
    return np.concatenate(energies)


def synthetic_dataset(scale=1, *, burst=False, rng_seed=0):
    """Return DataFrame with simple spectrum and timestamps."""
    rng = np.random.default_rng(rng_seed)
    energies = synthetic_spectrum(scale, rng_seed=rng_seed)
    times = rng.uniform(0, 100, energies.size)
    if burst:
        burst_e = rng.normal(PEAKS["Po214"], 0.05, 30 * scale)
        burst_t = rng.uniform(40, 42, burst_e.size)
        energies = np.concatenate([energies, burst_e])
        times = np.concatenate([times, burst_t])
    adc = np.round(energies * 1000).astype(int)
    df = pd.DataFrame(
        {
            "fUniqueID": np.arange(adc.size),
            "fBits": np.zeros(adc.size, dtype=int),
            "timestamp": [pd.Timestamp(t, unit="s", tz="UTC") for t in times],
            "adc": adc,
            "fchannel": np.ones(adc.size, dtype=int),
        }
    )
    return df
