# Experimental analysis modes

These opt-in features let interested users test alternate background and likelihood formulations while keeping the default linear background and unextended likelihood unchanged.

- **loglin_unit background** – uses a unit-area log-linear shape scaled by a positive `S_bkg` parameter.
- **extended likelihood** – includes the Poisson term for the expected event count.

Enable them from the CLI or configuration; see the README for examples.
The spectral-fitting section now includes an EMG configuration snippet
showing `fitting.use_emg`, the per-isotope `tau` priors, and the
`fitting.emg_stable_mode` toggle that selects the stable EMG backend.
