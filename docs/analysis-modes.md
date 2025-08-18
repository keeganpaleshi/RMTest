# Opt-in analysis modes

These experimental options model a log-linear background and use an extended Poisson likelihood. They are opt-in; the default analysis remains linear with the current likelihood.

- `background_model: loglin_unit` — fits a unit-normalized log-linear background term.
- `likelihood: extended` — treats the fit as an extended Poisson likelihood.
