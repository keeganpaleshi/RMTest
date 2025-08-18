# Experimental analysis modes

Opt-in features allow testing a log-linear background model (`background_model: loglin_unit`) and an extended unbinned likelihood (`likelihood: extended`). These modes remain experimental and are disabled by default.

The log-linear background fits a straight line in log space normalised to unit area, while the extended likelihood incorporates Poisson fluctuations in the total event count.

Enable both via the CLI or in `config.yaml`:

```bash
python analyze.py --background-model loglin_unit --likelihood extended --input merged_data.csv
```

```yaml
analysis:
  background_model: loglin_unit
  likelihood: extended
```
