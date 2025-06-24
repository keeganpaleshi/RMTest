from dataclasses import dataclass
from typing import Any, Mapping

@dataclass
class Summary:
    timestamp: str
    config_used: str
    calibration: Any
    calibration_valid: Any
    spectral_fit: Mapping[str, Any]
    time_fit: Mapping[str, Mapping[str, Any]]
    systematics: Mapping[str, Any] | None
    baseline: Mapping[str, Any] | None
    radon_results: Mapping[str, Any] | None
    noise_cut: Mapping[str, Any]
    burst_filter: Mapping[str, Any]
    adc_drift_rate: Any
    adc_drift_mode: Any
    adc_drift_params: Any
    efficiency: Mapping[str, Any] | None
    random_seed: Any
    git_commit: str | None
    requirements_sha256: str | None
    cli_sha256: str | None
    cli_args: list[str]
    analysis: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

