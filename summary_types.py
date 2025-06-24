from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class Summary:
    timestamp: str
    config_used: str
    calibration: Dict[str, Any]
    calibration_valid: bool
    spectral_fit: Dict[str, Any]
    time_fit: Dict[str, Any]
    systematics: Dict[str, Any]
    baseline: Dict[str, Any]
    radon_results: Dict[str, Any]
    noise_cut: Dict[str, Any]
    burst_filter: Dict[str, Any]
    adc_drift_rate: Optional[float]
    adc_drift_mode: Optional[str]
    adc_drift_params: Optional[Dict[str, Any]]
    efficiency: Dict[str, Any]
    random_seed: int
    git_commit: str
    requirements_sha256: Optional[str]
    cli_sha256: Optional[str]
    cli_args: List[str]
    analysis: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of this summary."""
        from dataclasses import asdict

        return asdict(self)
