from dataclasses import dataclass, field
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
    adc_drift_rate: Optional[float] = None
    adc_drift_mode: Optional[str] = None
    adc_drift_params: Optional[Dict[str, Any]] = None
    efficiency: Dict[str, Any] = field(default_factory=dict)
    random_seed: Optional[int] = None
    git_commit: Optional[str] = None
    requirements_sha256: Optional[str] = None
    cli_sha256: Optional[str] = None
    cli_args: List[str] = field(default_factory=list)
    analysis: Dict[str, Any] = field(default_factory=dict)
