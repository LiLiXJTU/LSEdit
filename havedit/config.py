from dataclasses import dataclass, field


@dataclass
class RuntimeConfig:
    """Runtime defaults for launching HAVEdit inference."""
    backend: str = "flux2"
    model_path: str = "/data/ll/weight/black-forest-labs/FLUX.2-klein-base-9B"
    enable_cpu_offload: bool = True
    gpu_id: int = 0
    torch_dtype: str = "bfloat16"
    print_havsr_debug: bool = False
    print_softmap_terms_debug: bool = False
    havedit_end: int = 0
    havsr_gate_largest_component_ratio: float = 0.45
    havsr_gate_containment_ratio: float = 0.22
    background_pixel_ring_width: int = 1
    background_pixel_ring_alpha: float = 0.5


@dataclass
class WSPConfig:
    """Warmup semantic prior configuration used before aggregation."""
    warmup_steps: int = 6
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.0


@dataclass
class HAVSRConfig:
    """Head-aware value selective replacement parameters."""
    alpha: float = 2.0
    beta: float = 1.0
    threshold: float = 0.9
    local_kernel_size: int = 5
    eps: float = 1e-6
    block_scope: str = "double_stream"
    decision_granularity: str = "head_token"


@dataclass
class BHCConfig:
    """Boundary head consistency scoring defaults."""
    enabled: bool = True
    tau_low: float = 0.35
    tau_high: float = 0.65
    lambda_max: float = 0.15
    eps: float = 1e-6


@dataclass
class TrajectoryTrustConfig:
    """Short-horizon trust tracking defaults for HAVSR."""
    enabled: bool = True
    ema_decay: float = 0.7
    release_bias: float = 0.1
    release_scale: float = 8.0
    min_steps: int = 2
    eps: float = 1e-6


@dataclass
class HAVEditConfig:
    """Root HAVEdit configuration exposing all major subsystems."""
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    wsp: WSPConfig = field(default_factory=WSPConfig)
    havsr: HAVSRConfig = field(default_factory=HAVSRConfig)
    bhc: BHCConfig = field(default_factory=BHCConfig)
    trajectory_trust: TrajectoryTrustConfig = field(default_factory=TrajectoryTrustConfig)
    hav_steps: int = 15
