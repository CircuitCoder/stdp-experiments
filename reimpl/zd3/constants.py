from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConstants:
    schema: str = "zero-delay-three-trace-v1"
    n_input: int = 784
    n_exc: int = 400
    n_inh: int = 400
    dt_ms: float = 0.5
    stimulus_ms: float = 350.0
    rest_ms: float = 150.0
    initial_intensity: float = 2.0
    intensity_increment: float = 1.0
    minimum_exc_spikes: int = 5
    normalization_target: float = 78.0
    train_inhibition: float = 25.5
    inference_inhibition: float = 17.0
    exc_to_inh_weight: float = 10.4
    tau_ge_ms: float = 1.0
    tau_gi_ms: float = 2.0
    exc_tau_m_ms: float = 100.0
    inh_tau_m_ms: float = 10.0
    exc_v_rest_mv: float = -65.0
    inh_v_rest_mv: float = -60.0
    exc_v_reset_mv: float = -65.0
    inh_v_reset_mv: float = -45.0
    exc_v_threshold_mv: float = -52.0
    inh_v_threshold_mv: float = -40.0
    exc_e_exc_mv: float = 0.0
    exc_e_inh_mv: float = -100.0
    inh_e_exc_mv: float = 0.0
    inh_e_inh_mv: float = -85.0
    exc_refractory_ms: float = 5.0
    inh_refractory_ms: float = 2.0
    theta_offset_mv: float = 20.0
    theta_initial_mv: float = 20.0
    theta_plus_mv: float = 0.05
    theta_tau_ms: float = 1.0e7
    pre_tau_ms: float = 20.0
    post1_tau_ms: float = 20.0
    post2_tau_ms: float = 40.0
    depression_rate: float = 0.0001
    potentiation_rate: float = 0.01
    weight_min: float = 0.0
    weight_max: float = 1.0

    @property
    def stimulus_ticks(self) -> int:
        return round(self.stimulus_ms / self.dt_ms)

    @property
    def rest_ticks(self) -> int:
        return round(self.rest_ms / self.dt_ms)

    @property
    def attempt_ticks(self) -> int:
        return self.stimulus_ticks + self.rest_ticks

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


MODEL = ModelConstants()

