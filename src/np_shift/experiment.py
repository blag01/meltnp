from dataclasses import dataclass

PROJECT_NAME = "neural-processes-distribution-shift"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    train_shift: str
    test_shift: str


def build_default_experiment() -> ExperimentConfig:
    return ExperimentConfig(
        name="baseline",
        train_shift="known_gaussian_corruption",
        test_shift="unknown_context_shift",
    )


def run(config: ExperimentConfig) -> str:
    return (
        f"experiment={config.name} "
        f"train_shift={config.train_shift} "
        f"test_shift={config.test_shift}"
    )
