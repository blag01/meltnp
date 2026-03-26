from np_shift import PROJECT_NAME, build_default_experiment, run


def test_project_name_is_exposed() -> None:
    assert PROJECT_NAME == "neural-processes-distribution-shift"


def test_default_experiment_runs() -> None:
    config = build_default_experiment()
    result = run(config)
    assert "experiment=baseline" in result
