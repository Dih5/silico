from numpy import random

from silico import Experiment


def experiment_f(mean, sigma, seed):
    # All seeds should be initialized using a parameter for reproducibility
    random.seed(seed)
    # Return a dict with the results (must be pickleable)
    return {"value": random.normal(mean, sigma)}


def test_simple():
    """Test a simple experiment"""
    experiment = Experiment(
        [
            ("mean", [1, 2, 4]),
            ("sigma", [1, 2, 3]),
            ("seed", list(range(20))),
        ],
        experiment_f,  # Function
        "test-data",  # Folder where the results are stored
    )
    experiment.invalidate()
    experiment.run_all()
    df = experiment.get_results_df()
    assert set(df.columns) == {"_run_start", "_elapsed_seconds", "value"}
    assert len(df) == 180
    experiment.invalidate()


def test_error():
    """Test an experiment which raises errors"""

    def erroring_f(mean, sigma, seed):
        if mean == 2:
            raise ValueError("An example error raised when mean==2")
        random.seed(seed)
        return {"value": random.normal(mean, sigma)}

    experiment = Experiment(
        [
            ("mean", [1, 2, 4]),
            ("sigma", [1, 2, 3]),
            ("seed", list(range(5))),
        ],
        erroring_f,
        "test-data",
        "erroring"
    )
    experiment.invalidate()
    assert experiment.status() == {"total": 45, "done": 0, "errors": 0}
    experiment.run_all()
    assert experiment.status() == {"total": 45, "done": 45, "errors": 15}
    # Retrieve skipping errors
    df = experiment.get_results_df()
    assert len(df) == 30
    assert "_error" not in df.columns
    # Retrieve not skipping errors
    df = experiment.get_results_df(skip_errors=False)
    assert len(df) == 45
    assert "_error" in df.columns
    experiment.invalidate()
