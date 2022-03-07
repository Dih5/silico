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
