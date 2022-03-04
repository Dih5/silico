from numpy import random
from time import sleep

from silico import Experiment


def experiment_f(mean, sigma, seed):
    # All seeds should be initialized using a parameter for reproducibility
    random.seed(seed)
    # Delay for test purpose
    sleep(mean/100)
    # Return a dict with the results (must be pickleable)
    return {"value": random.normal(mean, sigma)}


experiment = Experiment(
    [
        ("mean", [1, 2, 4]),
        ("sigma", [1, 2, 3]),
        ("seed", list(range(20))),
    ],
    experiment_f,  # Function
    "experiment-demo",  # Folder where the results are stored
)
