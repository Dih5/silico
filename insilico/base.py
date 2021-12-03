from itertools import product
import base64
import hashlib
import json
import os
import pickle
from glob import glob

from functools import reduce

import pandas as pd

from .common import is_notebook

try:
    if not is_notebook():
        from tqdm import tqdm
    else:
        from tqdm.notebook import tqdm
except ImportError:
    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs["iterable"]


def prod(i):
    """Product of a list of numbers"""
    return reduce(lambda x, y: x * y, i, 1)


def _hash_function(w):
    """Hash function to provide a unique (negligible collision) string identifier from a dict of parameters"""
    h = hashlib.md5(w)
    return base64.b64encode(h.digest())[:12].decode("utf-8").replace("/", "_")


class Trial:
    """A Trial able to provide a result from a dict of parameters"""

    def __init__(self, kwargs, f, base_path="", base_name=None):
        """

        Args:
            kwargs (dict): Mapping of arguments to use in the call.
            f (callable): Function called when performing the trial.
            base_path (str): Path to the storage dir.
            base_name (str): Prefix for the file name. If None, a name will be extracted from f.

        """
        self.kwargs = kwargs
        self.f = f
        self.base_path = base_path

        self.base_name = base_name if base_name is not None else f.__name__

        self.results = {}

    def get_hash(self):
        """Get a hash identifying the trial"""
        str_form = json.dumps(self.kwargs, sort_keys=True)
        return _hash_function(str_form.encode('utf-8'))

    def get_file_name(self, extension=".pkl"):
        """Get a unique filename for the trial"""
        return "%s-%s%s" % (self.base_name, self.get_hash(), extension)

    def run(self):
        """Execute the trial"""
        return self.f(**self.kwargs)

    def run_and_save(self):
        """Execute the trial and store the results as a pickle and in the db"""
        result = self.run()
        pickle.dump(result, open(os.path.join(self.base_path, self.get_file_name()), "wb"))
        return result

    def load(self):
        """Load the results of the trial if available"""
        return pickle.load(open(os.path.join(self.base_path, self.get_file_name()), "rb"))

    def load_or_run(self):
        """Load the results if available, otherwise running the trial, storing the results, and returning them"""
        try:
            return self.load()
        except FileNotFoundError:
            pass
        return self.run_and_save()

    def delete(self):
        """Remove the stored results of the trial"""
        os.remove(self.get_file_name())


def ensure_dir_exists(path):
    """
    Ensure a directory exists, creating it if needed.
    Args:
        path (str): The path to the directory.
    """
    if path:  # Empty dir (cwd) always exists
        try:
            # Will fail either if exists or unable to create it
            os.makedirs(path)
        except OSError:
            # Also raised if the path exists
            pass

        if not os.path.exists(path):
            # There was an error on creation, so make sure we know about it
            raise OSError("Unable to create directory " + path)


class Variable:
    """A variable taking part in an experiment"""

    def __init__(self, name):
        self.name = name

    def iter_values(self):
        raise NotImplementedError


class GridVariable(Variable):
    """A variable whose values are defined on some grid points"""

    def __init__(self, name, grid):
        super(GridVariable, self).__init__(name)
        self.grid = grid

    def iter_values(self):
        for v in self.grid:
            yield v

    def __len__(self):
        return len(self.grid)


def implicit_variable_cast(variable):
    if isinstance(variable, Variable):
        return variable
    elif isinstance(variable, (tuple, list)) and len(variable) == 2:
        return GridVariable(name=variable[0], grid=variable[1])


class Experiment:
    """An experiment"""

    def __init__(self, variables, f, store, base_name=None):
        """

        Args:
            variables (list of Variable or (str, list) tuples): The list of variables.
            f (callable): A function which maps variables names into results, which should be a dict.
            store (str): A path to store the results.
            base_name (str): Prefix for the file name. If None, a name will be extracted from f.

        """
        self.variables = [implicit_variable_cast(v) for v in variables]
        self.f = f
        self.store = store
        self.base_name = base_name

        ensure_dir_exists(store)

        self._len = prod(len(v) for v in self.variables)

    def __len__(self):
        return self._len

    def iter_values(self):
        """Iterate all combinations of kwargs"""
        names = [v.name for v in self.variables]
        for t in product(*(v.iter_values() for v in self.variables)):
            yield dict(zip(names, t))

    def run_all(self):
        """Run all trials. If already run, kept."""
        for kwargs in tqdm(self.iter_values(), total=len(self)):
            Trial(kwargs, self.f, self.store, base_name=self.base_name).load_or_run()

    def iter_results(self):
        """Iterate pairs of kwargs, results

        If a result is not available, it is skipped
        """
        for kwargs in self.iter_values():
            try:
                yield kwargs, Trial(kwargs, self.f, self.store, base_name=self.base_name).load()
            except FileNotFoundError:  # Not available
                pass

    def get_results_df(self):
        """Get a dataframe with the available results"""
        results = []
        for kwargs, result in self.iter_results():
            if isinstance(result, dict):
                # TODO: Ensure no overlapping
                results.append({**kwargs, **result})
            else:
                if "result" in kwargs:
                    raise ValueError("Conflicting name result in kwarg")
                results.append({**kwargs, "result": result})

        return pd.DataFrame(results).set_index([v.name for v in self.variables])

    def invalidate(self, only_grid=False):
        """
        Remove all existing trial data

        Args:
            only_grid (bool): True to remove only files which correspond to grid values. Otherwise, all .pkl files are
                              removed.

        """
        if only_grid:
            for kwargs in self.iter_values():
                try:
                    Trial(kwargs, self.f, self.store, base_name=self.base_name).delete()
                except FileNotFoundError:
                    pass
        else:
            for file in glob(os.path.join(self.store, "*.pkl")):
                try:
                    os.remove(file)
                except FileNotFoundError:
                    pass


def _set_kwarg(f, fixed_kwargs):
    """Closure of a function fixing a kwarg"""

    def f2(*args, **kwargs):
        fixed_kwargs2 = {k: v for k, v in fixed_kwargs.items() if k not in kwargs}
        return f(*args, **fixed_kwargs2, **kwargs)

    return f2


class SubExperiment(Experiment):
    """An restriction of an experiment, where some of its variables are fixed"""

    def __init__(self, original, fixed):
        """

        Args:
            original (Experiment): The original experiment.
            fixed (callable): A mapping from variable names to their fixed values.

        """

        variables = [a for a in original.variables if a.name not in fixed]
        f = _set_kwarg(original.f, fixed)
        store = original.store
        super().__init__(variables, f, store)
