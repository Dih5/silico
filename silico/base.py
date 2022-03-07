import warnings
from itertools import product
import base64
import hashlib
import json
import os
import pickle
from glob import glob
from datetime import datetime
from multiprocessing import Pool

from functools import reduce

try:
    import pandas as pd
except ImportError:
    pd = None

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

    def run_and_save(self, add_stats=True):
        """Execute the trial and store the results as a pickle and in the db"""
        start = datetime.now()
        result = self.run()
        if add_stats:
            elapsed = datetime.now() - start
            result = {"_run_start": str(start), "_elapsed_seconds": elapsed.total_seconds(), **result}
        pickle.dump(result, open(os.path.join(self.base_path, self.get_file_name()), "wb"))
        return result

    def load(self):
        """Load the results of the trial if available"""
        return pickle.load(open(os.path.join(self.base_path, self.get_file_name()), "rb"))

    def load_or_run(self, add_stats=True):
        """Load the results if available, otherwise running the trial, storing the results, and returning them"""
        try:
            return self.load()
        except FileNotFoundError:
            pass
        return self.run_and_save(add_stats=add_stats)

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

    def __init__(self, name, standard=None):
        self.name = name
        self.standard = standard

    def iter_values(self):
        raise NotImplementedError


class GridVariable(Variable):
    """A variable whose values are defined on some grid points"""

    def __init__(self, name, grid, standard=None):
        super(GridVariable, self).__init__(name, standard=standard)
        self.grid = grid

    def iter_values(self):
        for v in self.grid:
            yield v

    def get_standard(self):
        if self.standard is not None:
            return self.standard
        l = list(self.grid)
        length = len(l)
        return sorted(l)[length // 2]

    def __len__(self):
        return len(self.grid)


def implicit_variable_cast(variable):
    if isinstance(variable, Variable):
        return variable
    elif isinstance(variable, (tuple, list)) and len(variable) == 2:
        return GridVariable(name=variable[0], grid=variable[1])


class Experiment:
    """An experiment"""

    def __init__(self, variables, f, store, base_name=None, add_stats=True, strategy="grid", mid_point=None):
        """

        Args:
            variables (list of Variable or (str, list) tuples): The list of variables.
            f (callable): A function which maps variables names into results, which should be a dict.
            store (str): A path to store the results.
            base_name (str): Prefix for the file name. If None, a name will be extracted from f.
            add_stats (bool): Whether to add running information to the results. If f's output is not a dict, this
                              will fail.
            strategy (str): A method defining how the parameter space is explore. Available options are:
                            - "grid": Explore a grid in order (cartesian product)
                            - "star": Consider only variations of each of the parameters. The "standard" point can
                                      be defined with the mid_point parameter.
            mid_point (dict of str): A mapping of parameters to their "default" values. Used if strategy is "star". The
                                     mid_point must be in the grid.

        """
        self.variables = [implicit_variable_cast(v) for v in variables]
        self.f = f
        self.store = store

        self.base_name = base_name
        self.add_stats = add_stats

        self.strategy = strategy

        if strategy == "grid":
            self._len = prod(len(v) for v in self.variables)
        elif strategy == "star":
            if mid_point is not None:
                self.mid_point = mid_point
            else:
                warnings.warn("Not specifying a mid_point uses the median if available. ")
                self.mid_point = {v.name: v.get_standard() for v in self.variables}

            # Sum of lengths, but do not repeat the mid_point
            self._len = sum(len(v) for v in self.variables) - len(self.variables) + 1

        ensure_dir_exists(store)

    def __len__(self):
        return self._len

    def iter_values(self):
        """Iterate all combinations of kwargs"""
        names = [v.name for v in self.variables]
        if self.strategy == "grid":
            for t in product(*(v.iter_values() for v in self.variables)):
                yield dict(zip(names, t))
        elif self.strategy == "star":
            mid_point = self.mid_point if self.mid_point is not None else {}
            yield mid_point
            for v in self.variables:
                for value in v.iter_values():
                    if value != mid_point[v.name]:  # Do not repeat mid point
                        yield {**mid_point, **{v.name: value}}

        else:
            raise ValueError("Invalid value for parameter strategy.")

    def _run_kwargs(self, **kwargs):
        """Helper pickleable function"""
        try:
            Trial(kwargs, self.f, self.store, base_name=self.base_name).load_or_run(add_stats=self.add_stats)
        except Exception:
            print("Skipping failed run with parameters %s\n" % ", ".join(
                "%s = %s" % (str(a), str(b)) for a, b in kwargs.items()))

    def run_all(self, method="sequential", threads=2):
        """Run all trials. If already run, kept."""
        method = method.lower()
        if method == "sequential":
            for kwargs in tqdm(self.iter_values(), total=len(self)):
                try:
                    Trial(kwargs, self.f, self.store, base_name=self.base_name).load_or_run(add_stats=self.add_stats)
                except Exception:
                    print("Skipping failed run with parameters %s\n" % ", ".join(
                        "%s = %s" % (str(a), str(b)) for a, b in kwargs.items()))
        elif method == "multithreading":
            with Pool(threads) as pool:
                results = [pool.apply_async(self._run_kwargs, kwds=kwargs) for kwargs in self.iter_values()]
                for result in tqdm(results, total=len(self)):
                    result.get()
        else:
            raise ValueError("Invalid method")

    def iter_results(self):
        """Iterate pairs of kwargs, results

        If a result is not available, it is skipped
        """
        for kwargs in self.iter_values():
            try:
                yield kwargs, Trial(kwargs, self.f, self.store, base_name=self.base_name).load()
            except FileNotFoundError:  # Not available
                pass

    def get_result(self, kwargs):
        """Get the result of a certain configuration, running it if not available"""
        return Trial(kwargs, self.f, self.store, base_name=self.base_name).load_or_run(add_stats=self.add_stats)

    def status(self):
        """
        Report the status of the experiment

        Returns:
            dict of str: A mapping of statistics of the process, including:
                             - total: The total number of instances.
                             - done: The trials already completed.
        """
        count = 0
        total = 0
        for kwargs in self.iter_values():
            total += 1
            try:
                Trial(kwargs, self.f, self.store, base_name=self.base_name).load()
                count += 1
            except FileNotFoundError:
                pass

        return {"total": total, "done": count}

    def get_results_df(self):
        """Get a dataframe with the available results"""
        if pd is None:
            raise ModuleNotFoundError("The pandas package is required")

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
