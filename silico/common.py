import warnings

# prod
try:
    from math import prod
except ImportError:
    from functools import reduce
    from operator import mul


    def prod(i):
        """Product of a list of numbers"""
        return reduce(mul, i, 1)


def is_notebook():
    """Detect if running in a notebook"""
    # Not sure this covers
    try:
        ipython_module = get_ipython().__module__
        if ipython_module in ['ipykernel.zmqshell', 'google.colab._shell']:
            # Surely a notebook
            return True
        elif ipython_module in ['IPython.terminal.interactiveshell']:
            # Surely not a notebook
            return False
        else:
            warnings.warn("Unknown iPython detected: %s. Assuming not a notebook.")
            return False
    except NameError:
        # Surely not iPython
        return False


def set_kwargs(f, fixed_kwargs):
    """Closure of a function fixing some kwargs"""

    def f2(*args, **kwargs):
        fixed_kwargs2 = {k: v for k, v in fixed_kwargs.items() if k not in kwargs}
        return f(*args, **fixed_kwargs2, **kwargs)

    return f2
