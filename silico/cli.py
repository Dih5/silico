# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import importlib

import click
from .base import Experiment


@click.group()
def cli():
    """Silico command line utilities"""
    pass


def get_experiment(file, experiment=None, report=True):
    """Load an experiment from a script where it is defined"""
    if file.lower().endswith(".py"):
        file = file[:-3]
    m = importlib.import_module(file)
    candidates = {k: e for k, e in m.__dict__.items() if isinstance(e, Experiment)}
    if experiment is not None:
        try:
            e = candidates[experiment]
        except KeyError:
            if report:
                print("Error: chosen experiment is not available in file. Available experiments are: %s." % ", ".join(
                    candidates.keys()))
            return None
    else:
        if len(candidates) == 1:
            e = list(candidates.values())[0]
        elif len(candidates) == 0:
            if report:
                print("Error: no experiment found in file.")
            return None
        else:
            if report:
                print(
                    "Error: multiple experiments available. Specify as --experiment plus one of the following: %s." % ", ".join(
                        candidates.keys()))
            return None
    return e


@cli.command()
@click.option('--experiment', help="Name of the experiment inside of the module.")
@click.argument('file')
def status(file, experiment):
    """Check the status of an experiment"""
    e = get_experiment(file, experiment)
    if e is None:
        return 1
    d = e.status()
    print("%d/%d (%g %%)" % (d["done"], d["total"], d["done"] / d["total"] * 100))


@cli.command()
@click.option('--experiment', help="Name of the experiment inside of the module.")
@click.option('--output', "-o", help="Output file. The extension determines the format.")
@click.argument('file')
def export_results(file, output, experiment):
    """Export the results of an experiment"""
    if output is not None and "." not in output:
        print("Error: the output must include an extension")
        return 1

    extensions = {"pkl", "tex", "csv", "json"}  # Must match the code below

    if output is not None:
        extension = output.rsplit(".", 1)[-1].lower()
        if extension not in extensions:
            print("Invalid extension. Available options are: %s" % ", ".join(extensions))
            return 1

    e = get_experiment(file, experiment)
    if e is None:
        return 1
    df = e.get_results_df()
    if output is None:
        print(df)
        print("Preview shown above. Use -o <path> to export, including an extension")
        return 0

    extension = output.rsplit(".", 1)[-1].lower()

    if extension == "pkl":
        df.to_pickle(output)
    elif extension == "tex":
        df.to_latex(output)
    elif extension == "csv":
        df.to_csv(output)
    elif extension == "json":
        df.to_json(output, indent=1)
    else:
        print("Error: unreachable code reached (extension-related problem)")
        return 1


@cli.command()
@click.option('--experiment', help="Name of the experiment inside of the module.")
@click.argument('file')
def run(file, experiment):
    """Run an experiment"""
    e = get_experiment(file, experiment)
    if e is None:
        return 1
    e.run_all()


if __name__ == "__main__":
    cli()
