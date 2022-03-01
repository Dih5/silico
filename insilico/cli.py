# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import importlib

import click
from .base import Experiment


@click.group()
def cli():
    """Insilico command line utilities"""
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
@click.argument('file')
def run(file, experiment):
    """Run an experiment"""
    e = get_experiment(file, experiment)
    if e is None:
        return 1
    e.run_all()


if __name__ == "__main__":
    cli()
