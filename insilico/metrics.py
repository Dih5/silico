from collections import OrderedDict

from sklearn import metrics

import numpy as np

from .common import set_kwargs

# Available metrics for binary classifiers, not depending on a target class
_classifier_metrics = OrderedDict([
    ("Accuracy", metrics.accuracy_score),
    ("Cohen kappa", metrics.cohen_kappa_score),
    ("Matthews Phi", metrics.matthews_corrcoef),
])

# TODO: Not used yet
# ROC uses probability score instead of values
_classifier_score_metrics = OrderedDict([
    ("ROC AUC", metrics.roc_auc_score),
])

_classifier_metrics_averaged = OrderedDict([
    ("Precision", lambda criterium: set_kwargs(metrics.precision_score, {'average': criterium})),
    ("Recall", lambda criterium: set_kwargs(metrics.recall_score, {'average': criterium})),
    ("F1", lambda criterium: set_kwargs(metrics.f1_score, {'average': criterium}))
])

_average_criteria = ['micro', 'macro', 'weighted']
"""Criteria to average a target-dependent metric. Note None can be used instead to retrieve the list of
target-dependent values"""


def get_classification_metrics(y, predictions, classes=None):
    if classes is None:
        classes = list(set(y))
    par_list = []
    # Fixed pars
    for metric, f in _classifier_metrics.items():
        try:
            par_list.append((metric, f(y, predictions)))
        except (ValueError, TypeError) as e:
            par_list.append((metric, np.nan))

    # Parameters depending on target/averaged
    for metric, f in _classifier_metrics_averaged.items():
        try:
            par_list.append((metric,
                             {**{criterium: f(criterium)(y, predictions) for
                                 criterium in _average_criteria}, **{
                                 "target": f(None)(y, predictions,
                                                   labels=classes).tolist()}}))
        except (ValueError, TypeError) as e:
            par_list.append((metric, np.nan))

    return OrderedDict(par_list)
