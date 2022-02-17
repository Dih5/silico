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


def plot_confusion_matrix(m, labels=None, figure_kwargs=None):
    """

    Args:
        m (list of list of float): The confusion matrix.
        labels (list of str): Names of the classes in the order of the confusion matrix.
        figure_kwargs (dict): Parameters to create the Figure.

        Returns:
            Axes: An axes instance for further tweaking.
    """
    import matplotlib
    import seaborn as sns

    if figure_kwargs is None:
        figure_kwargs = {}
    fig = matplotlib.pyplot.figure(**figure_kwargs)

    gs0 = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[20, 2], hspace=0.05)
    gs00 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[1], hspace=0)

    ax = fig.add_subplot(gs0[0])
    cax1 = fig.add_subplot(gs00[0])
    cax2 = fig.add_subplot(gs00[1])

    vmin = np.min(m)
    vmax = np.max(m)
    off_diag_mask = np.eye(*m.shape, dtype=bool)

    sns.heatmap(m, annot=True, mask=~off_diag_mask, cmap='Blues', vmin=vmin, vmax=vmax, ax=ax, cbar_ax=cax2,
                )
    sns.heatmap(m, annot=True, mask=off_diag_mask, cmap='OrRd', vmin=vmin, vmax=vmax, ax=ax, cbar_ax=cax1,
                xticklabels=labels, yticklabels=labels, cbar_kws=dict(ticks=[]))

    return ax
