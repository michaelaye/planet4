import logging
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hdbscan import HDBSCAN
from sklearn.preprocessing import scale, robust_scale

from . import markings
from ._utils import get_average_object

logger = logging.getLogger(__name__)


def post_processing(kind, data, clusterer, min_members=3):
    Marking = markings.Fan if kind == 'fan' else markings.Blotch
    cols = Marking.to_average

    reduced_data = []
    logger.debug("Post processing %s", kind)
    for cluster in clusterer.clustered_indices:
        clusterdata = data.loc[cluster, cols + ['user_name']]
        logger.debug("N of members in this cluster: %i", len(clusterdata))
        filtered = clusterdata.groupby('user_name').first().reset_index()
        if len(clusterdata) != len(filtered):
            logger.debug("N of members of this cluster after filtering: %i,"
                         " removed %i.", len(filtered),
                         len(clusterdata) - len(filtered))
            if len(filtered) < min_members:
                logger.debug("Throwing away this cluster for < min_samples")
                continue
        logger.debug("Calculating mean %s object.", kind)
        meandata = get_average_object(filtered[cols], kind)
        # # This returned a pd.Series object I can add more info to now:
        # # storing n_votes into the object for later.
        # meandata['n_votes'] = len(filtered)
        # meandata['image_id'] = pm.id_
        # meandata['image_name'] = self.marking_data.image_name.values[0]
        # # converting to dataframe and clean data format (rows=measurements)
        reduced_data.append(meandata.to_frame().T)
    logger.debug("Length of reduced data (total number clusters found): %i",
                 len(reduced_data))
    # self.reduced_data[kind] = reduced_data
    try:
        return pd.concat(reduced_data, ignore_index=True)
    except ValueError:
        return []


def plot_results(clusterer, data, p4id, kind, reduced_data=None, ax=None):
    functions = dict(blotch=p4id.plot_blotches,
                     fan=p4id.plot_fans)
    if ax is None:
        fig, ax = plt.subplots()

    plot_kwds = {'alpha': 0.8, 's': 10, 'linewidths': 0}
    palette = sns.color_palette('bright', clusterer.n_clusters)
    cluster_colors = [palette[x] if x >= 0 else (0.5, 0.5, 0.5)
                      for x in clusterer.hdbscan.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, clusterer.hdbscan.probabilities_)]
    p4id.show_subframe(ax=ax)
    ax.scatter(data.loc[:, 'x'], data.loc[:, 'y'], c=cluster_member_colors,
               **plot_kwds)
    # pick correct function for kind of marking:
    if reduced_data is not None:
        functions[kind](ax=ax, data=reduced_data, lw=1)


def parameter_scan(img_id, kind,
                   cols='x y'.split(), only_core=True, do_scale=True,
                   proba_cut=0.9, factor=0.1):
    p4id = markings.TileID(img_id, scope='planet4')
    functions = dict(blotch=p4id.plot_blotches,
                     fan=p4id.plot_fans)
    min_samples_base = round(factor * p4id.n_marked_classifications)
    min_cluster_size_vals = [min_samples_base,
                             round(1.5 * min_samples_base)]
    min_samples_vals = [1,
                        min_samples_base,
                        round(1.5 * min_samples_base)]
    data = p4id.filter_data(kind)
    X = data[cols].as_matrix()
    if do_scale:
        X = robust_scale(X)
    fig, ax = plt.subplots(nrows=len(min_cluster_size_vals),
                           ncols=len(min_samples_vals) + 1)
    axes = ax.flatten()
    for ax, (mcs, ms) in zip(axes,
                             product(min_cluster_size_vals,
                                     min_samples_vals)):
        logger.debug("Running with %i and %i.", mcs, ms)
        if ms > mcs:
            ax.set_title('ms > mcs')
            ax.set_axis_off()
            continue
        # elif ms == mcs and ms == 2 * min_samples_base:
        #     p4id.show_subframe(ax=ax)
        #     continue
        clusterer = HDBScanner(X, mcs, ms, proba_cut=proba_cut,
                               only_core=only_core,
                               metric='manhattan')
        reduced_data = post_processing(kind, data, clusterer)
        plot_results(clusterer, data, p4id, kind, reduced_data, ax=ax)
        ax.set_title('MCS: {}, MS: {}\nn_clusters: {}, averaged: {}'
                     .format(mcs, ms, clusterer.n_clusters,
                             len(reduced_data)),
                     fontsize=6)

        threshold = pd.Series(clusterer.hdbscan.outlier_scores_).quantile(0.9)
        outliers = np.where(clusterer.hdbscan.outlier_scores_ > threshold,
                            True, False)
        ax.scatter(data.loc[outliers, 'x'],
                   data.loc[outliers, 'y'],
                   marker='x', s=15, linewidth=1, c='red', alpha=0.75)
    p4id.show_subframe(ax=axes[-1])
    functions[kind](ax=axes[-2], lw=0.25)
    fig.suptitle("n_class: {}, ncols: {}, factor: {}, scale: {}"
                 .format(p4id.n_marked_classifications, len(cols),
                         factor, do_scale))
    savepath = ("plots/{}/{}_lencols{}_factor{}_scale{}.png"
                .format(kind, img_id, len(cols), factor, do_scale))
    fig.savefig(savepath, dpi=200)


class HDBScanner(object):

    """Execute clustering and create mean cluster markings.

    The instantiated object will execute:

        * _run_DBSCAN() to perform the clustering itself
        * _post_analysis() to create mean markings from the clustering results


    Parameters
    ----------
    current_X : numpy.array
        array holding the data to be clustered, preprocessed in ClusterManager
    min_cluster_size : int
        Set this to the smallest size grouping that you wish to consider
        a cluster.
    min_samples : int, optional
        Despite its name, this parameter controls how `conservative` the
        clustering will be. By default at the value of `min_cluster_size`,
        this will be allowing progressively more clusters to appear when
        decreased below the value of min_cluster_size.
    """

    def __init__(self, current_X, min_cluster_size=3,
                 min_samples=None, proba_cut=0.9, only_core=True,
                 metric='euclidean'):
        self.current_X = current_X
        self.min_cluster_size = min_cluster_size
        if min_samples is None:
            self.min_samples = min_cluster_size
        else:
            self.min_samples = min_samples
        self.proba_cut = proba_cut
        self.only_core = only_core
        self.metric = metric
        logger.debug("min_cluster_size: %i", self.min_cluster_size)
        logger.debug("min_samples: %i", self.min_samples)
        # these lines execute the clustering
        self._run_HDBSCAN()

    def _run_HDBSCAN(self):
        """Perform the HDBSCAN clustering."""
        logger.debug("Running HDBSCAN")
        clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size,
                            min_samples=self.min_samples,
                            metric=self.metric).fit(self.current_X)
        self.hdbscan = clusterer

        unique_labels = sorted(set(clusterer.labels_))
        self.n_clusters = len(unique_labels) - (1 if -1 in clusterer.labels_ else 0)
        logger.debug("Estimated number of clusters: %i", self.n_clusters)

        # outliers
        threshold = pd.Series(clusterer.outlier_scores_).quantile(0.9)
        not_outliers = np.where(clusterer.outlier_scores_ < threshold, True, False)

        # probability cut
        high_prob = np.where(self.hdbscan.probabilities_ > self.proba_cut, True, False)

        self.clustered_indices = []  # list of `kind` cluster average objects
        self.n_rejected = 0

        # loop over unique labels.
        for label in unique_labels:

            # get boolean mask for members of this cluster
            label_member_mask = (clusterer.labels_ == label)

            # treat noise, no storage
            if label == -1:
                self.n_rejected = len(label_member_mask)
                continue

            # if I only want the best:
            if self.only_core:
                cluster_members = label_member_mask & high_prob & not_outliers
                logger.debug('n_true: %i', np.count_nonzero(cluster_members))
            else:
                cluster_members = label_member_mask
            self.clustered_indices.append(cluster_members)
