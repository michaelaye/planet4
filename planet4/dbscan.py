import logging
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import robust_scale

from . import markings
from ._utils import get_average_object

logger = logging.getLogger(__name__)


def post_processing(kind, data, dbscanner, min_members=3):
    Marking = markings.Fan if kind == 'fan' else markings.Blotch
    cols = Marking.to_average

    reduced_data = []
    logger.debug("Post processing %s", kind)
    for cluster in dbscanner.clustered_indices:
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
    palette = sns.color_palette('bright', clusterer.n_clusters_)
    cluster_colors = [palette[x] if x >= 0 else (0.5, 0.5, 0.5)
                      for x in clusterer.dbscan.labels_]
    p4id.show_subframe(ax=ax)
    ax.scatter(data.loc[:, 'x'], data.loc[:, 'y'], c=cluster_colors,
               **plot_kwds)
    # pick correct function for kind of marking:
    if reduced_data is not None:
        functions[kind](ax=ax, data=reduced_data, lw=1)


def parameter_scan(img_id, kind, msf_values, eps_values,
                   cols=None, do_scale=True, do_radii=False):
    if cols is None:
        cols = 'x y'.split()
    p4id = markings.ImageID(img_id, scope='planet4')
    functions = dict(blotch=p4id.plot_blotches,
                     fan=p4id.plot_fans)
    data = p4id.filter_data(kind)
    if do_radii:
        cols += ['radius_1', 'radius_2']
    X = data[cols].as_matrix()
    if do_scale:
        X = robust_scale(X)
    fig, ax = plt.subplots(nrows=len(msf_values),
                           ncols=len(eps_values) + 1,
                           figsize=(10, 5))
    axes = ax.flatten()
    for ax, (msf, eps) in zip(axes,
                              product(msf_values,
                                      eps_values)):
        min_samples = round(msf * p4id.n_marked_classifications)
        dbscanner = DBScanner(X, eps=eps, min_samples=min_samples)
        reduced_data = post_processing(kind, data, dbscanner)
        plot_results(dbscanner, data, p4id, kind, reduced_data, ax=ax)
        ax.set_title('MS: {}, MSF: {}, EPS: {}\nn_clusters: {}, averaged: {}'
                     .format(min_samples, msf, eps, dbscanner.n_clusters_,
                             len(reduced_data)),
                     fontsize=8)

    p4id.show_subframe(ax=axes[-1])
    functions[kind](ax=axes[-2], lw=0.25, with_center=True)
    fig.suptitle("n_class: {}, ncols: {}, radii: {}, scale: {}"
                 .format(p4id.n_marked_classifications, len(cols),
                         do_radii, do_scale))
    savepath = ("plots/{}/{}_lencols{}_radii{}_scale{}.png"
                .format(kind, img_id, len(cols), do_radii, do_scale))
    # fig.tight_layout()
    fig.savefig(savepath, dpi=200)


class DBScanner(object):

    """Execute clustering and create mean cluster markings.

    The instantiated object will execute:

        * _run_DBSCAN() to perform the clustering itself
        * _post_analysis() to create mean markings from the clustering results


    Parameters
    ----------
    current_X : numpy.array
        array holding the data to be clustered, preprocessed in ClusterManager
    eps : int, optional
        Distance criterion for DBSCAN algorithm. Samples further away than this value don't
        become members of the currently considered cluster. Default: 10
    min_samples : int, optional
        Mininum number of samples required for a cluster to be created. Default: 3
    """

    def __init__(self, current_X, eps=10, min_samples=3, only_core=True):
        self.current_X = current_X
        self.eps = eps
        self.min_samples = min_samples
        self.only_core = only_core

        logger.debug("DBScanner received: eps=%i, min_samples=%i", eps, min_samples)
        logger.debug("Shape of X: %i, %i", *current_X.shape)
        # these lines execute the clustering
        self._run_DBSCAN()

    def _run_DBSCAN(self):
        """Perform the DBSCAN clustering."""
        logger.debug("Running DBSCAN")
        db = DBSCAN(self.eps, self.min_samples).fit(self.current_X)
        self.dbscan = db
        labels = db.labels_
        unique_labels = sorted(set(labels))

        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        self.n_clusters_ = len(unique_labels) - (1 if -1 in labels else 0)
        logger.debug("%i clusters found.", self.n_clusters_)
        self.clustered_indices = []  # list of `kind` cluster average objects
        self.n_rejected = 0
        # loop over unique labels.
        for k in unique_labels:
            # get indices for members of this cluster
            class_member_mask = (labels == k)
            # treat noise
            if k == -1:
                self.n_rejected = np.sum(class_member_mask)
                continue
            if self.only_core:
                cluster_members = (class_member_mask & core_samples_mask)
            else:
                cluster_members = class_member_mask
            self.clustered_indices.append(cluster_members)

        self.core_samples_mask = core_samples_mask
        logger.debug("Length of clustered_indices: %i", len(self.clustered_indices))
