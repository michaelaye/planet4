import logging

import numpy as np
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


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

    def __init__(self, current_X, eps=10, min_samples=3, only_core=False):
        self.current_X = current_X
        self.eps = eps
        self.min_samples = min_samples
        self.only_core = only_core

        # these lines execute the clustering
        self._run_DBSCAN()

    def _run_DBSCAN(self):
        """Perform the DBSCAN clustering."""
        logger.debug("Running DBSCAN")
        db = DBSCAN(self.eps, self.min_samples).fit(self.current_X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        labels = db.labels_
        unique_labels = set(labels)

        self.n_clusters_ = len(unique_labels) - (1 if -1 in labels else 0)

        self.clustered_indices = []  # list of `kind` cluster average objects
        self.n_rejected = 0
        # loop over unique labels.
        for k in unique_labels:
            # get indices for members of this cluster
            class_member_mask = (labels == k)
            if self.only_core:
                cluster_members = (class_member_mask & core_samples_mask)
            else:
                cluster_members = class_member_mask

            # treat noise
            if k == -1:
                self.n_rejected = np.sum(class_member_mask)
            # if label is a cluster member:
            else:
                self.clustered_indices.append(cluster_members)

        self.db = db
        self.core_samples_mask = core_samples_mask
