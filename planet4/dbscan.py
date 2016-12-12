import logging

import numpy as np
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN


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

    def __init__(self, current_X, eps=10, min_samples=3):
        self.current_X = current_X
        self.eps = eps
        self.min_samples = min_samples

        # these lines execute the clustering
        self._run_DBSCAN()
        self._post_analysis()

    def _run_DBSCAN(self):
        """Perform the DBSCAN clustering."""
        logging.debug("Running DBSCAN")
        clusterer = DBSCAN(self.eps, self.min_samples).fit(self.current_X)
        labels = clusterer.labels_.astype('int')
        self.core_samples = clusterer.core_sample_indices_
        unique_labels = set(labels)
        self.n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        self.labels = labels
        self.unique_labels = unique_labels
        logging.debug("Estimated number of clusters: %i", self.n_clusters)

    def _post_analysis(self):
        """Use clustering results to create mean markings."""
        self.clustered_data = []  # list of `kind` cluster average objects
        self.n_rejected = 0
        # loop over unique labels.
        for label in self.unique_labels:
            # get indices for members of this cluster
            cluster_members = [i[0] for i in np.argwhere(self.labels == label)]

            # treat noise
            if label == -1:
                self.n_rejected = len(cluster_members)
            # if label is a cluster member:
            else:
                self.clustered_data.append(cluster_members)


class HDBScanner(object):

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

    def __init__(self, current_X, min_cluster_size=3,
                 min_samples=1, proba_cut=0.0):
        self.current_X = current_X
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.proba_cut = proba_cut
        # these lines execute the clustering
        self._run_HDBSCAN()
        self._post_analysis()

    def _run_HDBSCAN(self):
        """Perform the HDBSCAN clustering."""
        logging.debug("Running HDBSCAN")
        clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size,
                            min_samples=self.min_samples).fit(self.current_X)
        labels = clusterer.labels_
        core_samples_mask = np.zeros_like(clusterer.labels_, dtype=bool)
        core_samples_mask[clusterer.probabilities_ > self.proba_cut] = True
        self.core_samples_mask = core_samples_mask
        unique_labels = set(labels)
        self.n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        self.labels = labels
        self.unique_labels = unique_labels
        logging.debug("Estimated number of clusters: %i", self.n_clusters)

    def _post_analysis(self):
        """Use clustering results to create mean markings."""
        self.clustered_data = []  # list of `kind` cluster average objects
        self.n_rejected = 0

        # loop over unique labels.
        for label in self.unique_labels:
            # get indices for members of this cluster
            class_member_mask = (self.labels == label)
            cluster_members = (class_member_mask & self.core_samples_mask)
            # treat noise
            if label == -1:
                self.n_rejected = len(cluster_members)
            # if label is a cluster member:
            else:
                self.clustered_data.append(cluster_members)
