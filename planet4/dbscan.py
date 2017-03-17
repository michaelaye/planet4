import logging
import math
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import circmean
from sklearn.cluster import DBSCAN

from . import markings, io

logger = logging.getLogger(__name__)


def get_average_objects(clusters, kind):
    """Create the average object out of a sequence of clusters.

    Parameters
    ----------
    clusters : sequence of pandas.DataFrames
        table with rows of markings (fans or blotches) to be averaged
    kind : {'fan', 'blotch}
        Switch to control the circularity for the average angle calculation.

    Returns
    -------
    Generator providing single row pandas.DataFrames with the average values
    """
    logger.debug("Averaging clusters.")
    for cluster_df in clusters:
        # first filter for outliers more than 1 std away
        # for
        # reduced = df[df.apply(lambda x: np.abs(x - x.mean()) / x.std() < 1).all(axis=1)]
        logger.debug("Averaging %i objects.", len(cluster_df))
        meandata = cluster_df.mean()
        # this determines the upper limit for circular mean
        high = 180 if kind == 'blotch' else 360
        avg = circmean(cluster_df.angle, high=high)
        meandata.angle = avg
        meandata['n_votes'] = len(cluster_df)
        yield meandata.to_frame().T


def plot_results(p4id, labels, data=None, kind=None, reduced_data=None, ax=None):
    functions = dict(blotch=p4id.plot_blotches,
                     fan=p4id.plot_fans)
    if ax is None:
        fig, ax = plt.subplots()

    plot_kwds = {'alpha': 0.8, 's': 10, 'linewidths': 0}
    palette = sns.color_palette('bright', len(labels))
    cluster_colors = [palette[x] if x >= 0 else (0.75, 0.75, 0.75)
                      for x in labels]
    p4id.show_subframe(ax=ax)
    if data is not None:
        ax.scatter(data.loc[:, 'x'], data.loc[:, 'y'], c=cluster_colors,
                   **plot_kwds)
    markings.set_subframe_size(ax)
    # pick correct function for kind of marking:
    if any(reduced_data):
        functions[kind](ax=ax, data=reduced_data, lw=1)


class DBScanner(object):
    """Potential replacement for ClusteringManager

    Parameters
    ----------
    msf : float
        m_ean s_amples f_actor: Factor to multiply number of markings with to calculate the
        min_samples value for DBSCAN to use
    savedir : str, pathlib.Path
        Path where to store clustered results
    with_angles, with_radii : bool
        Switches to control if clustering should include angles and radii respectively.
    split_by_size : bool
        Switch to control if splitting the clustering by size of markings shall occur.
    save_results : bool
        Switch to control if the resulting clustered objects should be written to disk.
    """
    # set all the different eps values for the different clustering loops here:
    eps_values = {
        'fan': {
            'xy': {  # in pixels
                'small': 10,
                'large': 50,
            },
            'angle': 20,  # degrees
            'radius': {
                'small': None,  # not in use currently for fans`
                'large': None,  # ditto
            }
        },
        'blotch': {
            'xy': {  # in pixels
                'small': 15,
                'large': 50,
            },
            'angle': 20,  # degrees
            'radius': {
                'small': 30,
                'large': 50,
            }
        }
    }

    def __init__(self, msf=0.13, savedir=None, with_angles=True, with_radii=True,
                 split_by_size=True, save_results=True):
        self.msf = msf
        self.savedir = savedir
        self.with_angles = with_angles
        self.with_radii = with_radii
        self.split_by_size = split_by_size
        self.save_results = save_results

    def show_markings(self, id_):
        p4id = markings.ImageID(id_)
        p4id.plot_all()

    def cluster_any(self, X, eps):
        logger.debug("Clustering any.")
        db = DBSCAN(eps, self.min_samples).fit(X)
        labels = db.labels_
        unique_labels = sorted(set(labels))

        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        self.n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        logger.debug("%i cluster(s) found with:", self.n_clusters)

        self.labels = labels

        # loop over unique labels.
        for k in unique_labels:
            class_member_mask = (labels == k)
            if k == -1:
                continue
            indices = class_member_mask & core_samples_mask
            logger.debug("%i members.", np.count_nonzero(indices))
            yield indices

    def cluster_xy(self, data, eps):
        logger.debug("Clustering x,y.")
        X = data[['x', 'y']].as_matrix()
        for cluster_index in self.cluster_any(X, eps):
            yield data.loc[cluster_index]

    def split_markings_by_size(self, data, limit=210):
        kind = data.marking.value_counts()
        if len(kind) > 1:
            raise TypeError("Data had more than 1 marking kind.")
        if kind.index[0] == 'blotch':
            f1 = data.radius_1 > limit
            f2 = data.radius_2 > limit
            data_large = data[f1 | f2]
            data_small = data[~(f1 | f2)]
        else:
            f1 = data.distance > limit
            data_large = data[f1]
            data_small = data[~f1]
        return data_small, data_large

    def cluster_angles(self, xy_clusters, kind):
        logger.debug("Clustering angles.")
        cols_to_cluster = dict(blotch=['y_angle'],
                               fan=['x_angle', 'y_angle'])
        eps_degrees = self.eps_values[kind]['angle']
        # convert to radians
        # calculated value of euclidean distance of unit vector
        # end points per degree
        eps_per_degree = math.tau / 360
        eps = eps_degrees * eps_per_degree
        for xy_cluster in xy_clusters:
            X = xy_cluster[cols_to_cluster[kind]]
            for indices in self.cluster_any(X, eps):
                yield xy_cluster.loc[indices]

    def cluster_radii(self, angle_clusters, eps):
        logger.debug("Clustering radii.")
        cols_to_cluster = ['radius_1', 'radius_2']
        for angle_cluster in angle_clusters:
            X = angle_cluster[cols_to_cluster]
            for indices in self.cluster_any(X, eps):
                yield angle_cluster.loc[indices]

    def cluster_and_plot(self, img_id, kind, msf=None, eps_values=None, ax=None, fontsize=None):
        """Cluster and plot the results for one P4 image_id.

        Parameters
        ----------
        img_ig : str
            Planet Four image_id
        kind : {'fan', 'blotch'}
            Kind of marking
        eps_values : dictionary, optional
            Dictionary with clustering values to be used. If not given, use stored default one.
            This is mostly used for `self.parameter_scan`.
        ax : matplotlib.axis, optional
            Matplotlib axis to be used for plotting. If not given, a new figure and axis is
            created.
        fontsize : int, optional
            Fontsize for the plots' headers.
        """
        if msf is not None:
            self.msf = msf
        if eps_values is None:
            # if not given, use stored default values:
            eps_values = self.eps_values

        self.cluster_image_id(img_id, msf, eps_values)

        reduced_data = self.reduced_data[kind]

        try:
            n_reduced = len(reduced_data)
        except TypeError:
            n_reduced = 0

        if ax is None:
            fig, ax = plt.subplots()
        if n_reduced > 0:
            plot_results(self.p4id, self.labels, kind=kind,
                         reduced_data=reduced_data, ax=ax)
        else:
            self.p4id.show_subframe(ax=ax)
        eps = eps_values[kind]['xy']['small']
        eps_large = eps_values[kind]['xy']['large']
        ax.set_title("MS: {}, n_clusters: {}\nEPS: {}, EPS_LARGE: {}, "
                     .format(self.min_samples, n_reduced, eps, eps_large),
                     fontsize=fontsize)

    @property
    def min_samples(self):
        """Calculate min_samples for DBSCAN.

        From current self.msf value and no of classifications.
        """
        min_samples = round(self.msf * self.p4id.n_marked_classifications)
        return max(3, min_samples)  # never use less than 3

    def cluster_image_name(self, image_name, msf=None, eps_values=None):
        "Cluster all image_ids for a given image_name (i.e. HiRISE obsid)"
        db = io.DBManager()
        data = db.get_image_name_markings(image_name)
        image_ids = data.image_id.unique()
        for image_id in image_ids:
            print(image_id)
            self.cluster_image_id(image_id, msf, eps_values)

    def cluster_image_id(self, img_id, msf=None, eps_values=None):
        """Interface function for users to cluster data for one P4 image_id.

        This method does the data splitting in case it is required and calls the
        `_cluster_pipeline` that goes over all dimensions to cluster.


        Parameters
        ----------
        img_ig : str
            Planet Four image_id
        msf : float, optional
            mean_samples_factor to be used for calculating min_samples. Default as given
            during __init__.
        eps_values : dictionary, optional
            Dict with eps values for clustering, in the format as given in `self.eps_values`.
            If not provided, the default stored `self.eps_values` is used.

        Returns
        -------
        At the end the data from differently-sized clustering is concatenated into the same
        results pd.DataFrame and then being stored per marking kind in the dictionary
        `self.reduced_data`.
        """
        self.p4id = markings.ImageID(img_id, scope='planet4')
        self.img_id = img_id

        if msf is not None:
            # this sets the stored msf, automatically changing min_samples accordingly
            self.msf = msf

        eps_values = self.eps_values if eps_values is None else eps_values

        # set up storage for results
        reduced_data = {}
        for kind in ['fan', 'blotch']:
            # fill in empty list in case we need to bail for not enough data
            reduced_data[kind] = []
            logger.debug('%s loop', kind)
            data = self.p4id.filter_data(kind)
            if len(data) < self.min_samples:
                # skip all else if we have not enough markings
                continue
            if self.split_by_size is True:
                datasets = self.split_markings_by_size(data)
                sizes = ['small', 'large']
            else:
                # doing it like this enables to use the same loop for both cases
                datasets = [data]
                sizes = ['small']
            # this loop either executes once or twice (more?) for the split datasets.
            for dataset, size in zip(datasets, sizes):
                eps_xy = eps_values[kind]['xy'][size]
                eps_rad = eps_values[kind]['radius'][size]
                logger.debug("Length of dataset: %i", len(dataset))
                if len(dataset) < self.min_samples:
                    logger.warning("Skipping due to lack of data.")
                    reduced_data[kind].append(pd.DataFrame())
                    continue
                reduced_data[kind].append(self._cluster_pipeline(kind, dataset, eps_xy, eps_rad))
            # merging large and small markings clusters
            try:
                reduced_data[kind] = pd.concat(reduced_data[kind], ignore_index=True)
            except ValueError as e:
                # i can just continue here, as I stored an empty list above already
                continue

        if self.save_results:
            self.store_clustered(reduced_data)
        self.reduced_data = reduced_data

    def _cluster_pipeline(self, kind, data, eps, eps_rad):
        """Cluster pipeline that can cluster over xy, angles and radii.

        It does so without knowledge of different marking sizes, it just receives data and
        will cluster it together, successively.
        """
        xyclusters = self.cluster_xy(data, eps)
        xyclusters = list(xyclusters)
        if self.with_angles:
            last = self.cluster_angles(xyclusters, kind)
        else:
            last = xyclusters
        last = list(last)
        if self.with_radii and kind == 'blotch':
            finalclusters = self.cluster_radii(last, eps_rad)
        else:
            finalclusters = last
        finalclusters = list(finalclusters)
        averaged = get_average_objects(finalclusters, kind)
        try:
            reduced_data = pd.concat(averaged, ignore_index=True)
        except ValueError as e:
            if e.args[0].startswith("No objects to concatenate"):
                logger.warning("No clusters survived.")
                return None
            else:
                raise e
        return reduced_data

    def parameter_scan(self, img_id, kind, msf_vals_to_scan, eps_vals_to_scan,
                       size_to_scan='large', do_scale=False, create_plot=True):
        """Method to scan parameter space and plot results in multi-figure plot.

        Parameters
        ----------
        kind : {'fan', 'blotch'}
            Marking kind
        msf_values : iterable (list, array, tuple), length of 2
            1D container for msf values to use
        eps_values : iterable, length of 3
            1D container for eps_values to be used. If they are used for the small or large
            items is determined by `size_to_scan`
        size_to_scan : {'small', 'large'}
            Switch to interpret which eps_values I have received. If 'small' to scan, I take
            the large value from `self.eps_values` as constant, and vice versa.
        do_scale : bool
            Switch to control if scaling is applied.
        """
        self.kind = kind
        fig, ax = plt.subplots(nrows=len(msf_vals_to_scan),
                               ncols=len(eps_vals_to_scan) + 1,
                               figsize=(10, 5))
        axes = ax.flatten()

        for ax, (msf, eps) in zip(axes, product(msf_vals_to_scan,
                                                eps_vals_to_scan)):
            eps_values = self.eps_values.copy()

            eps_values[kind]['xy'][size_to_scan] = eps

            self.cluster_and_plot(img_id, kind, msf, eps_values, ax=ax, fontsize=8)
            t = ax.get_title()
            ax.set_title("MSF: {}, {}".format(msf, t),
                         fontsize=8)

        # plot input tile
        self.p4id.show_subframe(ax=axes[-1])
        axes[-1].set_title("Input tile", fontsize=8)
        # plot marking data
        self.p4id.plot_markings(kind, ax=axes[-2], lw=0.25, with_center=True)
        axes[-2].set_title("{} marking data".format(kind), fontsize=8)
        fig.suptitle("ID: {}, n_class: {}, angles: {}, radii: {}"
                     .format(img_id, self.p4id.n_marked_classifications,
                             self.with_angles, self.with_radii))
        if create_plot:
            savepath = f"plots/{img_id}_{kind}_angles{self.with_angles}_radii{self.with_radii}.png"
            Path(savepath).parent.mkdir(exist_ok=True)
            fig.savefig(savepath, dpi=200)

    def store_clustered(self, reduced_data):
        "Store the clustered but as of yet unfnotched data."
        pm = io.PathManager(self.img_id, obsid=self.p4id.image_name)

        for outpath, outdata in zip([pm.reduced_blotchfile, pm.reduced_fanfile],
                                    [reduced_data['blotch'], reduced_data['fan']]):
            outpath.parent.mkdir(exist_ok=True, parents=True)
            if outpath.exists():
                outpath.unlink()
            if len(outdata) == 0:
                continue
            df = outdata
            # df = outdata.apply(pd.to_numeric, errors='ignore')
            try:
                df['n_votes'] = df['n_votes'].astype('int')
            # when df is just list of Nones, will create TypeError
            # for bad indexing into list.
            except TypeError:
                # nothing to write
                logger.warning("Outdata was empty, nothing to store.")
                return
            df.to_csv(str(outpath.with_suffix('.csv')), index=False)
