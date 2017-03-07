import logging
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import circmean
from sklearn.cluster import DBSCAN

from . import io, markings

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
        functions[kind](ax=ax, data=reduced_data, lw=1, user_color=color)


class DBScanner(object):
    """Potential replacement for ClusteringManager

    Parameters
    ----------
    img_id : str
        planet4 image_id string. Can be the right-hand minimal identifier,
        lik 'pbr', will be padded to the full one.
    """
    # shortcut translator
    t = dict(b='blotch',
             f='fan',
             blotch='blotch',
             fan='fan',
             blotches='blotch',
             fans='fan')

    radii_eps = 30

    def __init__(self, img_id, output_dir_clustered=None):
        self.img_id = img_id
        self.p4id = markings.ImageID(img_id, scope='planet4')
        self.output_dir_clustered = output_dir_clustered
        self.pm = io.PathManager(img_id)

    def show_markings(self):
        self.p4id.plot_all()

    def cluster_any(self, X, eps, min_samples):
        logger.debug("Clustering any.")
        db = DBSCAN(eps, min_samples).fit(X)
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

    def cluster_xy(self, eps, min_samples):
        logger.debug("Clustering x,y.")
        X = self.data[['x', 'y']].as_matrix()
        for cluster_index in self.cluster_any(X, eps, min_samples):
            yield self.data.loc[cluster_index]

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

    def cluster_angles(self, xy_clusters,
                       min_samples,
                       eps_fanangle=20,
                       eps_blotchangle=20):
        logger.debug("Clustering angles.")
        cols_to_cluster = dict(blotch=['y_angle'],
                               fan=['x_angle', 'y_angle'])
        kind = self.kind
        eps_degrees = eps_fanangle if kind == 'fan' else eps_blotchangle
        # convert to radians
        # calculated value of euclidean distance of unit vector
        # end points per degree
        eps_per_degree = 2 * np.pi / 360
        eps = eps_degrees * eps_per_degree
        for xy_cluster in xy_clusters:
            X = xy_cluster[cols_to_cluster[kind]]
            for indices in self.cluster_any(X, eps, min_samples):
                yield xy_cluster.loc[indices]

    def cluster_radii(self, angle_clusters, min_samples):
        logger.debug("Clustering radii.")
        cols_to_cluster = ['radius_1', 'radius_2']
        for angle_cluster in angle_clusters:
            X = angle_cluster[cols_to_cluster]
            for indices in self.cluster_any(X, self.radii_eps, min_samples):
                yield angle_cluster.loc[indices]

    def cluster_and_plot(self, kind, eps, min_samples, with_angles=True,
                         with_radii=True, ax=None, fontsize=None,
                         eps_large=None):
        self.kind = self.t[kind]
        data = self.p4id.filter_data(self.kind)
        if eps_large is not None:
            datasets = self.split_markings_by_size(data)
            epsilons = [eps, eps_large]
            radii_eps = [20, 50]
        else:
            datasets = [data]
            epsilons = [eps]
            radii_eps = [30]
        reduced_data = []
        for dataset, epsnow, radeps in zip(datasets, epsilons, radii_eps):
            logger.info("Clustering with eps=%i", epsnow)
            self.data = dataset
            logger.debug("Length of dataset: %i", len(self.data))
            if len(self.data) < min_samples:
                logger.info("Skipping due to lack of data.")
                reduced_data.append(pd.DataFrame())
                continue
            self.radii_eps = radeps
            reduced_data.append(self.pipeline(epsnow, min_samples,
                                              with_angles, with_radii)
                                )
        # merging large and small markings clusters
        try:
            reduced_data = pd.concat(reduced_data, ignore_index=True)
        except ValueError as e:
            if e.args[0].startswith("All objects passed were None"):
                logger.warning("Nothing survived.")
                return
            else:
                raise e

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
        ax.set_title("MS: {}, n_clusters: {}\nEPS: {}, EPS_LARGE: {}, "
                     .format(min_samples, n_reduced, eps, eps_large),
                     fontsize=fontsize)
        self.reduced_data = reduced_data
        self.n_reduced = n_reduced

    @property
    def store_folder(self):
        return self.pm.datapath / self.p4id.image_name / self.img_id

    def store_clustered(self, reduced_data):
        "Store the clustered but as of yet unfnotched data."
        outdir = self.store_folder
        outdir.mkdir(exist_ok=True)
        for outfname, outdata in zip([self.pm.reduced_blotchfile, self.pm.reduced_fanfile],
                                     [self.reduced_data['blotch'],
                                      self.reduced_data['fan']]):
            if outfname.exists():
                outfname.unlink()
            if len(outdata) == 0:
                continue
            df = pd.concat(outdata, ignore_index=True)
            # make
            df = df.apply(pd.to_numeric, errors='ignore')
            df['n_votes'] = df['n_votes'].astype('int')
            self.save(df, outfname)

    def pipeline(self, eps, min_samples, with_angles=True,
                 with_radii=True):
        kind = self.kind
        xyclusters = self.cluster_xy(eps, min_samples)
        xyclusters = list(xyclusters)
        if with_angles:
            last = self.cluster_angles(xyclusters, min_samples)
        else:
            last = xyclusters
        last = list(last)
        if with_radii and kind == 'blotch':
            finalclusters = self.cluster_radii(last, min_samples)
        else:
            finalclusters = last
        finalclusters = list(finalclusters)
        averaged = get_average_objects(finalclusters, kind)
        try:
            reduced_data = pd.concat(averaged, ignore_index=True)
        except ValueError as e:
            if e.args[0].startswith("No objects to concatenate"):
                logger.warning("No clusters survived.")
                return pd.DataFrame()
            else:
                raise e
        return reduced_data

    def parameter_scan(self, kind, msf_values, eps_values, do_scale=False,
                       with_angles=True, with_radii=True):
        kind = self.kind = self.t[kind]
        fig, ax = plt.subplots(nrows=len(msf_values),
                               ncols=len(eps_values) + 1,
                               figsize=(10, 5))
        axes = ax.flatten()
        for ax, (msf, eps) in zip(axes, product(msf_values, eps_values)):
            min_samples = round(msf * self.p4id.n_marked_classifications)
            # don't allow less than 3 min_samples:
            min_samples = max(3, min_samples)
            self.cluster_and_plot(kind, 10, min_samples,
                                  with_angles=with_angles,
                                  with_radii=with_radii, eps_large=eps,
                                  ax=ax, fontsize=8)
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
                     .format(self.img_id, self.p4id.n_marked_classifications,
                             with_angles, with_radii))
        savepath = ("plots/{id_}_{kind}_scale{s}_radii{r}.png"
                    .format(kind=kind, id_=self.img_id,
                            s=do_scale, r=with_radii))
        fig.savefig(savepath, dpi=200)
