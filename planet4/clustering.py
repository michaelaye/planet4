from __future__ import division, print_function

import logging

import importlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import FloatText
from pathlib import Path
from sklearn.cluster import DBSCAN

from . import io
from . import markings, plotting
from .exceptions import NoDataToClusterError, UnknownClusteringScopeError

importlib.reload(logging)
logpath = Path.home() / 'p4reduction.log'
logging.basicConfig(filename=str(logpath), filemode='w', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

matplotlib.style.use('bmh')


class DBScanner(object):
    """Execute clustering and create mean cluster markings.

    The instantiated object will execute:

        * _get_current_X() to split off the clustering input from `data`
        * _run_DBSCAN() to perform the clustering itself
        * _post_analysis() to create mean markings from the clustering results


    Parameters
    ----------
    data : pandas.DataFrame
        dataframe holding the data to be clustered.
    kind : {'fan', 'blotch'}
        String switch indicating the kind of markings to be clustered.
    eps : int, optional
        Distance criterion for DBSCAN algorithm. Samples further away than this value don't
        become members of the currently considered cluster. Default: 10
    min_samples : int, optional
        Mininum number of samples required for a cluster to be created. Default: 3
    ax : matplotlib.axes, optional
        If provided, plots will be produced on that axes.
    scope : {'hirise', 'planet4'}
        String switch to decide on which coordinates to cluster. 'planet4' means that the
        original planet4 image_id coordinates will be used. This means that `data` should only
        contain data for one image_id. Respectively, if given as 'hirise', the clustering will
        be performed on the image_x,image_y HiRISE image coordinates. This would work as well
        for data only for one planet4 image_id, but makes more sense for all data for one
        HiRISE image_name.
    linestyle : str
        matplotlib linestyle character symbol.

    Attributes
    ----------
    marking_cols : dict
        Dictionary with the column names that describe each kind of marking, minus their
        clustering coordinates
    MarkingClass : dict
        Dictionary with the class handles for the marking objects, chosen by the `kind`
        switch.
    n_reduced_data
    """
    marking_cols = {'fan': 'angle spread distance'.split(),
                    'blotch': 'angle radius_1 radius_2'.split()}
    MarkingClass = {'fan': markings.Fan,
                    'blotch': markings.Blotch}

    def __init__(self, data, kind, eps=10, min_samples=3, ax=None,
                 scope='hirise', linestyle='-'):
        self.data = data
        self.kind = kind  # fans or blotches
        self.eps = eps
        self.min_samples = min_samples
        if scope == 'planet4':
            self.coords = ['x', 'y']
        elif scope == 'hirise':
            self.coords = ['image_x', 'image_y']
        else:
            raise UnknownClusteringScopeError
        self.scope = scope
        self.ax = ax
        self.linestyle = linestyle

        # these lines execute the clustering
        self._get_current_X()
        self._run_DBSCAN()
        self._post_analysis()

    def _get_current_X(self):
        """Determine the clustering input matrix."""
        current_X = self.data[self.coords].values
        if len(current_X) == 0:
            raise NoDataToClusterError
        self.current_X = current_X

    def _run_DBSCAN(self):
        """Perform the DBSCAN clustering."""
        logging.debug("Running DBSCAN")
        db = DBSCAN(self.eps, self.min_samples).fit(self.current_X)
        labels = db.labels_.astype('int')
        self.core_samples = db.core_sample_indices_
        unique_labels = set(labels)
        self.n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        self.labels = labels
        self.unique_labels = unique_labels
        logging.debug("Estimated number of clusters:", self.n_clusters)

    def _post_analysis(self):
        """Use clustering results to create mean markings."""
        colors = plt.cm.Spectral(np.linspace(0, 1, len(self.unique_labels)))
        reduced_data = [] # list of `kind` cluster average objects
        n_rejected = 0
        # loop over unique labels.
        for k, color in zip(self.unique_labels, colors):
            label_members = [i[0] for i in np.argwhere(self.labels == k)]
            if k == -1:  # i.e. if it's noise, rejected from any cluster
                n_rejected = len(label_members)
            if self.ax:
                self.process_plotting(k, label_members)
            if k > -0.5:  # i.e. if it's a member of a cluster
                # `cluster` will be a Fan or Blotch object.
                cluster = self.get_mean_marking(label_members)
                # storing n_members into the object for later.
                cluster.n_members = len(label_members)
                # storing this saved marker for later in ClusteringManager
                cluster.saved = False
                reduced_data.append(cluster)
                if self.ax:
                    self.process_cluster_plotting(cluster, color)
            if self.ax:
                markings.set_subframe_size(self.ax)
        self.reduced_data = reduced_data
        self.n_rejected = n_rejected
        logging.debug("Reduced data to {} {}(e)s."
                      .format(self.n_reduced_data, self.kind))

    @property
    def n_reduced_data(self):
        "int : Length of list of average cluster objects."
        return len(self.reduced_data)

    def get_mean_marking(self, label_members):
        """Create a mean object out of cluster label members.

        The `marking_cols` are used to determine which data columns need to be averaged for the
        `kind`marking object.

        Note that I take the image_id of the marking of the first member of the cluster as
        image_id for the whole cluster. In very rare circumstances, this could be wrong for
        clusters in the overlap region.

        Parameters
        ----------
        label_members : list
            list of indices that belong to the current cluster label

        Returns
        -------
        markings.Fan or markings.Blotch
            As determined by `MarkingClass`.
        """
        cols = self.coords + self.marking_cols[self.kind]
        clusterdata = self.data[cols].iloc[label_members]
        meandata = clusterdata.mean()
        if self.scope == 'hirise':
            meandata.rename(index={'image_x': 'x', 'image_y': 'y'}, inplace=True)
        obj = self.MarkingClass[self.kind](meandata)
        # store the image_id from first cluster member for whole cluster
        obj.image_id = self.data[['image_id']].iloc[label_members].iloc[0]
        return obj

    def process_cluster_plotting(self, cluster, color):
        cluster.set_color(color)
        if self.kind == 'blotch':
            self.ax.add_artist(cluster)
        else:
            self.ax.add_line(cluster)
            cluster.add_semicircle(self.ax, color=color)
            cluster.add_mean_wind_pointer(self.ax, color=color,
                                          ls=self.linestyle)

    def process_plotting(self, k, label_members):
        if k == -1:  # process noise markers
            color = 'w'
            markersize = 5
        for i in label_members:
            x = self.current_X[i]
            if i in self.core_samples and k != -1:
                markersize = 8
            else:
                markersize = 5
            self.ax.plot(x[0], x[1], 'o', markerfacecolor=color,
                         markedgecolor='k', markersize=markersize)


class ClusteringManager(object):

    """Control class to manage the clustering pipeline.

    Parameters
    ----------
    dbname : str or pathlib.Path, optional
        Path to the database used by DBManager. If not provided, DBManager will find
        the most recent one and use that.
    scope : {'hirise', 'planet4'}
        Switch to control in what coordinates the clustering is happening. 'hirise' is
        required to automatically take care of tile overlaps, while 'planet4' is required
        to be able to check the quality of single Planet4 tiles.
    min_distance : int
        Parameter to control the distance below which fans and blotches are determined
        to be 2 markings of the same thing, creating a FNOTCH chimera. The ratio between
        blotch and fan marks determines a fnotch value that can be used as a discriminator
        for the final object catalog.
    eps : int
        Parameter to control the exclusion distance for the DBSCAN
    fnotched_dir : str or pathlib.Path
        Path to folder where to store output. Default: io.data_root / 'output'
    output_format : {'hdf', 'csv', 'both'}
        Format to save the output in. Default: 'hdf'
    cut : float
        Value to apply for fnotch cutting.

    Attributes
    ----------
    confusion : list
        List of confusion data.
    fnotched_fans : list
        List of Fan objects after fnotches have been removed.
    fnotched_blotches : list
        List of Blotch objects after fnotches have been removed.
    fnotches : list
        List of Fnotch objects, as determined by `do_the_fnotch`.
    clustered_fans : list
        List of clustered Fan objects after clustering and averaging.
    clustered_blotches : list
        List of clustered Blotch objects after clustering and averaging.
    n_clustered_fans
    n_clustered_blotches
    output_dir_clustered : pathlib.Path
        Path to full clustering results, without removal of fnotched clusters.
    cut_dir : pathlib.Path
        Path to final fan and blotch clusters, after applying `cut`.
    """

    def __init__(self, dbname=None, scope='hirise', min_distance=10, eps=10,
                 fnotched_dir=None, output_format='hdf', cut=0.5):
        self.db = io.DBManager(dbname)
        self.dbname = dbname
        self.scope = scope
        self.min_distance = min_distance
        self.eps = eps
        self.cut = cut
        self.fnotched_dir = fnotched_dir
        self.confusion = []
        self.output_format = output_format

        self.setup_folders()

    def setup_folders(self):
        "Setup folder paths and create them if required."
        fnotched_dir = self.fnotched_dir
        if self.fnotched_dir is None:
            fnotched_dir = Path(io.data_root) / 'output'
        self.fnotched_dir = Path(fnotched_dir)
        self.fnotched_dir.mkdir(exist_ok=True)

        # storage path for the clustered data before fnotching
        output_dir_clustered = self.fnotched_dir.with_name(
                                    self.fnotched_dir.stem+'_clustered')
        output_dir_clustered.mkdir(exist_ok=True)
        self.output_dir_clustered = output_dir_clustered

        # storage path for the final catalog after applying `cut`
        cut_dir = fnotched_dir.with_name(
            fnotched_dir.stem + '_cut_{:.1f}'.format(self.cut))
        cut_dir.mkdir(exist_ok=True)
        self.cut_dir = cut_dir

    @property
    def n_clustered_fans(self):
        "int : Number of clustered fans."
        return len(self.clustered_fans)

    @property
    def n_clustered_blotches(self):
        "int : Number of clustered blotches."
        return len(self.clustered_blotches)

    def cluster_data(self, data):
        """Basic clustering.

        For each fan and blotch markings in `data` a DBScanner object is created
        that executes the actual clustering. Depending on `scope`, this could be over
        marking data for one image_id only, or for all data for one HiRISE image_name.

        Parameters
        ----------
        data : pandas.DataFrame
            containing both fan and blotch data to be clustered.
        """
        logging.debug('ClusterManager: cluster_data()')
        clustered_blotches = []
        clustered_fans = []
        for kind in ['fan', 'blotch']:
            # filter for the marking for `kind`
            markings = data[data.marking == kind]
            if len(markings) == 0:
                continue
            dbscanner = DBScanner(markings, kind, eps=self.eps, scope=self.scope)
            self.confusion.append((self.data_id, kind, len(markings),
                                   dbscanner.n_reduced_data,
                                   dbscanner.n_rejected))
            if kind == 'fan':
                clustered_fans.extend(dbscanner.reduced_data)
            else:
                clustered_blotches.extend(dbscanner.reduced_data)
        self.clustered_fans = clustered_fans
        self.clustered_blotches = clustered_blotches

    def do_the_fnotch(self):
        """Combine fans and blotches if necessary.

        Use `min_distance` as criterion for linear algebraic distance between average cluster
        markings to determine if they belong to a Fnotch, a chimera object of indecision
        between a Fan and a Blotch, to be decided later in the process by applying a `cut`
        on the resulting Fnotch objects.

        See Also
        --------
        markings.Fnotch : The Fnotch object with a `get_marking` method for a `cut` value.

        """
        logging.debug("CM: do_the_fnotch")
        from numpy.linalg import norm
        n_close = 0
        fnotches = []
        blotches = []
        fans = []
        for blotch in self.clustered_blotches:
            for fan in self.clustered_fans:
                delta = blotch.center - fan.midpoint
                if norm(delta) < self.min_distance:
                    fnotch_value = calc_fnotch(fan.n_members, blotch.n_members)
                    fnotch = markings.Fnotch(fnotch_value, fan, blotch)
                    fnotch.n_fan_members = fan.n_members
                    fnotch.n_blotch_members = blotch.n_members
                    fnotches.append(fnotch)
                    n_close += 1
                    blotch.saved = True
                    fan.saved = True
            # only after going through all fans for this one blotch, I can store it as an
            # unfnotched blotch:
            if not blotch.saved:
                blotches.append(blotch)
        # I have to wait until the loop over blotches is over, before I know that a fan really
        # never was matched with a blotch, before I store it as an unfnotched Fan.
        for fan in self.clustered_fans:
            if not fan.saved:
                fans.append(fan)
                fan.saved = True

        self.fnotches = fnotches
        self.fnotched_blotches = blotches
        self.fnotched_fans = fans

    def execute_pipeline(self, data):
        """Execute the standard list of methods for catalog production.

        Parameters
        ----------
        data : pandas.DataFrame
            The dataframe containing the data to be clustered.
        """
        self.cluster_data(data)
        self.do_the_fnotch()
        logging.debug("Clustering and fnotching completed.")
        self.store_output()
        self.apply_fnotch_cut()

    def cluster_image_id(self, image_id):
        """Process the clustering for one image_id.

        Parameters
        ----------
        image_id : str
            Planetfour `image_id`
        """
        logging.info("Clustering data for {}".format(image_id))
        self.data_id = image_id
        self.p4id = markings.ImageID(image_id, self.dbname)
        self.execute_pipeline(self.p4id.data)

    def cluster_image_name(self, image_name):
        """Process the clustering and fnoching pipeline for a HiRISE image_name."""
        logging.info("Clustering data for {}".format(image_name))
        data = self.db.get_image_name_markings(image_name)
        self.data_id = image_name
        self.execute_pipeline(data)

    def store_output(self):
        "Write out the clustered and fnotched data."

        logging.debug('CM: Writing output files.')
        logging.debug('CM: Output dir: {}'.format(self.fnotched_dir))
        outfnotch = self.data_id + '_fnotches'
        outblotch = self.data_id + '_blotches'
        outfan = self.data_id + '_fans'
        # first write the fnotched data
        for outfname, outdata in zip([outfnotch, outblotch, outfan],
                                     [self.fnotches, self.fnotched_blotches,
                                      self.fnotched_fans]):
            if len(outdata) == 0:
                continue
            outpath = self.fnotched_dir / outfname
            series = [cluster.store() for cluster in outdata]
            df = pd.DataFrame(series)
            self.save(df, outpath)
        # store the unfnotched data as well:
        for outfname, outdata in zip([outblotch, outfan],
                                     [self.clustered_blotches,
                                      self.clustered_fans]):
            if len(outdata) == 0:
                continue
            outpath = self.output_dir_clustered / outfname
            series = [cluster.store() for cluster in outdata]
            df = pd.DataFrame(series)
            self.save(df, outpath)

    def cluster_all(self):
        image_names = self.db.image_names
        ft = FloatText()
        display(ft)
        for i, image_name in enumerate(image_names):
            perc = 100 * i / len(image_names)
            # print('{:.1f}'.format())
            ft.value = round(perc, 1)
            self.cluster_image_name(image_name)

    def report(self):
        print("Fnotches:", len(self.fnotches))
        print("Fans:", len(self.fnotched_fans))
        print("Blotches:", len(self.fnotched_blotches))

    @property
    def confusion_data(self):
        return pd.DataFrame(self.confusion, columns=['image_name', 'kind',
                                                     'n_markings',
                                                     'n_cluster_members',
                                                     'n_rejected'])

    def save_confusion_data(self, fname):
        self.confusion_data.to_csv(fname)

    def get_newfans_newblotches(self):
        df = self.resman.fnotchdf

        # apply Fnotch method `get_marking` with given cut.
        final_clusters = df.apply(markings.Fnotch.from_series, axis=1).\
            apply(lambda x: x.get_marking(self.cut))

        def filter_for_fans(x):
            if isinstance(x, markings.Fan):
                return x

        def filter_for_blotches(x):
            if isinstance(x, markings.Blotch):
                return x

        # now need to filter for whatever object was returned by Fnotch.get_marking
        self.newfans = final_clusters[
            final_clusters.apply(filter_for_fans).notnull()]
        self.newblotches = final_clusters[
            final_clusters.apply(filter_for_blotches).notnull()]

    def save(self, obj, path):
        obj.to_hdf(path.with_suffix('.hdf'), 'df')
        obj.to_csv(path.with_suffix('.csv'))

    def apply_fnotch_cut(self, id_):
        self.resman = plotting.ResultManager(id_, self.fnotched_dir)

        self.get_newfans_newblotches()

        if len(self.newfans) > 0:
            newfans = self.newfans.apply(lambda x: x.store())
            try:
                completefans = pd.DataFrame(
                    self.resman.fandf()).append(newfans, ignore_index=True)
            except OSError:
                completefans = newfans
        else:
            completefans = self.resman.fandf()
        if len(self.newblotches) > 0:
            newblotches = self.newblotches.apply(lambda x: x.store())
            try:
                completeblotches = pd.DataFrame(
                    self.resman.blotchdf()).append(newblotches, ignore_index=True)
            except OSError:
                completeblotches = newblotches
        else:
            completeblotches = self.resman.blotchdf()
        outpath = self.cut_dir
        self.save(completefans, str(outpath / self.resman.fanfile().name))
        self.save(completeblotches, str(outpath / self.resman.blotchfile().name))


def get_mean_position(fan, blotch, scope):
    if scope == 'hirise':
        columns = ['hirise_x', 'hirise_y']
    else:
        columns = ['x', 'y']

    df = pd.DataFrame([fan.data[columns], blotch.data[columns]])
    return df.mean()


def calc_fnotch(nfans, nblotches):
    return (nfans)/(nfans+nblotches)


def gold_star_plotter(gold_id, axis, blotches=True, kind='blotches'):
    for goldstar, color in zip(markings.gold_members,
                               markings.gold_plot_colors):
        if blotches:
            gold_id.plot_blotches(user_name=goldstar, ax=axis,
                                  user_color=color)
        if kind == 'fans':
            gold_id.plot_fans(user_name=goldstar, ax=axis, user_color=color)
        markings.gold_legend(axis)


def is_catalog_production_good():
    from pandas.core.index import InvalidIndexError
    db = DBManager(get_current_database_fname())
    not_there = []
    invalid_index = []
    value_error = []
    for image_name in db.image_names:
        try:
            ResultManager(image_name)
        except InvalidIndexError:
            invalid_index.append(image_name)
        except ValueError:
            value_error.append(image_name)
        except:
            not_there.append(image_name)
    if len(value_error) == 0 and len(not_there) == 0 and\
            len(invalid_index) == 0:
        return True
    else:
        return False


def main():
    gold_ids = io.common_gold_ids()

    p4img = markings.ImageID(gold_ids[10])
    golddata = p4img.data[p4img.data.user_name.isin(markings.gold_members)]
    golddata = golddata[golddata.marking == 'fan']
    # citizens = set(p4img.data.user_name) - set(markings.gold_members)

    # create plot window
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))
    fig.tight_layout()
    axes = ax.flatten()

    # fill images, 0 and 2 get it automatically
    for i in [1, 3]:
        p4img.show_subframe(ax=axes[i])

    # remove pixel coord axes
    for ax in axes:
        ax.axis('off')

    # citizen stuff
    p4img.plot_fans(ax=axes[0])
    axes[0].set_title('Citizen Markings')
    DBScanner(p4img.get_fans(), 'fan', ax=axes[1], eps=7, min_samples=5,
              linestyle='-')
    axes[1].set_title('All citizens clusters (including science team)')

    # gold stuff
    gold_star_plotter(p4img, axes[2], fans=True, blotches=False)
    axes[2].set_title('Science team markings')
    DBScanner(golddata, 'fan', ax=axes[1], min_samples=2, eps=11,
              linestyle='--')
    axes[3].set_title('Science team clusters')

    plt.show()


if __name__ == '__main__':
    main()
