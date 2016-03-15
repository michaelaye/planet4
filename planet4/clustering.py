"""Managing clustering, fnotching and cut application here."""
from __future__ import division, print_function

import importlib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.stats import circmean
import numpy as np
from . import io, markings
from .dbscan import DBScanner
import logging


importlib.reload(logging)
logpath = Path.home() / 'p4reduction.log'
logging.basicConfig(filename=str(logpath), filemode='w', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

matplotlib.style.use('bmh')


class ClusteringManager(object):

    """Control class to manage the clustering pipeline.

    Parameters
    ----------
    dbname : str or pathlib.Path, optional
        Path to the database used by DBManager. If not provided, DBManager
        will find the most recent one and use that.
    scope : {'hirise', 'planet4'}
        Switch to control in what coordinates the clustering is happening.
        'hirise' is required to automatically take care of tile overlaps, while
        'planet4' is required to be able to check the quality of single
        Planet4 tiles.
    min_distance : int
        Parameter to control the distance below which fans and blotches are
        determined to be 2 markings of the same thing, creating a FNOTCH
        chimera. The ratio between blotch and fan marks determines a fnotch
        value that can be used as a discriminator for the final object catalog.
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
    current_coords : list
        List of coordinate columns currently used for clustering
    current_markings : pandas.DataFrame
        Dataframe with the currently to-be-clustered marking data
    reduced_data : dictionary
        Stores reduced average cluster markings
    n_clustered_fans
    n_clustered_blotches
    output_dir_clustered : pathlib.Path
        Path to full clustering results, without removal of fnotched clusters.
    cut_dir : pathlib.Path
        Path to final fan and blotch clusters, after applying `cut`.
    """

    def __init__(self, dbname=None, scope='hirise', min_distance=10, eps=10,
                 fnotched_dir=None, output_format='csv', cut=0.5,
                 include_angle=True, id_=None, pm=None,
                 include_distance=False, include_radius=False):
        self.db = io.DBManager(dbname)
        self.dbname = dbname
        self.scope = scope
        self.min_distance = min_distance
        self.eps = eps
        self.cut = cut
        self.include_angle = include_angle
        self.include_distance = include_distance
        self.include_radius = include_radius
        self.confusion = []
        self.output_format = output_format

        # to be defined at runtime:
        self.current_coords = None
        self.current_markings = None
        self.reduced_data = None
        self.fnotches = None
        self.fnotched_blotches = None
        self.fnotched_fans = None
        self.p4id = None
        self.newfans = None
        self.newblotches = None

        if pm is not None:
            self.pm = pm
        else:
            self.pm = io.PathManager(fnotched_dir, id_=id_,
                                     suffix='.'+output_format)
        self.pm.setup_folders()

    def __getattr__(self, name):
        """Search for attributes in PathManager if not offered in this object."""
        return getattr(self.pm, name)

    @property
    def n_clustered_fans(self):
        """int : Number of clustered fans."""
        return len(self.clustered_data['fan'])

    @property
    def n_clustered_blotches(self):
        """int : Number of clustered blotches."""
        return len(self.clustered_data['blotch'])

    def pre_processing(self, data, kind):
        # filter for the marking for `kind`
        marking_data = data[data.marking == kind]
        if len(marking_data) == 0:
            return None

        if self.scope == 'planet4':
            coords = ['x', 'y']
        elif self.scope == 'hirise':
            coords = ['image_x', 'image_y']

        if kind == 'fan':
            if self.include_distance:
                coords.append('distance')
            if self.include_angle:
                coords.append('angle')
        else:
            if self.include_radius:
                coords += ['radius_1', 'radius_2']
        # Determine the clustering input matrix
        current_X = marking_data[coords].values
        self.current_coords = coords
        self.current_markings = marking_data
        return current_X

    def post_processing(self, dbscanner, kind):
        """Create mean objects out of cluster label members.

        Note: I take the image_id of the marking of the first member of
        the cluster as image_id for the whole cluster. In rare circumstances,
        this could be wrong for clusters in the overlap region.

        Stores output in self.reduced_data dictionary

        Parameters
        ----------
        dbscanner : dbscan.DBScanner
            DBScanner object
        kind : {'fan', 'blotch'}
            current kind of marking to post-process.

        """
        if kind == 'fan':
            cols = markings.Fan.to_average
            Marking = markings.Fan
        elif kind == 'blotch':
            cols = markings.Blotch.to_average
            Marking = markings.Blotch

        reduced_data = []
        data = self.current_markings
        for cluster_members in dbscanner.reduced_data:
            clusterdata = data[cols].iloc[cluster_members]
            meandata = clusterdata.mean()
            meandata.angle = np.rad2deg(circmean(np.deg2rad(clusterdata.angle)))
            cluster = Marking(meandata)
            # storing n_members into the object for later.
            cluster.n_members = len(cluster_members)
            # storing this saved marker for later in ClusteringManager
            cluster.saved = False
            # store the image_id from first cluster member for whole cluster
            try:
                image_id = data['image_id'].iloc[cluster_members][0]
            # inelegant fudge to account for Categories not having iloc.
            except KeyError:
                image_id = data['image_id'].iloc[cluster_members].values[0]
            cluster.image_id = image_id

            reduced_data.append(cluster)

        self.reduced_data[kind] = reduced_data
        logging.debug("Reduced data to %i %s(e)s.", len(reduced_data), kind)

    def cluster_data(self, data):
        """Basic clustering.

        For each fan and blotch markings in `data` a DBScanner object is
        created that executes the actual clustering. Depending on `scope`, this
        could be over marking data for one image_id only, or for all data for
        one HiRISE image_name.

        Parameters
        ----------
        data : pandas.DataFrame
            containing both fan and blotch data to be clustered.
        """
        logging.debug('ClusterManager: cluster_data()')
        # reset stored clustered data
        self.reduced_data = {}
        for kind in ['fan', 'blotch']:
            # self.include_angle = False if kind == 'blotch' else True
            current_X = self.pre_processing(data, kind)
            if current_X is not None:
                dbscanner = DBScanner(current_X, eps=self.eps)
            else:
                self.reduced_data[kind] = []
                continue
            # storing of clustered data happens in here:
            self.post_processing(dbscanner, kind)
            self.confusion.append((self.pm.id_, kind,
                                   len(self.current_markings),
                                   len(self.reduced_data[kind]),
                                   dbscanner.n_rejected))

    def do_the_fnotch(self):
        """Combine fans and blotches if necessary.

        Use `min_distance` as criterion for linear algebraic distance between
        average cluster markings to determine if they belong to a Fnotch, a
        chimera object of indecision between a Fan and a Blotch, to be decided
        later in the process by applying a `cut` on the resulting Fnotch
        objects.

        See Also
        --------
        markings.Fnotch : The Fnotch object with a `get_marking` method for a
            `cut` value.
        """
        # check first if both blotchens and fans were found, if not, we don't
        # need to fnotch.
        if not all(self.reduced_data.values()):
            self.fnotches = []
            self.fnotched_blotches = self.reduced_data['blotch']
            self.fnotched_fans = self.reduced_data['fan']
            return

        logging.debug("CM: do_the_fnotch")
        from numpy.linalg import norm
        n_close = 0
        fnotches = []
        blotches = []
        fans = []
        for blotch in self.reduced_data['blotch']:
            for fan in self.reduced_data['fan']:
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
        for fan in self.reduced_data['fan']:
            if not fan.saved:
                fans.append(fan)
                fan.saved = True

        self.fnotches = fnotches
        self.fnotched_blotches = blotches
        self.fnotched_fans = fans
        logging.debug("CM: do_the_fnotch: Found %i fnotches.", n_close)

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

    def cluster_image_id(self, image_id, data=None):
        """Process the clustering for one image_id.

        Parameters
        ----------
        image_id : str
            Planetfour `image_id`
        """
        image_id = io.check_and_pad_id(image_id)
        logging.info("Clustering data for %s", image_id)
        self.pm.id_ = image_id
        if data is None:
            self.p4id = markings.ImageID(image_id, self.dbname)
            data = self.p4id.data
        self.execute_pipeline(data)

    def cluster_image_name(self, image_name, data=None):
        """Process the clustering and fnoching pipeline for a HiRISE image_name."""
        logging.info("Clustering data for %s", image_name)
        if data is None:
            data = self.db.get_image_name_markings(image_name)
        self.pm.id_ = image_name
        self.execute_pipeline(data)

    def store_output(self):
        """Write out the clustered and fnotched data."""
        logging.debug('CM: Writing output files.')
        logging.debug('CM: Output dir: %s', self.fnotched_dir)
        # first write the fnotched data
        for outfname, outdata in zip(['fnotchfile', 'blotchfile', 'fanfile'],
                                     [self.fnotches, self.fnotched_blotches,
                                      self.fnotched_fans]):
            if len(outdata) == 0:
                continue
            # get the path from PathManager object
            series = [cluster.store() for cluster in outdata]
            df = pd.DataFrame(series)
            self.save(df, getattr(self.pm, outfname))
        # store the unfnotched data as well:
        outdir = self.output_dir_clustered
        outdir.mkdir(exist_ok=True)
        for outfname, outdata in zip(['reduced_blotchfile', 'reduced_fanfile'],
                                     [self.reduced_data['blotch'],
                                      self.reduced_data['fan']]):
            if len(outdata) == 0:
                continue
            series = [cluster.store() for cluster in outdata]
            df = pd.DataFrame(series)
            self.save(df, getattr(self.pm, outfname))

    def cluster_all(self):
        image_names = self.db.image_names
        for image_name in image_names:
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
        df = self.pm.fnotchdf

        # check if we got a fnotch dataframe. If not, we assume none were found.
        if df is None:
            self.newfans = self.newblotches = []
            return

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
        try:
            obj.to_hdf(str(path.with_suffix('.hdf')), 'df')
            obj.to_csv(str(path.with_suffix('.csv')), index=False)
        # obj could be NoneType if no blotches or fans were found. Catching it here.
        except AttributeError:
            pass

    def apply_fnotch_cut(self, cut=None):
        if cut is None:
            cut = self.cut

        # storage path for the final catalog after applying `cut`
        # PathManager self.pm is doing that.
        self.pm.create_cut_folder(cut)

        self.get_newfans_newblotches()

        if len(self.newfans) > 0:
            newfans = self.newfans.apply(lambda x: x.store())
            try:
                completefans = pd.DataFrame(
                    self.pm.fandf).append(newfans, ignore_index=True)
            except OSError:
                completefans = newfans
        else:
            completefans = self.pm.fandf
        if len(self.newblotches) > 0:
            newblotches = self.newblotches.apply(lambda x: x.store())
            try:
                completeblotches = pd.DataFrame(
                    self.pm.blotchdf).append(newblotches, ignore_index=True)
            except OSError:
                completeblotches = newblotches
        else:
            completeblotches = self.pm.blotchdf
        self.save(completefans, self.pm.final_fanfile)
        self.save(completeblotches, self.final_blotchfile)

######
# Functions
#####


def get_mean_position(fan, blotch, scope):
    if scope == 'hirise':
        columns = ['hirise_x', 'hirise_y']
    else:
        columns = ['x', 'y']

    df = pd.DataFrame([fan.data[columns], blotch.data[columns]])
    return df.mean()


def calc_fnotch(nfans, nblotches):
    return (nfans) / (nfans + nblotches)


def gold_star_plotter(gold_id, axis, kind='blotches'):
    for goldstar, color in zip(markings.gold_members,
                               markings.gold_plot_colors):
        if kind == 'blotches':
            gold_id.plot_blotches(user_name=goldstar, ax=axis,
                                  user_color=color)
        if kind == 'fans':
            gold_id.plot_fans(user_name=goldstar, ax=axis, user_color=color)
        markings.gold_legend(axis)


def is_catalog_production_good():
    from pandas.core.index import InvalidIndexError
    db = io.DBManager(io.get_current_database_fname())
    not_there = []
    invalid_index = []
    value_error = []
    for image_name in db.image_names:
        try:
            io.PathManager(image_name)
        except InvalidIndexError:
            invalid_index.append(image_name)
        except ValueError:
            value_error.append(image_name)
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
    # TODO: fix use syntax of DBScanner here
    # DBScanner(p4img.get_fans(), eps=7, min_samples=5)
    axes[1].set_title('All citizens clusters (including science team)')

    # gold stuff
    gold_star_plotter(p4img, axes[2], kind='fans')
    axes[2].set_title('Science team markings')
    # TODO: refactor for plotting version of DBSCanner
    # DBScanner(golddata, ax=axes[1], min_samples=2, eps=11,
    #           linestyle='--')
    axes[3].set_title('Science team clusters')

    plt.show()


if __name__ == '__main__':
    main()
