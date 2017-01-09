"""Managing clustering, fnotching and cut application here."""
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import io, markings
from .dbscan import DBScanner, HDBScanner
from ._utils import get_average_object

logger = logging.getLogger(__name__)

matplotlib.style.use('bmh')


class NotEnoughMarkingData(Exception):
    def __init__(self):
        Exception.__init__(self, "Not enough data to cluster (< 3 items).")


def angle_to_xy(angles, kind):
    """Convert angles in degrees to coordinates on unit circle.

    Parameters
    ----------
    angles : np.array, list
        List of angles

    """
    y = np.sin(np.deg2rad(angles))
    out = [y]
    if kind == 'fan':
        x = np.cos(np.deg2rad(angles))
        out = [x] + out
    return np.vstack(out).T


def cluster_angles(data, kind, min_samples=3, eps_fanangle=20, eps_blotchangle=20):
    """ Cluster only by angle.

    Parameters
    ----------
    data : pd.dataframe
        the pandas dataframe with data
    kind : {'fan','blotch'}
        The marking kind currently used. Important because blotches are only clustered
        by their np.sin(angle), while fans also on np.cos(angle)
    min_samples : int
        Mininum sample number for DBSCAN
    eps_fanangle : int
        Angle in degree for allowed separation between fan angles
    eps_blotchangle : int
        Angle in degrees for allowed separation between blotch angles
    """
    # calculated value of euclidean distance of unit vector end points per degree
    dist_per_degree = 2 * np.pi / 360
    X = angle_to_xy(data.angle, kind)
    delta = eps_fanangle if kind == 'fan' else eps_blotchangle
    clusterer = DBScanner(X, eps=delta * dist_per_degree,
                          min_samples=min_samples)
    return clusterer.clustered_indices


class ClusteringManager(object):

    """Control class to manage the clustering pipeline.

    Parameters
    ----------
    dbname : str or pathlib.Path, optional
        Path to the database used by DBManager. If not provided, DBManager
        will find the most recent one and use that.
    fnotch_distance : int
        Parameter to control the distance below which fans and blotches are
        determined to be 2 markings of the same thing, creating a FNOTCH
        chimera. The ratio between blotch and fan marks determines a fnotch
        value that can be used as a discriminator for the final object catalog.
    eps : int
        Parameter to control the exclusion distance for the DBSCAN
    output_dir : str or pathlib.Path
        Path to folder where to store output. Default: io.data_root / 'output'
    output_format : {'hdf', 'csv', 'both'}
        Format to save the output in. Default: 'hdf'
    cut : float
        Value to apply for fnotch cutting.
    min_samples_factor : float
        Value to multiply the number of unique classifications per image_id with
        to determine the `min_samples` value for DBSCAN to use. Default: 0.1
    do_dynamic_min_samples : bool, default is False
        Switch to decide if `min_samples` is being dynamically calculated.

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
    reduced_data : dictionary
        Stores reduced average cluster markings
    n_clustered_fans
    n_clustered_blotches
    output_dir_clustered : pathlib.Path
        Path to full clustering results, without removal of fnotched clusters.
    cut_dir : pathlib.Path
        Path to final fan and blotch clusters, after applying `cut`.
    """

    def __init__(self, dbname=None, fnotch_distance=10, eps=10,
                 output_dir=None, output_format='csv', cut=0.5,
                 min_samples_factor=0.15,
                 include_angle=True, id_=None, pm=None,
                 do_dynamic_min_samples=False,
                 use_DBSCAN=True,
                 hdbscan_min_samples_diff=0,
                 min_samples=None,
                 proba_cut=0.0,
                 eps_fanangle=20,
                 eps_blotchangle=20,
                 include_radii=False,
                 only_core=True,
                 s23=False):
        self.db = io.DBManager(dbname, s23=s23)
        self.dbname = self.db.dbname
        self.fnotch_distance = fnotch_distance
        self.output_dir = output_dir
        self.output_format = output_format
        self.eps = eps
        self.cut = cut
        self.include_angle = include_angle
        self.confusion = []
        self.min_samples_factor = min_samples_factor
        self.do_dynamic_min_samples = do_dynamic_min_samples
        self.use_DBSCAN = use_DBSCAN
        self.hdbscan_min_samples_diff = hdbscan_min_samples_diff
        self.min_samples = min_samples
        self.proba_cut = proba_cut
        self.eps_fanangle = eps_fanangle
        self.eps_blotchangle = eps_blotchangle
        self.include_radii = include_radii
        self.only_core = only_core
        # to be defined at runtime:
        self.current_coords = None
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
            self.pm = io.PathManager(output_dir, id_=id_,
                                     suffix='.' + self.output_format)

    @property
    def n_clustered_fans(self):
        """int : Number of clustered fans."""
        return len(self.reduced_data['fan'])

    @property
    def n_clustered_blotches(self):
        """int : Number of clustered blotches."""
        return len(self.reduced_data['blotch'])

    def pre_processing(self):
        """Preprocess before clustering.

        Depending on the flags used when constructing this manager,
        different columns end up being clustered on.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with data to cluster on
        kind : str
            String indicating if to cluster on blotch or fan data
        """

        # add unit circle coordinates for angles
        marking_data = self.marking_data

        if len(marking_data) < 3:
            return None

        # basic coordinates to cluster on
        coords = ['x', 'y']

        if self.include_radii and self.kind == 'blotch':
            coords += ['radius_1', 'radius_2']

        # Determine the clustering input matrix
        current_X = marking_data[coords].values

        # store stuff for later
        self.current_coords = coords
        return current_X

    def cluster_angles(self, data):
        return cluster_angles(data, self.kind, self.min_samples,
                              self.eps_fanangle, self.eps_blotchangle)

    def post_processing(self):
        """Create mean objects out of cluster label members.

        Note: I take the image_id of the marking of the first member of
        the cluster as image_id for the whole cluster. In rare circumstances,
        this could be wrong for clusters in the overlap region.

        Stores output in self.reduced_data dictionary
        """
        kind = self.kind
        Marking = markings.Fan if kind == 'fan' else markings.Blotch
        cols = Marking.to_average

        reduced_data = []
        data = self.marking_data
        logger.debug("Post processing %s", kind)
        for xy_cluster in self.clusterer.clustered_indices:
            xy_clusterdata = data.loc[xy_cluster, cols + ['user_name']]
            logger.debug("N of members in this xy_cluster: %i", len(xy_clusterdata))
            # now sub-cluster on angles, to distinguish between different fan directions
            # and to increase precision on blotch alignments
            angle_clustered_indices = self.cluster_angles(xy_clusterdata)
            if len(angle_clustered_indices) > 0:
                logger.debug("Entering angle clustering")
            for angle_cluster in angle_clustered_indices:
                angle_clusterdata = xy_clusterdata.loc[angle_cluster, cols + ['user_name']]
                # if the same user is inside one cluster, just take
                # the first entry per user:
                filtered = angle_clusterdata.groupby('user_name').first()
                # still only taking stuff if it has more than min_samples markings.
                logger.debug("N of members of this angle cluster before filtering: %i",
                             len(angle_clusterdata))
                logger.debug("N of members of this angle cluster after filtering: %i",
                             len(filtered))
                if len(filtered) < self.min_samples:
                    logger.debug("Throwing away this cluster for < min_samples")
                    continue
                logger.debug("Calculating mean %s object.", kind)
                meandata = self.get_average_object(angle_clusterdata)
                # This returned a pd.Series object I can add more info to now:
                # storing n_votes into the object for later.
                meandata['n_votes'] = len(angle_cluster)
                meandata['image_id'] = self.pm.id_
                meandata['image_name'] = self.marking_data.image_name.values[0]
                # converting to dataframe and clean data format (rows=measurements)
                reduced_data.append(meandata.to_frame().T)
        logger.debug("Length of reduced data (total number clusters found): %i",
                     len(reduced_data))

        self.reduced_data[kind] = reduced_data
        logger.debug("Reduced data to %i %s(e)s.", len(reduced_data), kind)

    def cluster_data(self):
        """Basic clustering.

        For each fan and blotch markings in `data` a DBScanner object is
        created that executes the actual clustering.
        To be able to apply dynamic calculation of `min_samples`, this will
        always be on 'planet4' tile coordinates.

        Parameters
        ----------
        data : pandas.DataFrame
            containing both fan and blotch data to be clustered.
        """
        logger.debug('ClusterManager: cluster_data()')
        # reset stored clustered data
        self.reduced_data = {}

        # Calculate the unique classification_ids so that the mininum number of
        # samples for DBScanner can be calculated (15 % currently)
        # use only class_ids that actually contain fan and blotch markings
        f1 = self.data.marking == 'fan'   # this creates a boolean filter
        f2 = self.data.marking == 'blotch'
        # combine filters with logical OR:
        n_classifications = self.data[f1 | f2].classification_id.nunique()

        if self.do_dynamic_min_samples:
            min_samples = round(self.min_samples_factor * n_classifications)
            # ensure that min_samples is at least 3:
            min_samples = max(min_samples, 3)
            self.min_samples = min_samples
        elif self.min_samples is None:
            # 3 turned out to be a well working default min_samples requirement
            min_samples = 3
            self.min_samples = min_samples
        else:
            min_samples = self.min_samples

        for kind in ['fan', 'blotch']:
            # what is included for clustering is decided in pre_processing
            self.marking_data = self.data[self.data.marking == kind]
            self.kind = kind
            current_X = self.pre_processing()
            if current_X is not None:
                if self.use_DBSCAN:
                    logger.debug("Using DBSCAN")
                    clusterer = DBScanner(current_X, eps=self.eps,
                                          min_samples=min_samples,
                                          only_core=self.only_core)
                else:
                    logger.debug("Using HDBSCAN")
                    hdbscan_min_samples = self.min_samples - \
                        self.hdbscan_min_samples_diff
                    clusterer = HDBScanner(current_X,
                                           min_cluster_size=min_samples,
                                           min_samples=hdbscan_min_samples,
                                           proba_cut=self.proba_cut,
                                           only_core=self.only_core)
                # store the scanner object in both cases into `self`
                self.clusterer = clusterer
            else:
                # current_X is empty so store empty results and skip to next `kind`
                self.reduced_data[kind] = []
                continue
            # storing of clustered data happens in here:
            self.post_processing()
            self.confusion.append((self.pm.id_, kind,
                                   len(self.marking_data),
                                   len(self.reduced_data[kind]),
                                   clusterer.n_rejected))
        self.n_classifications = n_classifications
        logger.info("n_classifications: %i", self.n_classifications)
        logger.info("min_samples: %i", self.min_samples)

    def cluster_image_id(self, image_id, data=None):
        """Process the clustering for one image_id.

        Parameters
        ----------
        image_id : str
            Planetfour `image_id`
        data : pd.DataFrame, optional
            Dataframe with data for this clustering run
        """
        image_id = io.check_and_pad_id(image_id)
        logger.info("Clustering data for %s", image_id)
        self.pm.id_ = image_id
        if data is None:
            self.data = self.db.get_image_id_markings(image_id)
            logger.debug("DB used: %s", self.dbname)
        else:
            self.data = data
        self.cluster_data()
        logger.debug("Clustering completed.")
        self.store_clustered()

    def cluster_obsid(self, *args, **kwargs):
        "Alias to cluster_image_name."
        self.cluster_image_name(*args, **kwargs)

    def store_clustered(self):
        "Store the unfnotched data."
        outdir = self.pm.output_dir_clustered
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

    @property
    def confusion_data(self):
        return pd.DataFrame(self.confusion,
                            columns=['image_name', 'kind',
                                     'n_markings',
                                     'n_cluster_members',
                                     'n_rejected'])

    def save(self, obj, path):
        try:
            if self.output_format in ['hdf', 'both']:
                obj.to_hdf(str(path.with_suffix('.hdf')), 'df')
            if self.output_format in ['csv', 'both']:
                obj.to_csv(str(path.with_suffix('.csv')), index=False)
        # obj could be NoneType if no blotches or fans were found. Catching it here.
        except AttributeError:
            pass

    def save_confusion_data(self, fname):
        self.confusion_data.to_csv(fname)


######
# Functions
#####

def gold_star_plotter(gold_id, axis, kind='blotches'):
    """Plot gold data."""
    for goldstar, color in zip(markings.gold_members,
                               markings.gold_plot_colors):
        if kind == 'blotches':
            gold_id.plot_blotches(user_name=goldstar, ax=axis,
                                  user_color=color)
        if kind == 'fans':
            gold_id.plot_fans(user_name=goldstar, ax=axis, user_color=color)
        markings.gold_legend(axis)


def is_catalog_production_good():
    """A simple quality check for the catalog production."""
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
    """Exeucute gold data plotting by default. Should probably moved elsewhere.

    Also, most likely not working currently.
    """
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
