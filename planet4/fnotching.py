import logging

import pandas as pd
from numpy.linalg import norm

from . import markings

logger = logging.getLogger(__name__)


def calc_fnotch(nfans, nblotches):
    """Calculate the fnotch value (or fan-ness)."""
    return (nfans) / (nfans + nblotches)


def do_the_fnotch(self):
    """Combine fans and blotches if necessary.

    Use `fnotch_distance` as criterion for linear algebraic distance between
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
        logger.debug("CM: no fnotching required.")
        self.fnotches = []
        self.fnotched_blotches = self.reduced_data['blotch']
        self.fnotched_fans = self.reduced_data['fan']
        return

    logger.debug("CM: do_the_fnotch")
    n_close = 0
    fnotches = []
    blotches = []
    fans = []
    for blotch in self.reduced_data['blotch']:
        for fan in self.reduced_data['fan']:
            delta = blotch.center - fan.midpoint
            if norm(delta) < self.fnotch_distance:
                fnotch_value = calc_fnotch(fan.n_members, blotch.n_members)
                fnotch = markings.Fnotch(fnotch_value, fan, blotch,
                                         scope='hirise')
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
    logger.debug("CM: do_the_fnotch: Found %i fnotches.", n_close)


def store_fnotched(self):
    """Write out the clustered and fnotched data."""
    logger.debug('CM: Writing output files.')
    logger.debug('CM: Output dir: %s', self.datapath)

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


# def execute_pipeline(self, data):
#     """Execute the standard list of methods for catalog production.
#
#     Parameters
#     ----------
#     data : pandas.DataFrame
#         The dataframe containing the data to be clustered.
#     """
#     self.cluster_data(data)
#     # self.do_the_fnotch()
#     # self.apply_fnotch_cut()
#     self.store_clustered()
#     # self.store_fnotched()


def get_newfans_newblotches(self):
    logger.debug("Executing get_newfans_newblotches")
    df = self.pm.fnotchdf

    # check if we got a fnotch dataframe. If not, we assume none were found.
    if df is None:
        logger.debug("No fnotches found on disk.")
        self.newfans = []
        self.newblotches = []
        return

    # apply Fnotch method `get_marking` with given cut.
    fnotches = df.apply(markings.Fnotch.from_series, axis=1,
                        args=('hirise',))
    final_clusters = fnotches.apply(lambda x: x.get_marking(self.cut))

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


def apply_fnotch_cut(self, cut=None):
    logger.debug("Executing apply_fnotch_cut")
    if cut is None:
        cut = self.cut

    # storage path for the final catalog after applying `cut`
    # PathManager self.pm is doing that.
    self.pm.get_cut_folder(cut)

    self.get_newfans_newblotches()

    if len(self.newfans) > 0:
        newfans = self.newfans.apply(lambda x: x.store())
        try:
            completefans = pd.DataFrame(
                self.pm.fandf).append(newfans, ignore_index=True)
        except OSError:
            completefans = newfans
        logger.debug("No of fans now: %i" % len(completefans))
    else:
        logger.debug("Apply fnotch cut: No new fans found.")
        completefans = self.pm.fandf
    if len(self.newblotches) > 0:
        newblotches = self.newblotches.apply(lambda x: x.store())
        try:
            completeblotches = pd.DataFrame(
                self.pm.blotchdf).append(newblotches, ignore_index=True)
        except OSError:
            completeblotches = newblotches
        logger.debug("No of blotches now: %i" % len(completeblotches))
    else:
        logger.debug('Apply fnotch cut: no blotches survived.')
        completeblotches = self.pm.blotchdf
    self.save(completefans, self.pm.final_fanfile)
    self.save(completeblotches, self.final_blotchfile)
    logger.debug("Finished apply_fnotch_cut.")


def cluster_image_name(self, image_name, data=None):
    """Process the clustering and fnotching pipeline for a HiRISE image_name.

    Parameters
    ----------
    image_name : str
        HiRISE image_name (= obsid in HiLingo) to cluster on.
        Used for storing the data in `obsid` indicated subfolders.
    data : pd.DataFrame
        Dataframe containing the data to cluster on.
    """
    logger.info("Clustering data for %s", image_name)
    if data is None:
        namedata = self.db.get_image_name_markings(image_name)
    else:
        namedata = data
    image_ids = namedata.image_id.unique()
    logger.debug("Found %i image_ids.", len(image_ids))
    self.pm.image_name = image_name
    for image_id in image_ids:
        logger.debug('Clustering image_id %s', image_id)
        self.pm.id_ = image_id
        self.data = namedata[namedata.image_id == image_id]
        self.cluster_data()
        self.store_clustered()
