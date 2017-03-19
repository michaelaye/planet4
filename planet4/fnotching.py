import logging

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

from . import io, markings

logger = logging.getLogger(__name__)


def get_id_from_path(path):
    return path.parent.name


def get_clusters_in_path(path):
    "find csv files in path and combine into DataFrame"
    clusters = []
    id_ = get_id_from_path(path)
    for kind in ['fans', 'blotches']:
        try:
            df = pd.read_csv(str(path / f"L1A_{id_}_{kind}.csv"))
        except FileNotFoundError:
            df = pd.DataFrame()
        clusters.append(df)
    return clusters


def data_to_centers(df):
    if all(np.isnan(df.distance)):
        # only the blotch arrays have distance un-defined
        Marking = markings.Blotch
    elif all(np.isnan(df.radius_1)) and all(np.isnan(df.radius_2)):
        Marking = markings.Fan
    return np.vstack(Marking(row, scope='hirise').center for _, row in df.iterrows())


def fnotch_image_ids(obsid, eps=20):
    "Cluster each image_id for an obsid separately."
    # the clustering results were stored as L1A products
    pm = io.PathManager(obsid=obsid)
    paths = pm.get_obsid_paths('L1A')
    for path in paths:
        id_ = get_id_from_path(path)
        pm.id = id_
        # make sure the L1B folder exists
        pm.reduced_fanfile.parent.mkdir(parents=True, exist_ok=True)

        fans, blotches = get_clusters_in_path(path)
        if not any([fans.empty, blotches.empty]):
            logger.debug("Fnotching %s", id_)
            distances = cdist(data_to_centers(fans), data_to_centers(blotches))
            X, Y = np.where(distances < eps)
            # X are the indices along the fans input, Y for blotches respectively

            # loop over fans and blotches that are within `eps` pixels:
            fnotches = []
            for fan_loc, blotch_loc in zip(X, Y):
                fan = fans.iloc[[fan_loc]]
                blotch = blotches.iloc[[blotch_loc]]
                fnotches.append(markings.Fnotch(fan, blotch).data)

            # store the combined fnotches into one file. The `votes_ratio` is
            # stored as well, making it simple to filter/cut on these later for the
            # L1C product.
            try:
                pd.concat(fnotches).to_csv(pm.fnotchfile)
            except ValueError as e:
                if e.args[0].startswith("No objects to concatenate"):
                    logger.debug("No fnotches found for %s.", id_)
                else:
                    raise ValueError

            # write out the fans and blotches that where not within fnotching distance:
            fans.loc[set(fans.index) - set(X)].to_csv(pm.reduced_fanfile, index=False)
            blotches.loc[set(blotches.index) - set(Y)].to_csv(pm.reduced_blotchfile, index=False)
        elif fans.empty:
            logger.debug("Only blotches found.")
            blotches.to_csv(pm.reduced_blotchfile, index=False)
        elif blotches.empty:
            logger.debug("Only fans found.")
            fans.to_csv(pm.reduced_fanfile, index=False)


def get_slashed_for_path(path, pm):
    id_ = get_id_from_path(path)
    logger.debug("Slashing %s", id_)
    pm.id = id_
    try:
        fnotches = pm.fnotchdf
    except FileNotFoundError:
        return
    # apply cut
    slashed = fnotches[fnotches.vote_ratio > pm.cut]
    return slashed


def write_l1c(kind, slashed, pm):
    """Write the L1C for marking `kind`.

    Parameters
    ----------
    kind : {'fan', 'blotch'}
        P4 marking kind
    slashed : pd.DataFrame
        The remaining fnotch data after applying the cut
    pm : io.PathManager
        The PathManager for the current image_id
    """
    try:
        new_kinds = slashed.loc[[kind]]
    except KeyError:
        logger.debug("No %s in slashed dataframe.", kind)
        new_kinds = pd.DataFrame()
    l1c = getattr(pm, f"final_{kind}file")
    l1c.parent.mkdir(parents=True, exist_ok=True)
    try:
        # the pathmanager can read the csv files as well:
        old_kinds = getattr(pm, f"reduced_{kind}df")
    except FileNotFoundError:
        logger.debug('No old %s file.', kind)
        new_kinds.dropna(how='all', axis=1, inplace=True)
        new_kinds.to_csv(str(l1c), index=False)
    else:
        logger.debug("Combining. Writing to %s", str(l1c))
        combined = pd.concat([old_kinds, new_kinds], ignore_index=True)
        combined.dropna(how='all', axis=1, inplace=True)
        combined.to_csv(str(l1c), index=False)


def apply_cut(obsid, cut=0.5):
    """Loop over all image_id paths for an obsid and apply cut to fnotches.

    Parameters
    ----------
    obsid : str
        HiRISE obsid, i.e. P4 `image_name`
    cut : float, 0..1
        Value where to cut the vote_ratio of the fnotches.
    """
    pm = io.PathManager(obsid=obsid, cut=cut)
    paths = pm.get_obsid_paths('L1B')
    for path in paths:
        slashed = get_slashed_for_path(path, pm)
        if slashed is not None:
            for kind in ['fan', 'blotch']:
                write_l1c(kind, slashed, pm)
