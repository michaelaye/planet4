import logging
import math
import random

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist

from . import io, markings

logger = logging.getLogger(__name__)


def get_id_from_path(path):
    return path.parent.name


def get_clusters_in_path(path):
    """Find csv files in path and combine into DataFrame.

    Parameters
    ----------
    path : str, pathlib.Path
        Path in where to search for L1A csv files.

    Returns
    -------
    clusters : list
        List with 2 pd.DataFrames
    """
    clusters = []
    id_ = get_id_from_path(path)
    for kind in ["fans", "blotches"]:
        try:
            df = pd.read_csv(str(path / f"{id_}_L1A_{kind}.csv"))
        except FileNotFoundError:
            df = None
        clusters.append(df)
    return clusters


def data_to_centers(df, kind, scope="hirise"):
    """Convert a dataframe with marking data to an array of center coords.

    Parameters
    ----------
    df : pd.dataframe
        Dataframe with either fan or blotch marking data. It probes itself
        which one it is by looking at if distances and radii are defined.
    kind : {'fan', 'blotch'}
    Returns
    -------
    np.array
        Array with the center coordinates, dimensions: (rows, 2)
    """
    if kind == "blotch":
        # only the blotch arrays have distance un-defined
        Marking = markings.Blotch
    else:
        Marking = markings.Fan
    return np.vstack(Marking(row, scope=scope).center for _, row in df.iterrows())


def calc_indices_from_index(n, c):
    """calculate source indices from condensed distance matrix.

    The `pdist` function returns its measurements in a (01, 02, 03, 12, 13...)
    fashion and this function can be used to get out the original coordinates
    of the 2 inputs.

    Parameters
    ----------
    n : int
        Length of condensed matrix
    c : int
        Index of the distance value of interest

    Returns
    -------
    int, int
        Coordinate pair of the 2 indices that were used to calculate distance
        at index c of the condensed distance matrix.
    """
    n = math.ceil(math.sqrt(2 * n))
    ti = np.triu_indices(n, 1)
    return ti[0][c], ti[1][c]


def remove_opposing_fans(fans, eps=20):
    """Find fans that have opposite orientation and remove lower voted one.

    First check if any fans are close enough to be fnotched (same criteria
    as blotch-fan fnotching), then check if any of those have opposite orientation.
    Delete the one with lower votes.
    If number of votes is equal, take a random choice.

    Parameters
    ----------
    fans : pd.DataFrame
        Fan marking data
    eps : int

    Returns
    -------
    pd.DataFrame
        Data with opposing fans removed.
    """
    distances = pdist(data_to_centers(fans, "fan"))
    close_indices = np.where(distances < eps)[0]
    ind_to_remove = []
    for index in close_indices:
        fan_indices = calc_indices_from_index(len(distances), index)
        # use squeeze to force creation of pd.Series
        f1 = fans.iloc[fan_indices[0]].squeeze()
        f2 = fans.iloc[fan_indices[1]].squeeze()
        angle_diff = f1.angle - f2.angle
        # if they differ by between 175 and 185:
        if abs(angle_diff - 180) < 5:
            if f1.n_votes < f2.n_votes:
                ind_to_remove.append(fan_indices[0])
            elif f1.n_votes > f2.n_votes:
                ind_to_remove.append(fan_indices[1])
            else:
                ind_to_remove.append(fan_indices[random.randint(0, 1)])
    return fans.drop(ind_to_remove)


def fnotch_image_ids(obsid, eps=20, savedir=None, scope="hirise"):
    "Cluster each image_id for an obsid separately."
    # the clustering results were stored as L1A products
    pm = io.PathManager(obsid=obsid, datapath=savedir)
    paths = pm.get_obsid_paths("L1A")
    if len(paths) == 0:
        logger.warning("No paths to fnotch found for %s", obsid)
    for path in paths:
        id_ = get_id_from_path(path)
        pm.id = id_
        # make sure the L1B folder exists
        pm.reduced_fanfile.parent.mkdir(parents=True, exist_ok=True)

        fans, blotches = get_clusters_in_path(path)
        if fans is not None and len(fans) > 1:
            # clean up fans with opposite angles
            fans = remove_opposing_fans(fans)
        if not any([fans is None, blotches is None]):
            logger.debug("Fnotching %s", id_)
            distances = cdist(
                data_to_centers(fans, "fan", scope=scope),
                data_to_centers(blotches, "blotch", scope=scope),
            )
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
                # this is fine, just means notching to fnotch.
                if e.args[0].startswith("No objects to concatenate"):
                    logger.debug("No fnotches found for %s.", id_)
                else:
                    # if it's a different error, raise it though:
                    raise ValueError

            # write out the fans and blotches that where not within fnotching distance:
            fans_remaining = fans.loc[set(fans.index) - set(X)]
            if len(fans_remaining) > 0:
                fans_remaining.to_csv(pm.reduced_fanfile, index=False)
            blotches_remaining = blotches.loc[set(blotches.index) - set(Y)]
            if len(blotches_remaining) > 0:
                blotches_remaining.to_csv(pm.reduced_blotchfile, index=False)
        else:
            if blotches is not None:
                blotches.to_csv(pm.reduced_blotchfile, index=False)
            if fans is not None:
                fans.to_csv(pm.reduced_fanfile, index=False)


def fnotch_image_ids_with_shapely(obsid, eps=20, savedir=None, scope="hirise"):
    "Cluster each image_id for an obsid separately."
    # the clustering results were stored as L1A products
    pm = io.PathManager(obsid=obsid, datapath=savedir)
    paths = pm.get_obsid_paths("L1A")
    if len(paths) == 0:
        logger.warning("No paths to fnotch found for %s", obsid)
    for path in paths:
        id_ = get_id_from_path(path)
        pm.id = id_
        # make sure the L1B folder exists
        pm.reduced_fanfile.parent.mkdir(parents=True, exist_ok=True)

        fans, blotches = get_clusters_in_path(path)
        if fans is not None and len(fans) > 1:
            # clean up fans with opposite angles
            fans = remove_opposing_fans(fans)
        if not any([fans is None, blotches is None]):
            logger.debug("Fnotching %s", id_)
            distances = cdist(
                data_to_centers(fans, "fan", scope=scope),
                data_to_centers(blotches, "blotch", scope=scope),
            )
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
                # this is fine, just means notching to fnotch.
                if e.args[0].startswith("No objects to concatenate"):
                    logger.debug("No fnotches found for %s.", id_)
                else:
                    # if it's a different error, raise it though:
                    raise ValueError

            # write out the fans and blotches that where not within fnotching distance:
            fans_remaining = fans.loc[set(fans.index) - set(X)]
            if len(fans_remaining) > 0:
                fans_remaining.to_csv(pm.reduced_fanfile, index=False)
            blotches_remaining = blotches.loc[set(blotches.index) - set(Y)]
            if len(blotches_remaining) > 0:
                blotches_remaining.to_csv(pm.reduced_blotchfile, index=False)
        else:
            if blotches is not None:
                blotches.to_csv(pm.reduced_blotchfile, index=False)
            if fans is not None:
                fans.to_csv(pm.reduced_fanfile, index=False)


def fnotch_obsid(obsid, eps=20, savedir=None):
    pm = io.PathManager(obsid=obsid, datapath=savedir)
    paths = pm.get_obsid_paths("L1A")
    if len(paths) == 0:
        logger.warning("No paths to fnotch found for %s", obsid)
    fans = []
    blotches = []
    for path in paths:
        f, b = get_clusters_in_path(path)
        fans.append(f)
        blotches.append(b)
    try:
        fans = pd.concat(fans, ignore_index=True)
    except ValueError:
        fans = None
    try:
        blotches = pd.concat(blotches, ignore_index=True)
    except ValueError:
        blotches = None
    if fans is not None and len(fans) > 1:
        # clean up fans with opposite angles
        fans = remove_opposing_fans(fans)
    if not any([fans is None, blotches is None]):
        distances = cdist(
            data_to_centers(fans, "fan", scope="hirise"),
            data_to_centers(blotches, "blotch", scope="hirise"),
        )
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
            pm.fnotchfile.parent.mkdir(exist_ok=True)
            pd.concat(fnotches).to_csv(pm.fnotchfile)
        except ValueError as e:
            # this is fine, just means notching to fnotch.
            if e.args[0].startswith("No objects to concatenate"):
                logger.debug("No fnotches found for %s.", obsid)
            else:
                # if it's a different error, raise it though:
                raise ValueError

        # write out the fans and blotches that where not within fnotching distance:
        fans_remaining = fans.loc[set(fans.index) - set(X)]
        if len(fans_remaining) > 0:
            fans_remaining.to_csv(pm.reduced_fanfile, index=False)
        blotches_remaining = blotches.loc[set(blotches.index) - set(Y)]
        if len(blotches_remaining) > 0:
            blotches_remaining.to_csv(pm.reduced_blotchfile, index=False)
    else:
        if blotches is not None:
            pm.reduced_blotchfile.parent.mkdir(parents=True, exist_ok=True)
            blotches.to_csv(pm.reduced_blotchfile, index=False)
        if fans is not None:
            pm.reduced_fanfile.parent.mkdir(parents=True, exist_ok=True)
            fans.to_csv(pm.reduced_fanfile, index=False)


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
    logger.debug("Writing l1c for %s", kind)
    try:
        new_kinds = slashed.loc[[kind]].copy()
    except KeyError:
        logger.debug("No %s in slashed dataframe.", kind)
        new_kinds = pd.DataFrame()
    l1c = getattr(pm, f"final_{kind}file")
    l1c.parent.mkdir(parents=True, exist_ok=True)
    try:
        # the pathmanager can read the csv files as well:
        old_kinds = getattr(pm, f"reduced_{kind}df")
    except FileNotFoundError:
        logger.debug("No old %s file.", kind)
        old_kinds = pd.DataFrame()
    logger.debug("Combining. Writing to %s", str(l1c))
    combined = pd.concat([old_kinds, new_kinds], ignore_index=True, sort=False)
    combined.dropna(how="all", axis=1, inplace=True)
    if len(combined) > 0:
        logger.debug("Writing %s", str(l1c))
        combined.to_csv(str(l1c), index=False)


def apply_cut_obsid(obsid, cut=0.5, savedir=None):
    pm = io.PathManager(obsid=obsid, cut=cut, datapath=savedir)
    try:
        fnotches = pm.fnotchdf
    except FileNotFoundError:
        # no fnotch df was found. Now need to copy over
        # standard files to L1C folder
        pm.final_blotchfile.parent.mkdir(exist_ok=True)
        if pm.reduced_blotchfile.exists():
            logger.debug("Writing final_blotchfile for %s", obsid)
            pm.reduced_blotchdf.to_csv(pm.final_blotchfile, index=False)
        if pm.reduced_fanfile.exists():
            logger.debug("Writing final_fanfile for %s", obsid)
            pm.reduced_fandf.to_csv(pm.final_fanfile, index=False)
    else:
        # apply cut
        slashed = fnotches[fnotches.vote_ratio > pm.cut]
        for kind in ["fan", "blotch"]:
            write_l1c(kind, slashed, pm)


def apply_cut(obsid, cut=0.5, savedir=None):
    """Loop over all image_id paths for an obsid and apply cut to fnotches.

    Parameters
    ----------
    obsid : str
        HiRISE obsid, i.e. P4 `image_name`
    cut : float, 0..1
        Value where to cut the vote_ratio of the fnotches.
    """
    pm = io.PathManager(obsid=obsid, cut=cut, datapath=savedir)
    paths = pm.get_obsid_paths("L1B")
    for path in paths:
        id_ = get_id_from_path(path)
        logger.debug("Slashing %s", id_)
        pm.id = id_
        try:
            fnotches = pm.fnotchdf
        except FileNotFoundError:
            # no fnotch df was found. Now need to copy over
            # standard files to L1C folder
            pm.final_blotchfile.parent.mkdir(exist_ok=True)
            if pm.reduced_blotchfile.exists():
                logger.debug("Writing final_blotchfile for %s", id_)
                pm.reduced_blotchdf.to_csv(pm.final_blotchfile, index=False)
            if pm.reduced_fanfile.exists():
                logger.debug("Writing final_fanfile for %s", id_)
                pm.reduced_fandf.to_csv(pm.final_fanfile, index=False)
        else:
            # apply cut
            slashed = fnotches[fnotches.vote_ratio > pm.cut]
            for kind in ["fan", "blotch"]:
                write_l1c(kind, slashed, pm)
