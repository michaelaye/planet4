import itertools
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from ipywidgets import interact

from . import io, markings, projection

logger = logging.getLogger(__name__)


def get_clustering_log(pm):
    fpath = pm.blotchfile.parent / "clustering_settings.yaml"
    with open(fpath) as fp:
        return yaml.safe_load(fp)


def plot_image_id_pipeline(
    image_id,
    dbname=None,
    datapath=None,
    save=False,
    savetitle="",
    saveroot=None,
    do_title=True,
    via_obsid=False,
    figsize=None,
    **kwargs,
):
    """Plotting tool to show the results along the P4 pipeline.

    Parameters
    ----------
    image_id : str
        String in full or short form (without APF000x in front) of the
        P4 image id.
    dbname : {str, pathlib.Path}, optional
        Path to database file to be used for plotting the raw data.
    datapath : {str, pathlib.Path}, optional
        Path to directory where the clustering results are stored.
    save : bool, optional
        Flag controlling if plots should be saved.
    savetitle : str, optional
        Additional filename postfix string for plot filenames.
        The filename will be "{0}_{1}.pdf", with {0} being the `image_id`
        and {1} being `savetitle`
    figtitle : str, optional
        Additional figure title to be displayed.
    via_obsid : bool
        Switch to look for different path when fnotched on hirise images
    """
    pm = io.PathManager(image_id, datapath=datapath)
    if datapath is not None:
        datapath = Path(datapath)
    else:
        datapath = pm.path_so_far.parent
    imgid = markings.TileID(image_id, dbname=dbname, scope="planet4")

    clustering_settings = get_clustering_log(pm)
    min_samples = clustering_settings["min_samples"]
    # min_samples = 'no_log'

    n_classifications = imgid.n_marked_classifications
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize, constrained_layout=True)
    axes = axes.ravel()
    for ax in axes:
        imgid.show_subframe(ax=ax)
    imgid.plot_fans(ax=axes[1])
    imgid.plot_blotches(ax=axes[2])

    n_clust_fans = plot_clustered_markings(
        image_id, "fan", ax=axes[4], datapath=datapath, **kwargs
    )
    n_clust_blotches = plot_clustered_markings(
        image_id, "blotch", ax=axes[5], datapath=datapath, **kwargs
    )
    plot_finals(image_id, ax=axes[3], datapath=datapath, via_obsid=via_obsid)

    # for ax in axes:
    #     ax.set_axis_off()

    if do_title is True:
        figtitle = "min_samples: {}".format(min_samples)
        figtitle += ", n_(blotch|fan) classif.: {}".format(n_classifications)
        figtitle += "\nn_clustered_fans: {}, n_clustered_blotches: {}".format(
            n_clust_fans, n_clust_blotches
        )
        fig.suptitle(image_id + ", " + str(figtitle))

    # fig.subplots_adjust(left=None, top=None, bottom=None, right=None,
    # wspace=0.01, hspace=0.01)
    if save:
        fname = "P4_pipeline_{}_{}.png".format(imgid.imgid, savetitle)
        if saveroot is None:
            saveroot = pm.final_fanfile.parent / "plots"
        else:
            saveroot = Path(saveroot)
        saveroot.mkdir(exist_ok=True)
        fpath = saveroot / fname
        logging.debug("Saving plot at %s", str(fpath))
        plt.savefig(str(fpath), dpi=200, transparent=False, facecolor='white')
        plt.savefig(fpath.with_suffix(".pdf"), facecolor='white')


def plot_raw_fans(id_, ax=None, dbname=None):
    imgid = markings.TileID(id_, scope="planet4", dbname=dbname)

    imgid.plot_fans(ax=ax)


def plot_raw_blotches(id_, ax=None):
    imgid = markings.TileID(id_, scope="planet4")

    imgid.plot_blotches(ax=ax)


def plot_finals_with_input(id_, datapath=None, horizontal=True, scope="planet4"):
    imgid = markings.TileID(id_, scope=scope)
    pm = io.PathManager(id_=id_, datapath=datapath)
    if horizontal is True:
        kwargs = {"ncols": 2}
    else:
        kwargs = {"nrows": 2}
    fig, ax = plt.subplots(figsize=(4, 5), **kwargs)
    ax[0].set_title(imgid.imgid, fontsize=6)
    imgid.show_subframe(ax=ax[0])
    for marking in ["fan", "blotch"]:
        try:
            df = getattr(pm, f"final_{marking}df")
        except:
            continue
        else:
            data = df[df.image_id == id_]
            imgid.plot_markings(marking, data, ax=ax[1])


def plot_finals(
    id_, datapath=None, ax=None, scope="planet4", via_obsid=False, **kwargs
):
    id_ = io.check_and_pad_id(id_)
    imgid = markings.TileID(id_, scope=scope)
    if not via_obsid:
        pm = io.PathManager(id_=id_, datapath=datapath)
    else:
        pm = io.PathManager(obsid=imgid.image_name, datapath=datapath)
        logger.debug("via_obsid active. Fan path: %s", str(pm.final_fanfile))
        logger.debug("via_obsid active. Blotch path: %s", str(pm.final_blotchfile))

    if ax is None:
        _, ax = plt.subplots()
    if pm.final_fanfile.exists():
        df = pm.final_fandf
        data = df[df.image_id == id_]
        logger.debug("Len of data: %i", len(df))
        imgid.plot_fans(ax=ax, data=data, with_center=False, **kwargs)
    else:
        logger.info("File not found: %s", str(pm.final_fanfile))
    if pm.final_blotchfile.exists():
        df = pm.final_blotchdf
        data = df[df.image_id == id_]
        imgid.plot_blotches(ax=ax, data=data, with_center=False, **kwargs)
    else:
        logger.info("File not found: %s", str(pm.final_blotchfile))


def blotches_all(id_, datapath=None):
    fig, axes = plt.subplots(ncols=2)
    plot_raw_blotches(id_, ax=axes[0])
    plot_clustered_markings(id_, kind="blotch", datapath=datapath, ax=axes[1])
    fig.subplots_adjust(
        left=None, top=None, bottom=None, right=None, wspace=0.001, hspace=0.001
    )


def plot_clustered_markings(
    image_id, kind, datapath=None, ax=None, obsid=None, **kwargs
):
    """Plot the reduced/clustered objects.

    Plot the clustered fans for a planet4 image_id.
    This will plot the extra stored only clustered fans before they are being
    fnotched together.

    Parameters
    ----------
    id_ : str
        PlanetFour image id
    kind : {"fan", "blotch"}
        Which markings to plot
    datapath : pathlib.Path or str, optional
        Path to folder where the analysis run is stored. If None, defaults to
        data_root / 'clustering'
    ax : matplotlib.axis, optional
        Axis to plot on
    obsid : str
        obsid (= image_name) to search in for image_id data.
    """
    imgid = markings.TileID(image_id, scope="planet4")
    if obsid is not None:
        pm = io.PathManager(obsid=obsid, datapath=datapath)
        objects = getattr(pm, f"{kind}df")
        data = objects[objects.image_id == imgid.imgid]
    else:
        pm = io.PathManager(id_=image_id, datapath=datapath)
        objectfile = getattr(pm, f"{kind}file")
        if not objectfile.exists():
            logger.info("Clustered/reduced %sfile not found", kind)
            return
        data = getattr(pm, f"{kind}df")

    imgid.plot_markings(kind=kind, data=data, ax=ax, **kwargs)
    return len(data)


def fans_all(id_, datapath=None):
    fig, axes = plt.subplots(ncols=2)
    plot_raw_fans(id_, ax=axes[0])
    plot_clustered_markings(id_, "fan", datapath=datapath, ax=axes[1])
    fig.subplots_adjust(
        left=None, top=None, bottom=None, right=None, wspace=0.001, hspace=0.001
    )


def get_tile_image(df, xtile, ytile):
    filtered = df.query("x_tile=={} and y_tile=={}".format(xtile, ytile))
    return io.get_subframe(filtered.image_url.iloc[0])


def get_four_tiles_df(df, x0, y0):
    query = "x_tile > {} and x_tile < {} and y_tile > {} and y_tile < {}".format(
        x0 - 1, x0 + 2, y0 - 1, y0 + 2
    )
    return df.query(query).sort_values(by=["x_tile", "y_tile"])


def get_four_tiles_img(obsid, x0, y0):
    df = io.DBManager().get_image_name_markings(obsid)
    tiles = []
    # loop along columns (= to the right)
    for xtile in [x0, x0 + 1]:
        # loop along rows (= down)
        for ytile in [y0, y0 + 1]:
            tiles.append(get_tile_image(df, xtile, ytile))

    # tiles[0] and tiles[1] are the left most tiles
    # we have overlap of 100 pixels in all directions
    left = np.vstack([tiles[0], tiles[1][100:]])
    right = np.vstack([tiles[2], tiles[3][100:]])
    # now slicing on axis=1, because I combine in column-direction
    all_ = np.hstack([left, right[:, 100:]])
    return all_


def browse_images(obsid):
    df = io.DBManager().get_image_name_markings(obsid)
    xmax = df.x_tile.max()
    ymax = df.y_tile.max()

    def view_image(xtile=1, ytile=1):
        img = get_four_tiles_img(obsid, xtile, ytile)
        print(img.shape)
        plt.imshow(img, origin="upper", aspect="auto")
        plt.title(f"x_tile: {xtile}, y_tile: {ytile}")
        plt.show()

    interact(view_image, xtile=(1, xmax - 1), ytile=(1, ymax - 1))


def get_finals_from_obsid(obsid, datapath, kind="blotch", ids=None):
    # get final data for case of no given image_id, meaning it's for
    # fnotched on hirise level
    pm = io.PathManager(obsid=obsid, datapath=datapath)
    df = getattr(pm, f"final_{kind}df")
    if ids is not None:
        return pd.concat([df[df.image_id == i] for i in ids], ignore_index=True)
    else:
        return df


def plot_four_tiles(obsid, x0, y0, ax=None):
    img_x_size = 840
    img_y_size = 648

    all_ = get_four_tiles_img(obsid, x0, y0)
    UL = projection.xy_to_hirise(0, 0, x0, y0)
    LR = projection.xy_to_hirise(img_x_size, img_y_size, x0 + 1, y0 + 1)
    extent = [UL[0], LR[0], LR[1], UL[1]]
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(all_, origin="upper", extent=extent, aspect="auto")
    return UL, LR, ax


def plot_four_tiles_finals(obsid, datapath, x0, y0):
    colors = itertools.cycle(sns.color_palette("bright", 12))

    obsid_data = io.DBManager().get_image_name_markings(obsid)
    p4data = get_four_tiles_df(obsid_data, x0, y0)
    image_ids = p4data.image_id.unique()
    # plotting
    UL, LR, ax = plot_four_tiles(obsid, x0, y0)
    for id_ in image_ids:
        plot_finals(id_, datapath, ax, scope="hirise", via_obsid=False)

    ax.set_xlim(UL[0], LR[0])
    ax.set_ylim(LR[1], UL[1])
