import logging
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from . import io, markings

logger = logging.getLogger(__name__)


def get_clustering_settings_log(img_id):
    pm = io.PathManager(img_id)
    fpath = pm.blotchfile.parent / 'clustering_settings.yaml'
    with open(fpath) as fp:
        return yaml.load(fp)


def plot_image_id_pipeline(image_id, dbname=None, datapath=None, save=False,
                           savetitle='', figtitle='', **kwargs):
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
    """
    pm = io.PathManager(image_id)
    if datapath is not None:
        datapath = Path(datapath)
    else:
        datapath = pm.path_so_far.parent
    imgid = markings.ImageID(image_id, dbname=dbname, scope='planet4')

    clustering_settings = get_clustering_settings_log(image_id)
    min_samples = clustering_settings['min_samples']

    n_classifications = imgid.n_marked_classifications
    fig, axes = plt.subplots(nrows=2, ncols=3)
    axes = axes.ravel()
    for ax in axes:
        imgid.show_subframe(ax=ax)
    imgid.plot_fans(ax=axes[1])
    imgid.plot_blotches(ax=axes[2])

    n_clust_fans = plot_clustered_fans(image_id,
                                       ax=axes[4],
                                       datapath=datapath,
                                       **kwargs)
    n_clust_blotches = plot_clustered_blotches(image_id,
                                               ax=axes[5],
                                               datapath=datapath,
                                               **kwargs)
    plot_finals(image_id, ax=axes[3], datapath=datapath)

    # for ax in axes:
    #     ax.set_axis_off()

    if figtitle == '':
        figtitle = "min_samples: {}".format(min_samples)
        figtitle += ", n_(blotch|fan) markings: {}".format(n_classifications)
        figtitle += ("\nn_clustered_fans: {}, n_clustered_blotches: {}"
                     .format(n_clust_fans, n_clust_blotches))
    fig.suptitle(image_id + ', ' + str(figtitle))
    fig.subplots_adjust(left=None, top=None, bottom=None, right=None,
                        wspace=0.01, hspace=0.01)
    if save:
        fname = "{}_{}.pdf".format(imgid.imgid, savetitle)
        saveroot = pm.final_fanfile.parent / 'plots'
        saveroot.mkdir(exist_ok=True)
        fpath = saveroot / fname
        plt.savefig(str(fpath))


def plot_raw_fans(id_, ax=None, dbname=None):
    imgid = markings.ImageID(id_, scope='planet4', dbname=dbname)

    imgid.plot_fans(ax=ax)


def plot_raw_blotches(id_, ax=None):
    imgid = markings.ImageID(id_, scope='planet4')

    imgid.plot_blotches(ax=ax)


def plot_finals(id_, datapath=None, ax=None, scope='planet4'):

    pm = io.PathManager(id_=id_, datapath=datapath)
    if not all([pm.final_blotchfile.exists(),
                pm.final_fanfile.exists()]):
        logger.warning("Some files not found.")

    imgid = markings.ImageID(id_, scope=scope)

    if ax is None:
        _, ax = plt.subplots()
    if pm.final_fanfile.exists():
        imgid.plot_fans(ax=ax, data=pm.final_fandf, with_center=True)
    if pm.final_blotchfile.exists():
        imgid.plot_blotches(ax=ax, data=pm.final_blotchdf, with_center=True)


def plot_clustered_blotches(id_, scope='planet4', datapath=None, ax=None,
                            **kwargs):
    pm = io.PathManager(id_=id_, datapath=datapath)
    if not pm.blotchfile.exists():
        print("Clustered blotchfile not found")
        return
    # blotches = markings.BlotchContainer.from_fname(pm.blotchfile,
    #                                                        scope)
    data = pm.blotchdf
    imgid = markings.ImageID(id_, scope=scope)

    imgid.plot_blotches(data=data, ax=ax, **kwargs)
    return len(data)


def blotches_all(id_, datapath=None):
    fig, axes = plt.subplots(ncols=2)
    plot_raw_blotches(id_, ax=axes[0])
    plot_clustered_blotches(id_, datapath=datapath, ax=axes[1])
    fig.subplots_adjust(left=None, top=None, bottom=None, right=None,
                        wspace=0.001, hspace=0.001)


def plot_clustered_fans(image_id, datapath=None, ax=None, scope_id=None,
                        **kwargs):
    """Plot the reduced/clustered fans.

    Plot the clustered fans for a planet4 image_id.
    This will plot the extra stored only clustered fans before they are being
    fnotched together.

    Parameters
    ----------
    id_ : str
        PlanetFour image id
    datapath : pathlib.Path or str, optional
        Path to folder where the analysis run is stored. If None, defaults to
        data_root / 'clustering'
    ax : matplotlib.axis, optional
        Axis to plot on
    scope_id : str
        obsid (= image_name) to search in for image_id data.
    """
    if scope_id is not None:
        pm = io.PathManager(id_=scope_id, datapath=datapath)
        fans = pm.fandf
        data = fans[fans.image_id == image_id]
    else:
        pm = io.PathManager(id_=image_id, datapath=datapath)
        if not pm.fanfile.exists():
            print("Clustered/reduced fanfile not found")
            return
        data = pm.fandf
        # fans = markings.FanContainer.from_fname(pm.fanfile,
        #                                                 scope='planet4')
    imgid = markings.ImageID(image_id, scope='planet4')

    imgid.plot_fans(data=data, ax=ax, **kwargs)
    return len(data)


def fans_all(id_, datapath=None):
    fig, axes = plt.subplots(ncols=2)
    plot_raw_fans(id_, ax=axes[0])
    plot_clustered_fans(id_, datapath=datapath, ax=axes[1])
    fig.subplots_adjust(left=None, top=None, bottom=None, right=None,
                        wspace=0.001, hspace=0.001)
