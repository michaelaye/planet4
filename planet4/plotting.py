import matplotlib.pyplot as plt
from pathlib import Path

from . import io, markings


def plot_image_id_pipeline(image_id, dbname=None, datapath=None, save=False,
                           savetitle='', figtitle='', cm=None, **kwargs):
    """Plotting tool to show the results along the P4 pipeline.

    Parameters
    ----------
    image_id : str
        String in full or short form (without APF000x in front) of the
        P4 image id.
    dbname : {str, pathlib.Path}
        Path to database file to be used for plotting the raw data.
    datapath : {str, pathlib.Path}
        Path to directory where the clustering results are stored.
    save : bool
        Flag controlling if plots should be saved.
    savetitle : str
        Additional filename postfix string for plot filenames.
        The filename will be "{0}_{1}.pdf", with {0} being the `image_id`
        and {1} being `savetitle`
    figtitle : str
        Additional figure title to be displayed.
    cm : clustering.ClusteringManager
        Object that can be used to fish out current path informations
    """
    if cm is not None:
        datapath = cm.pm.datapath
    else:
        datapath = Path(datapath)
    if cm is not None and dbname is None:
        dbname = cm.dbname
    imgid = markings.ImageID(image_id, dbname=dbname, scope='planet4')

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
    axes = axes.ravel()
    for ax in axes:
        imgid.show_subframe(ax=ax)
    imgid.plot_fans(ax=axes[1])
    imgid.plot_blotches(ax=axes[2])

    plot_clustered_fans(image_id, ax=axes[4], datapath=datapath, **kwargs)
    plot_clustered_blotches(image_id, ax=axes[5], datapath=datapath, **kwargs)
    plot_finals(image_id, ax=axes[3], datapath=datapath)

    # for ax in axes:
    #     ax.set_axis_off()

    if figtitle == '' and cm is not None:
        figtitle = "min_samples: {}".format(cm.min_samples)
        figtitle += ", dynamic: {}".format(cm.do_dynamic_min_samples)
        figtitle += ", n_(blotch|fan) markings: {}".format(cm.n_classifications)
        figtitle += ("\nn_clustered_fans: {}, n_clustered_blotches: {}"
                     .format(cm.n_clustered_fans, cm.n_clustered_blotches))
    fig.suptitle(image_id + ', ' + str(figtitle))
    fig.subplots_adjust(left=None, top=None, bottom=None, right=None,
                        wspace=0.01, hspace=0.01)
    if save:
        fname = "{}_{}.pdf".format(imgid.imgid, savetitle)
        saveroot = datapath / 'plots'
        saveroot.mkdir(exist_ok=True)
        fpath = saveroot / fname
        plt.savefig(str(fpath))


def plot_raw_fans(id_, ax=None, dbname=None):
    imgid = markings.ImageID(id_, scope='planet4', dbname=dbname)

    imgid.plot_fans(ax=ax)


def plot_raw_blotches(id_, ax=None):
    imgid = markings.ImageID(id_, scope='planet4')

    imgid.plot_blotches(ax=ax)


def plot_finals(id_, scope='planet4', datapath=None, ax=None):
    pm = io.PathManager(id_=id_, datapath=datapath)
    try:
        if not all([pm.final_blotchfile.exists(),
                    pm.final_fanfile.exists()]):
            print("Some files not found.")
    except TypeError:
        print("Error while trying to plot finals.")
        return

    imgid = markings.ImageID(id_, scope=scope)

    if ax is None:
        _, ax = plt.subplots()
    if pm.final_fanfile.exists():
        imgid.plot_fans(ax=ax, fans=pm.final_fandf)
    if pm.final_blotchfile.exists():
        imgid.plot_blotches(ax=ax, blotches=pm.final_blotchdf)


def plot_clustered_blotches(id_, scope='planet4', datapath=None, ax=None, **kwargs):
    pm = io.PathManager(id_=id_, datapath=datapath)
    if not pm.reduced_blotchfile.exists():
        print("Clustered blotchfile not found")
        return
    # reduced_blotches = markings.BlotchContainer.from_fname(pm.reduced_blotchfile,
    #                                                        scope)
    data = pm.reduced_blotchdf
    imgid = markings.ImageID(id_, scope=scope)

    imgid.plot_blotches(blotches=data, ax=ax, **kwargs)


def blotches_all(id_, datapath=None):
    fig, axes = plt.subplots(ncols=2)
    plot_raw_blotches(id_, ax=axes[0])
    plot_clustered_blotches(id_, datapath, ax=axes[1])
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
        fans = pm.reduced_fandf
        data = fans[fans.image_id == image_id]
    else:
        pm = io.PathManager(id_=image_id, datapath=datapath)
        if not pm.reduced_fanfile.exists():
            print("Clustered/reduced fanfile not found")
            return
        data = pm.reduced_fandf
        # reduced_fans = markings.FanContainer.from_fname(pm.reduced_fanfile,
        #                                                 scope='planet4')
    imgid = markings.ImageID(image_id, scope='planet4')

    imgid.plot_fans(fans=data, ax=ax, **kwargs)


def fans_all(id_, datapath=None):
    fig, axes = plt.subplots(ncols=2)
    plot_raw_fans(id_, ax=axes[0])
    plot_clustered_fans(id_, datapath, ax=axes[1])
    fig.subplots_adjust(left=None, top=None, bottom=None, right=None,
                        wspace=0.001, hspace=0.001)
