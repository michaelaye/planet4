import matplotlib.pyplot as plt

from . import io
from . import markings


def plot_image_id_pipeline(image_id, **kwargs):
    save = kwargs.pop('save', False)
    savetitle = kwargs.pop('savetitle', '')
    pm = io.PathManager(id_=image_id, **kwargs)

    imgid = markings.ImageID(image_id)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
    axes = axes.ravel()
    for ax in axes:
        imgid.show_subframe(ax=ax)
    imgid.plot_fans(ax=axes[1])
    imgid.plot_blotches(ax=axes[2])

    plot_clustered_fans(image_id, ax=axes[4])
    plot_clustered_blotches(image_id, ax=axes[5])
    plot_finals(image_id, ax=axes[3])

    for ax in axes:
        ax.set_axis_off()

    fig.subplots_adjust(left=None, top=None, bottom=None, right=None,
                        wspace=0.001, hspace=0.001)
    if save:
        fname = "{}_{}.pdf".format(imgid.imgid, savetitle)
        fpath = pm.fnotched_dir / fname
        plt.savefig(str(fpath))


def plot_raw_fans(id_, ax=None):
    imgid = markings.ImageID(id_, scope='planet4')

    imgid.plot_fans(ax=ax)


def plot_raw_blotches(id_, ax=None):
    imgid = markings.ImageID(id_, scope='planet4')

    imgid.plot_blotches(ax=ax)


def plot_finals(id_, scope='planet4', _dir=None, ax=None):
    pm = io.PathManager(id_=id_, datapath=_dir)
    if not all([pm.final_blotchfile.exists(),
                pm.final_fanfile.exists()]):
        print("Some files not found.")
        return

    finalfans = markings.FanContainer.from_fname(pm.final_fanfile, scope)
    finalblotches = markings.BlotchContainer.from_fname(pm.final_blotchfile,
                                                        scope)

    imgid = markings.ImageID(id_)

    if ax is None:
        _, ax = plt.subplots()
    imgid.plot_fans(ax=ax, fans=finalfans.content)
    imgid.plot_blotches(ax=ax, blotches=finalblotches.content)


def plot_clustered_blotches(id_, scope='planet4', _dir=None, ax=None, **kwargs):
    pm = io.PathManager(id_=id_, datapath=_dir)
    if not pm.reduced_blotchfile.exists():
        print("Clustered blotchfile not found")
        return
    reduced_blotches = markings.BlotchContainer.from_fname(pm.reduced_blotchfile,
                                                           scope)
    imgid = markings.ImageID(id_)

    imgid.plot_blotches(blotches=reduced_blotches.content, ax=ax, **kwargs)


def blotches_all(id_, _dir=None):
    fig, axes = plt.subplots(ncols=2)
    plot_raw_blotches(id_, ax=axes[0])
    plot_clustered_blotches(id_, _dir, ax=axes[1])
    fig.subplots_adjust(left=None, top=None, bottom=None, right=None,
                        wspace=0.001, hspace=0.001)


def plot_clustered_fans(id_, scope='planet4', _dir=None, ax=None, **kwargs):
    pm = io.PathManager(id_=id_, datapath=_dir)
    if not pm.reduced_fanfile.exists():
        print("Clustered/reduced fanfile not found")
        return
    reduced_fans = markings.FanContainer.from_fname(pm.reduced_fanfile,
                                                    scope)
    imgid = markings.ImageID(id_)

    imgid.plot_fans(fans=reduced_fans.content, ax=ax, **kwargs)


def fans_all(id_, _dir=None):
    fig, axes = plt.subplots(ncols=2)
    plot_raw_fans(id_, ax=axes[0])
    plot_clustered_fans(id_, _dir, ax=axes[1])
    fig.subplots_adjust(left=None, top=None, bottom=None, right=None,
                        wspace=0.001, hspace=0.001)
