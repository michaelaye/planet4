import matplotlib.pyplot as plt

from . import io
from . import markings
from . import clustering


def plot_image_id_pipeline(image_id, min_distance=10):
    cm = clustering.ClusteringManager(id_=image_id, min_distance=min_distance)
    cm.cluster_image_id(image_id)
    pm = cm.pm

    imgid = markings.ImageID(image_id)

    reduced_fans = markings.FanContainer.from_fname(pm.reduced_fanfile)
    reduced_blotches = markings.BlotchContainer.from_fname(pm.reduced_blotchfile)

    finalfans = markings.FanContainer.from_fname(pm.final_fanfile)
    finalblotches = markings.BlotchContainer.from_fname(pm.final_blotchfile)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
    axes = axes.ravel()
    for ax in axes:
        imgid.show_subframe(ax=ax)
    imgid.plot_fans(ax=axes[1])
    imgid.plot_blotches(ax=axes[2])
    imgid.plot_fans(ax=axes[4], fans=reduced_fans.content)
    imgid.plot_blotches(ax=axes[5], blotches=reduced_blotches.content)
    imgid.plot_fans(ax=axes[3], fans=finalfans.content)
    imgid.plot_blotches(ax=axes[3], blotches=finalblotches.content)

    for ax in axes:
        ax.set_axis_off()

    fig.subplots_adjust(left=None, top=None, bottom=None, right=None,
                        wspace=0.001, hspace=0.001)
    plt.show()
    if pm:
        return pm
    else:
        return cm


def plot_raw_fans(id_):
    imgid = markings.ImageID(id_)

    imgid.plot_fans()
    plt.show()


def plot_raw_blotches(id_):
    imgid = markings.ImageID(id_)

    imgid.plot_blotches()
    plt.show()


def plot_finals(id_, dir=None):
    pm = io.PathManager(id_=id_, datapath=dir)
    if not all([pm.reduced_blotchfile.exists(),
                pm.reduced_fanfile.exists()]):
        cm = clustering.ClusteringManager(id_=id_, fnotched_dir=dir)
        cm.cluster_image_id(id_=id_)
        pm = cm.pm

    finalfans = markings.FanContainer.from_fname(pm.final_fanfile)
    finalblotches = markings.BlotchContainer.from_fname(pm.final_blotchfile)

    imgid = markings.ImageID(id_)

    fig, ax = plt.subplots()
    imgid.plot_fans(ax=ax, fans=finalfans.content)
    imgid.plot_blotches(ax=ax, blotches=finalblotches.content)
    plt.show()


def plot_clustered_blotches(id_, dir=None):
    pm = io.PathManager(id_=id_, datapath=dir)
    if not pm.reduced_blotchfile.exists():
        cm = clustering.ClusteringManager(id_=id_, fnotched_dir=dir)
        cm.cluster_image_id(id_)
        pm = cm.pm

    reduced_blotches = markings.BlotchContainer.from_fname(pm.reduced_blotchfile)
    imgid = markings.ImageID(id_)

    imgid.plot_blotches(blotches=reduced_blotches.content)
    plt.show()


def plot_clustered_fans(id_, dir=None):
    pm = io.PathManager(id_=id_, datapath=dir)
    if not pm.reduced_fanfile.exists():
        cm = clustering.ClusteringManager(id_=id_, fnotched_dir=dir)
        cm.cluster_image_id(id_)
        pm = cm.pm

    reduced_fans = markings.FanContainer.from_fname(pm.reduced_fanfile)
    imgid = markings.ImageID(id_)

    imgid.plot_fans(fans=reduced_fans.content)
    plt.show()
