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
