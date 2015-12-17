import matplotlib.pyplot as plt

from .io import PathManager
from .markings import ImageID


def plot_image_id_pipeline(image_id, data):
    imgid = ImageID(image_id, data)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
    axes = axes.ravel()
    for ax in axes:
        imgid.show_subframe(ax=ax)
    imgid.plot_fans(ax=axes[1])
    imgid.plot_blotches(ax=axes[2])

    for ax in axes:
        ax.set_axis_off()
    plt.show()
