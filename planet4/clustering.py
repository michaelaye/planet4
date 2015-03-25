from __future__ import print_function, division
from planet4 import markings
from planet4 import get_data
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('bmh')

ellipse_cols = 'x y radius_1 radius_2 angle'.split()
fan_cols = 'x y angle spread distance'.split()


def get_mean_marking(data, label_members, fan=False):
    if not fan:
        ellipsedata = data[ellipse_cols].iloc[label_members]
    else:
        ellipsedata = data[fan_cols].iloc[label_members]
    meandata = ellipsedata.mean()
    if not fan:
        return markings.Blotch(meandata)
    else:
        return markings.Fan(meandata)


def perform_dbscan(current_data, current_axis=None, eps=10, min_samples=3,
                   fans=False, linestyle='-', quiet=True):
    """took out axes as 2nd parameter"""
    center_only = ['x', 'y']
    # center_and_radii = 'x y radius_1 radius_2'.split()
    current_X = current_data[center_only].values

    # there could be no markings! Returning None then.
    if len(current_X) == 0:
        return None
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(current_X)
    labels = db.labels_.astype('int')
    core_samples = db.core_sample_indices_
    # number of clusters in labels, ignoring noise if present
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if not quiet:
        print("Estimated number of clusters:", n_clusters)

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    reduced_data = []
    for k, color in zip(unique_labels, colors):
        label_members = [index[0] for index in np.argwhere(labels == k)]

        # only do the following if we have a matplotlib axis
        if current_axis:
            if k == -1:
                color = 'w'
                markersize = 5
            for index in label_members:
                x = current_X[index]
                if index in core_samples and k != -1:
                    markersize = 8
                else:
                    markersize = 5
                current_axis.plot(x[0], x[1], 'o', markerfacecolor=color,
                                  markeredgecolor='k', markersize=markersize)
        # do this part always to get the mean ellipse data
        if k > -0.5:
            el = get_mean_marking(current_data, label_members,
                                  fan=fans)
            el.n_members = len(label_members)
            reduced_data.append(el)
            if current_axis:
                el.set_color(color)
                if not fans:
                    current_axis.add_artist(el)
                else:
                    current_axis.add_line(el)
                    el.add_semicircle(current_axis, color=color)
                    el.add_mean_wind_pointer(current_axis,
                                             color=color,
                                             ls=linestyle)
        if current_axis:
            markings.set_subframe_size(current_axis)
    return reduced_data


def gold_star_plotter(gold_id, axis, blotches=True, fans=False):
    for goldstar, color in zip(markings.gold_members,
                               markings.gold_plot_colors):
        if blotches:
            gold_id.plot_blotches(user_name=goldstar, ax=axis,
                                  user_color=color)
        if fans:
            gold_id.plot_fans(user_name=goldstar, ax=axis, user_color=color)
        markings.gold_legend(axis)


def main():
    gold_ids = get_data.common_gold_ids()

    p4img = markings.ImageID(gold_ids[10])
    golddata = p4img.data[p4img.data.user_name.isin(markings.gold_members)]
    golddata = golddata[golddata.marking == 'fan']
    # citizens = set(p4img.data.user_name) - set(markings.gold_members)

    # create plot window
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))
    fig.tight_layout()
    axes = ax.flatten()

    # fill images, 0 and 2 get it automatically
    for i in [1, 3]:
        p4img.show_subframe(ax=axes[i])

    # remove pixel coord axes
    for ax in axes:
        ax.axis('off')

    # citizen stuff
    p4img.plot_fans(ax=axes[0])
    axes[0].set_title('Citizen Markings')
    perform_dbscan(p4img.get_fans(), axes[1],
                   eps=7,
                   min_samples=5, fans=True, ls='-')
    axes[1].set_title('All citizens clusters (including science team)')

    # gold stuff
    gold_star_plotter(p4img, axes[2], fans=True, blotches=False)
    axes[2].set_title('Science team markings')
    perform_dbscan(golddata, axes[1],
                   min_samples=2,
                   eps=11, fans=True, ls='--')
    axes[3].set_title('Science team clusters')

    plt.show()


if __name__ == '__main__':
    main()
