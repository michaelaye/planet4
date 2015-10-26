from __future__ import division, print_function

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import FloatText
from pathlib import Path
from sklearn.cluster import DBSCAN

from . import markings, p4io
from .exceptions import NoDataToClusterError, UnknownClusteringScopeError

matplotlib.style.use('bmh')


class DBScanner(object):

    marking_cols = {'fan': 'angle spread distance'.split(),
                    'blotch': 'angle radius_1 radius_2'.split()}
    MarkingClass = {'fan': markings.Fan,
                    'blotch': markings.Blotch}

    def __init__(self, data, kind, eps=10, min_samples=3,
                 scope='planet4',
                 ax=None, linestyle='-', quiet=True):
        self.data = data
        self.kind = kind  # fans or blotches
        self.eps = eps
        self.min_samples = min_samples
        if scope == 'planet4':
            self.coords = ['x', 'y']
        elif scope == 'hirise':
            self.coords = ['image_x', 'image_y']
        else:
            raise UnknownClusteringScopeError
        self.scope = scope
        self.ax = ax
        self.linestyle = linestyle
        self.quiet = quiet

        # these lines execute the clustering
        self.get_current_X()
        self.run_DBSCAN()
        self.post_analysis()

    def get_current_X(self):
        current_X = self.data[self.coords].values
        if len(current_X) == 0:
            raise NoDataToClusterError
        self.current_X = current_X

    def run_DBSCAN(self):
        db = DBSCAN(self.eps, self.min_samples).fit(self.current_X)
        labels = db.labels_.astype('int')
        self.core_samples = db.core_sample_indices_
        unique_labels = set(labels)
        self.n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        self.labels = labels
        self.unique_labels = unique_labels
        if not self.quiet:
            print("Estimated number of clusters:", self.n_clusters)

    def post_analysis(self):
        colors = plt.cm.Spectral(np.linspace(0, 1, len(self.unique_labels)))
        reduced_data = []
        n_rejected = 0
        for k, color in zip(self.unique_labels, colors):
            label_members = [i[0] for i in np.argwhere(self.labels == k)]
            if k == -1:  # i.e. if it's noise
                n_rejected = len(label_members)
            if self.ax:
                self.process_plotting(k, label_members)
            if k > -0.5:  # i.e. if it's not noise marking
                cluster = self.get_mean_marking(label_members)
                cluster.n_members = len(label_members)
                reduced_data.append(cluster)
                if self.ax:
                    self.process_cluster_plotting(cluster, color)
            if self.ax:
                markings.set_subframe_size(self.ax)
        self.reduced_data = reduced_data
        self.n_rejected = n_rejected

    @property
    def n_reduced_data(self):
        return len(self.reduced_data)

    def get_mean_marking(self, label_members):
        # what columns to pick for averaging. Depends on
        # what kind (fans/blotches) we have:
        cols = self.coords + self.marking_cols[self.kind]
        clusterdata = self.data[cols].iloc[label_members]
        meandata = clusterdata.mean()
        if self.scope == 'hirise':
            meandata['x'] = meandata.image_x
            meandata['y'] = meandata.image_y
        return self.MarkingClass[self.kind](meandata)

    def process_cluster_plotting(self, cluster, color):
        cluster.set_color(color)
        if self.kind == 'blotch':
            self.ax.add_artist(cluster)
        else:
            self.ax.add_line(cluster)
            cluster.add_semicircle(self.ax, color=color)
            cluster.add_mean_wind_pointer(self.ax, color=color,
                                          ls=self.linestyle)

    def process_plotting(self, k, label_members):
        if k == -1:  # process noise markers
            color = 'w'
            markersize = 5
        for i in label_members:
            x = self.current_X[i]
            if i in self.core_samples and k != -1:
                markersize = 8
            else:
                markersize = 5
            self.ax.plot(x[0], x[1], 'o', markerfacecolor=color,
                         markedgecolor='k', markersize=markersize)


class ClusteringManager(object):

    "WRITE the effing docstring!"
    def __init__(self, dbname, scope='hirise', min_distance=10, output_dir=None):
        self.db = p4io.DBManager(dbname)
        self.dbname = dbname
        self.scope = scope
        self.min_distance = min_distance
        self.confusion = []
        self.dbscanners = []
        self.final_fans = []
        self.final_blotches = []
        self.final_fnotches = []
        if output_dir is None:
            output_dir = Path(p4io.data_root) / 'output'
        self.output_dir = Path(output_dir)
        try:
            self.output_dir.mkdir()
        except FileExistsError:
            pass

    @property
    def n_clustered_fans(self):
        return len(self.clustered_fans)

    @property
    def n_clustered_blotches(self):
        return len(self.clustered_blotches)

    def cluster_data(self, data):
        "Basic clustering."
        clustered_blotches = []
        clustered_fans = []
        for kind in ['fan', 'blotch']:
            markings = data[data.marking == kind]
            dbscanner = DBScanner(markings, kind, scope=self.scope)
            self.confusion.append((self.data_id, kind, len(markings),
                                   dbscanner.n_reduced_data,
                                   dbscanner.n_rejected))
            if kind == 'fan':
                clustered_fans.extend(dbscanner.reduced_data)
            else:
                clustered_blotches.extend(dbscanner.reduced_data)
        self.clustered_fans = clustered_fans
        self.clustered_blotches = clustered_blotches

    def do_the_fnotch(self):
        "Combine fans and blotches if necessary."
        from numpy.linalg import norm
        n_close = 0
        fnotches = []
        blotches = []
        fans = []
        for blotch in self.clustered_blotches:
            blotch_applied = False
            for fan in self.clustered_fans:
                delta = blotch.center - fan.midpoint
                if norm(delta) < self.min_distance:
                    fnotch_value = calc_fnotch(fan.n_members, blotch.n_members)
                    fnotches.append(markings.Fnotch(fnotch_value,
                                                    fan.data,
                                                    blotch.data))
                    n_close += 1
                    blotch_applied = True
                else:
                    fans.append(fan)
            if not blotch_applied:
                blotches.append(blotch)
        self.final_fnotches = fnotches
        self.final_blotches = blotches
        self.final_fans = fans

    def cluster_image_id(self, image_id):
        self.data_id = image_id
        self.p4id = markings.ImageID(image_id, self.dbname)
        self.cluster_data(self.p4id.data)
        self.do_the_fnotch()

    def cluster_image_name(self, image_name):
        data = self.db.get_image_name_markings(image_name)
        self.data_id = image_name
        self.cluster_data(data)
        self.do_the_fnotch()
        self.store_output(image_name)

    def store_output(self, image_name):
        outfnotch = image_name + '_fnotches.hdf'
        outblotch = image_name + '_blotches.hdf'
        outfan = image_name + '_fans.hdf'
        for outfname, outdata in zip([outfnotch, outblotch, outfan],
                                    [self.final_fnotches, self.final_blotches,
                                     self.final_fans]):
            outpath = self.output_dir / outfname
            series = [cluster.data for cluster in outdata]
            df = pd.DataFrame(series)
            df.to_csv(outpath.with_suffix('.csv').as_posix())

    def cluster_all(self):
        image_names = self.db.image_names
        ft = FloatText()
        display(ft)
        for i, image_name in enumerate(image_names):
            perc = 100 * i / len(image_names)
            # print('{:.1f}'.format())
            ft.value = round(perc, 1)
            self.cluster_image_name(image_name)


    def report(self):
        print("Fnotches:", len(self.final_fnotches))
        print("Fans:", len(self.final_fans))
        print("Blotches:", len(self.final_blotches))

    @property
    def confusion_data(self):
        return pd.DataFrame(self.confusion, columns=['image_name', 'kind',
                                                     'n_markings',
                                                     'n_cluster_members',
                                                     'n_rejected'])

    def save_confusion_data(self, fname):
        self.confusion_data.to_csv(fname)


def get_mean_position(fan, blotch, scope):
    if scope == 'hirise':
        columns = ['hirise_x', 'hirise_y']
    else:
        columns = ['x', 'y']

    df = pd.DataFrame([fan.data[columns], blotch.data[columns]])
    return df.mean()


def calc_fnotch(nfans, nblotches):
    return (nfans)/(nfans+nblotches)


def gold_star_plotter(gold_id, axis, blotches=True, kind='blotches'):
    for goldstar, color in zip(markings.gold_members,
                               markings.gold_plot_colors):
        if blotches:
            gold_id.plot_blotches(user_name=goldstar, ax=axis,
                                  user_color=color)
        if kind == 'fans':
            gold_id.plot_fans(user_name=goldstar, ax=axis, user_color=color)
        markings.gold_legend(axis)


def main():
    gold_ids = p4io.common_gold_ids()

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
    DBScanner(p4img.get_fans(), 'fan', ax=axes[1], eps=7, min_samples=5,
              linestyle='-')
    axes[1].set_title('All citizens clusters (including science team)')

    # gold stuff
    gold_star_plotter(p4img, axes[2], fans=True, blotches=False)
    axes[2].set_title('Science team markings')
    DBScanner(golddata, 'fan', ax=axes[1], min_samples=2, eps=11,
              linestyle='--')
    axes[3].set_title('Science team clusters')

    plt.show()


if __name__ == '__main__':
    main()
