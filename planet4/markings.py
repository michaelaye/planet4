#!/usr/bin/env python
import argparse
import math
from itertools import cycle
from math import cos, degrees, radians, sin, pi

import matplotlib.lines as lines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from numpy import linalg as LA
from numpy import arctan2

from . import io

data_root = io.data_root

img_x_size = 840
img_y_size = 648


img_shape = (img_y_size, img_x_size)

colors = cycle('rgbcym')

gold_members = ['michaelaye', 'mschwamb', 'Portyankina', 'CJ-DPI']
gold_plot_colors = list('cmyg')


def example_p4id():
    db = io.DBManager()
    return ImageID('APF00002rq', data=db.get_image_id_markings('APF00002rq'))


def calc_fig_size(width):
    """Calc figure height in ratio of subframes."""
    ratio = img_x_size / img_y_size
    return (width, width / ratio)


def gold_legend(ax):
    lcolors = list('cmyg')
    line1 = plt.Line2D(range(10), range(10), marker='o', color=lcolors[0])
    line2 = plt.Line2D(range(10), range(10), marker='o', color=lcolors[1])
    line3 = plt.Line2D(range(10), range(10), marker='o', color=lcolors[2])
    line4 = plt.Line2D(range(10), range(10), marker='o', color=lcolors[3])
    ax.legend((line1, line2, line3, line4),
              gold_members, numpoints=2, loc='best', fontsize=7)


def set_upper_left_corner(ul_x, ul_y):
    """Only works with PyQT this way!"""
    mngr = plt.get_current_fig_manager()
    # to put it into the upper left corner for example:
    geom = mngr.window.geometry()
    _, _, dx, dy = geom.getRect()
    mngr.window.setGeometry(ul_x, ul_y, dx, dy)


def diffangle(v1, v2, rads=True):
    """ Returns the angle in radians between vectors 'v1' and 'v2'."""
    cosang = np.dot(v1, v2)
    sinang = LA.norm(np.cross(v1, v2))
    res = np.arctan2(sinang, cosang)
    return res if rads else degrees(res)


def set_subframe_size(ax):
    """Set plot view limit on Planet 4 subframe size."""
    ax.set_xlim(0, img_x_size)
    ax.set_ylim(img_y_size, 0)


class ImageID(object):

    """Manage Planet 4 Image ids, getting data, plot stuff etc.

    At init this class will get the data for the given `imgid` using either the latest
    found database file or the optionally provided one.
    Parameters
    ----------
    imgid : str
        Planet4 image_id
    database_fname : str, optional
        Filepath to database name. The marking data for `imgid` will be extracted.
        Default: Latest one.
    data : pd.DataFrame, optional
        If the data was already extracted before init, it can be provided here.
    """
    def __init__(self, imgid, database_fname=None, data=None):
        self.imgid = io.check_and_pad_id(imgid)
        if data is not None:
            self.data = data
        else:
            if database_fname is None:
                database_fname = str(io.get_current_database_fname())
            self.data = pd.read_hdf(database_fname, 'df',
                                    where='image_id=' + self.imgid)

    @property
    def subframe(self):
        "np.array : Get tile url and return image tile using io funciton."
        url = self.data.iloc[0].image_url
        return io.get_subframe(url)

    def get_fans(self, user_name=None, without_users=None):
        """Return only data for fan markings."""
        mask = self.data.marking == 'fan'
        if user_name is not None:
            mask = (mask) & (self.data.user_name == user_name)
        if without_users is not None:
            mask = (mask) & (~self.data.user_name.isin(without_users))
        return self.data[mask]

    def get_blotches(self, user_name=None, without_users=None):
        """Return data for blotch markings."""
        mask = self.data.marking == 'blotch'
        if user_name is not None:
            mask = (mask) & (self.data.user_name == user_name)
        if without_users is not None:
            mask = (mask) & (~self.data.user_name.isin(without_users))
        return self.data[mask]

    def show_subframe(self, ax=None, aspect='auto'):
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=calc_fig_size(8))
        ax.imshow(self.subframe, origin='upper', aspect=aspect)
        ax.set_axis_off()
        if fig is not None:
            return fig

    def plot_blotches(self, n=None, img=True, user_name=None, ax=None,
                      user_color=None, without_users=None, blotches=None):
        """Plotting blotches using Blotch class and self.subframe."""
        if blotches is None:
            blotches = [Blotch(i) for _, i in
                        self.get_blotches(user_name, without_users).iterrows()]
        if ax is None:
            _, ax = plt.subplots(figsize=calc_fig_size(8))
        if img:
            self.show_subframe(ax)
        if n is None:
            n = len(blotches)
        for i, color in zip(range(len(blotches)), colors):
            if user_color is not None:
                color = user_color
            blotch = blotches[i]
            blotch.set_color(color)
            ax.add_patch(blotch)
            blotch.plot_center(ax, color=color)
            if i == n - 1:
                break
        set_subframe_size(ax)
        ax.set_axis_off()

    def plot_fans(self, n=None, img=True, user_name=None, ax=None,
                  user_color=None, without_users=None, fans=None):
        """Plotting fans using Fans class and self.subframe."""
        if fans is None:
            fans = [Fan(i) for _, i in
                    self.get_fans(user_name, without_users).iterrows()]
        if ax is None:
            _, ax = plt.subplots(figsize=calc_fig_size(8))
        if img:
            self.show_subframe(ax)
        if n is None:
            n = len(fans)
        for i, color in zip(range(len(fans)), colors):
            if user_color is not None:
                color = user_color
            fan = fans[i]
            fan.set_color(color)
            ax.add_line(fan)
            fan.add_semicircle(ax, color=color)
            fan.plot_center(ax, color=color)
            if i == n - 1:
                break
        set_subframe_size(ax)
        ax.set_axis_off()

    def plot_all(self):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.ravel()
        self.show_subframe(ax=axes[0])
        self.show_subframe(ax=axes[2])
        self.plot_fans(ax=axes[1])
        self.plot_blotches(ax=axes[3])
        for ax in axes:
            ax.set_axis_off()
        fig.subplots_adjust(left=None, top=None, bottom=None, right=None,
                            wspace=1e-3, hspace=1e-3)
        plt.show()


class Blotch(Ellipse):

    """Blotch management class for P4.

    Parameters
    ----------
    data : object with blotch data attributes
        object should provide attributes [`x`, `y`, `radius_1`, `radius_2`, `angle`]
    color : str, optional
        to control the color of the mpl.Ellipse object

    Attributes
    ----------
    to_average : list
        List of cols to be averaged after clustering
    data : object with blotch data attributes, as provided by `data`
    center : tuple (inherited from matplotlib.Ellipse)
        Coordinates of center, i.e. self.x, self.y
    """

    to_average = 'x y image_x image_y angle radius_1 radius_2'.split()

    def __init__(self, data, color='b'):
        try:
            self.x = data.x
            self.y = data.y
        except AttributeError:
            print("No x and y attributes in data:\n{}"
                  .format(data))
            raise AttributeError
        super(Blotch, self).__init__((self.x, self.y),
                                     data.radius_1 * 2, data.radius_2 * 2,
                                     data.angle, alpha=0.65,
                                     fill=False, linewidth=1, color=color)
        self.data = data

    @property
    def area(self):
        return pi * self.data.radius_1 * self.data.radius_2

    @property
    def x1(self):
        return math.cos(math.radians(self.angle)) * self.data.radius_1

    @property
    def y1(self):
        return math.sin(self.angle) * self.data.radius_1

    @property
    def p1(self):
        return np.array(self.center) + np.array([self.x1, self.y1])

    @property
    def p2(self):
        return np.array(self.center) - np.array([self.x1, self.y1])

    @property
    def x2(self):
        return math.cos(math.radians(self.angle+90)) * self.data.radius_2

    @property
    def y2(self):
        return math.sin(math.radians(self.angle+90)) * self.data.radius_2

    @property
    def p3(self):
        return np.array(self.center) + np.array([self.x2, self.y2])

    @property
    def p4(self):
        return np.array(self.center) - np.array([self.x2, self.y2])

    @property
    def limit_points(self):
        return [self.p1, self.p2, self.p3, self.p4]

    def plot_center(self, ax, color='b'):
        ax.scatter(self.x, self.y, color=color,
                   s=20, c='b', marker='.')

    def plot_limit_points(self, ax, color='b'):
        for x, y in self.limit_points:
            ax.scatter(x, y, color=color, s=20, c='b', marker='o')

    def store(self, fpath=None):
        out = self.data
        for p in range(1, 5):
            attr = 'p'+str(p)
            point = getattr(self, attr)
            out[attr+'_x'] = point[0]
            out[attr+'_y'] = point[1]
        if 'image_id' not in out.index:
            out['image_id'] = self.image_id
        if fpath is not None:
            out.to_hdf(str(fpath.with_suffix('.hdf')), 'df')
        return out

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.__str__()


class Fan(lines.Line2D):

    """Fan management class for P4.

    Parameters
    ----------
    data : object with fan data attributes
        object has to provide [`x`, `y`, `angle`, `spread`, `distance`]
    kwargs : dictionary, optional

    Attributes
    ----------
    to_average : list
        List of columns to average after clustering
    data : object with fan data attributes
        as provided by `data`.
    base : tuple
        base coordinates `x` and `y`.
    inside_half : float
        `data` divided by 2.0.
    armlength : float
        length of the fan arms.
    v1 : float[2]
        vector of first arm of fan.
    v2 : float[2]
        vector of second arm of fan.
    coords : float[3, 2]
        Set of coords to draw for MPL.Line2D object: arm1->base->arm2
    circle_base
    center
    radius
    midpoint
    base_to_midpoint_vec
    """

    to_average = 'x y image_x image_y angle spread distance'.split()

    def __init__(self, data, linewidth=0.5, **kwargs):
        self.data = data
        # first coordinate is the base of fan
        try:
            self.base = self.data.loc[['x', 'y']].values.astype('float')
        except KeyError:
            print("No x and y in the data:\n{}".format(data))
            raise KeyError
        # angles
        self.inside_half = self.data.spread / 2.0
        alpha = self.data.angle - self.inside_half
        beta = self.data.angle + self.inside_half
        # length of arms
        self.armlength = self.get_arm_length()
        # first arm
        self.v1 = self.rotate_vector([self.armlength, 0], alpha)
        # second arm
        self.v2 = self.rotate_vector([self.armlength, 0], beta)
        # vector matrix, stows the 1D vectors row-wise
        self.coords = np.vstack((self.base + self.v1,
                                 self.base,
                                 self.base + self.v2))
        # init fan line, first column are the x-components of the row-vectors
        lines.Line2D.__init__(self, self.coords[:, 0], self.coords[:, 1],
                              alpha=0.65, linewidth=linewidth, color='white',
                              **kwargs)

    def rotate_vector(self, v, angle):
        """Rotate vector by angle given in degrees."""
        rangle = radians(angle)
        rotmat = np.array([[cos(rangle), -sin(rangle)],
                           [sin(rangle), cos(rangle)]])
        return rotmat.dot(v)

    def get_arm_length(self):
        half = radians(self.inside_half)
        return self.data.distance / (cos(half) + sin(half))

    @property
    def circle_base(self):
        "float[2] : Vector between end of first arm and second arm of fan."
        return self.v1 - self.v2

    @property
    def center(self):
        """float[2] : vector from base to mid-point between arms.

        This is used for the drawing of the semi-circle at the end of the
        two fan arms.
        """
        return self.base + self.v2 + 0.5 * self.circle_base

    @property
    def radius(self):
        "float : for the semi-circle wedge drawing at the end of fan."
        return 0.5 * LA.norm(self.circle_base)

    def add_semicircle(self, ax, color='b'):
        "Draw a semi-circle at end of fan arms using MPL.Wedge."
        # reverse order of arguments for arctan2 input requirements
        theta1 = degrees(arctan2(*self.circle_base[::-1]))
        theta2 = theta1 + 180
        wedge = mpatches.Wedge(self.center, self.radius, theta1, theta2,
                               width=0.01 * self.radius, color=color, alpha=0.65)
        ax.add_patch(wedge)

    def add_mean_wind_pointer(self, ax, color='b', ls='-'):
        "Draw a thicker mean wind direction pointer for better visibility in plots."
        endpoint = self.rotate_vector([5 * self.armlength, 0], self.data.angle)
        coords = np.vstack((self.base,
                            self.base + endpoint))
        pointer = lines.Line2D(coords[:, 0], coords[:, 1],
                               alpha=0.65, linewidth=3, linestyle=ls)
        pointer.set_color(color)
        ax.add_line(pointer)

    @property
    def midpoint(self):
        """Calculate vector to half total length.

        As total length, I define the armlength + the radius of the semi-circle
        at the end.
        """
        mid_point_vec = self.rotate_vector([0.5 * (self.armlength + self.radius),
                                            0],
                                           self.data.angle)
        return self.base + mid_point_vec

    def plot_center(self, ax, color='b'):
        ax.scatter(self.midpoint[0], self.midpoint[1], color=color,
                   s=20, c='b', marker='.')

    @property
    def base_to_midpoint_vec(self):
        coords = np.vstack((self.base, self.midpoint))
        return coords

    def add_midpoint_pointer(self, ax, color='b', ls='-'):
        coords = self.base_to_midpoint_vec
        pointer = lines.Line2D(coords[:, 0], coords[:, 1],
                               alpha=0.65, linewidth=3, linestyle=ls)
        pointer.set_color(color)
        ax.add_line(pointer)

    def plot(self):
        _, ax = plt.subplots()
        img = np.ones(img_shape)
        ax.imshow(img)
        ax.add_line(self)
        plt.show()

    def __str__(self):
        out = 'base: {0}\narmlength: {1}\narm1: {2}\narm2: {3}'\
            .format(self.base, self.armlength, self.base+self.v1,
                    self.base+self.v2)
        return out

    def __repr__(self):
        return self.__str__()

    def store(self, fpath=None):
        out = self.data
        for i, arm in enumerate([self.v1, self.v2]):
            out['arm{}_x'.format(i+1)] = (self.base + arm)[0]
            out['arm{}_y'.format(i+1)] = (self.base + arm)[1]
        if 'image_id' not in out.index:
            out['image_id'] = self.image_id
        if fpath is not None:
            out.to_hdf(str(fpath.with_suffix('.hdf')), 'df')
        return out


class Fnotch(object):

    """Manage Fnotch by providing a cut during output.

    Parameters
    ----------
    value : float
        Fnotch value (= 1 - blotchiness), as calculated in clustering.ClusterManager()
    fandata : pandas.Series
        data set containing all required for Fan object (see `Fan`)
    blotchdata : pandas.Series
        data set containing all required for Blotch object (see `Blotch`)
    """

    @classmethod
    def from_series(cls, series):
        "Create Fnotch instance from series with fan_ and blotch_ indices."
        fan = Fan(series.filter(regex='fan_').rename(lambda x: x[4:]))
        blotch = Blotch(series.filter(regex='blotch_').rename(lambda x: x[7:]))
        return cls(series.fnotch_value, fan, blotch)

    def __init__(self, value, fan, blotch):
        self.value = value
        self.fandata = fan.data
        self.blotchdata = blotch.data

        fanstore = fan.store().copy()  # copy(),otherwise renaming original
        fanstore.rename_axis(lambda x: 'fan_'+x, inplace=True)
        blotchstore = blotch.store().copy()  # copy(),otherwise renaming original
        blotchstore.rename_axis(lambda x: 'blotch_'+x, inplace=True)
        df = pd.concat([fan.data, blotch.data])
        df['fnotch_value'] = self.value
        self.fan = fan
        self.fanstore = fanstore
        self.blotch = blotch
        self.blotchstore = blotchstore

    def get_marking(self, cut):
        """Return the right marking, depending on cut value.

        If the cut is at 0.8, the fnotch value has to be equal or better before
        we assign the fan to the Fnotch object. Otherwise we return a blotch.

        Parameters
        ----------
        cut : float
            Level where we separate fan from blotch

        Returns
        -------
        `Fan` or `Blotch` object, depending on `cut`
        """
        if cut > self.value:
            return Blotch(self.blotchdata)
        else:
            return Fan(self.fandata)

    def __str__(self):
        out = "Fnotch value: {:.2f}\n\n".format(self.value)
        out += "Fan:\n{}\n\n".format(self.fandata.__str__())
        out += "Blotch:\n{}".format(self.blotchdata.__str__())
        return out

    def __repr__(self):
        return self.__str__()

    def store(self, fpath=None):
        out = pd.concat([self.fanstore, self.blotchstore])
        out['fnotch_value'] = self.value
        if fpath is not None:
            out.to_hdf(str(fpath.with_suffix('.hdf')), 'df')
        return out


class Container(object):

    @classmethod
    def from_df(cls, df):
        rows = [i for _, i in df.iterrows()]
        return cls(rows)

    @classmethod
    def from_fname(cls, fname):
        df = pd.read_hdf(str(fname))
        return cls.from_df(df)


class FanContainer(Container):

    def __init__(self, iterable):
        self.content = [Fan(item) for item in iterable]


class BlotchContainer(Container):

    def __init__(self, iterable):
        self.content = [Blotch(item) for item in iterable]


class FnotchContainer(Container):

    def __init__(self, iterable):
        super().__init__(iterable, Fnotch.from_series)


def main():
    plt.switch_backend('Qt4Agg')
    parser = argparse.ArgumentParser()
    parser.add_argument('imgid',
                        help='zooniverse image id to plot')
    args = parser.parse_args()

    imgid = ImageID(args.imgid)
    imgid.plot_all()


if __name__ == '__main__':
    main()
