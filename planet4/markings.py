#!/usr/bin/env python
"""Module to manage Planet4 markings."""
import argparse
import logging
import math
from itertools import cycle
from math import cos, degrees, pi, radians, sin

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

    def __init__(self, imgid, database_fname=None, data=None, scope=None):
        self.imgid = io.check_and_pad_id(imgid)
        if data is not None:
            self.data = data
        else:
            if database_fname is None:
                db = io.DBManager()
            self.data = db.get_image_id_markings(self.imgid)
        self.scope = scope

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

    def plot_objects(self, objects, n=None, img=True, user_name=None, ax=None,
                     user_color=None, without_users=None):
        """Plotting either fans or blotches with p4 subframe background."""
        logging.debug("Entering markings.plot_objects")
        if ax is None:
            _, ax = plt.subplots(figsize=calc_fig_size(8))
            logging.debug("Created own axis.")
        if img:
            logging.debug("Plotting background image.")
            self.show_subframe(ax)
        counter = 0
        for obj, color in zip(objects, colors):
            if user_color is not None:
                color = user_color
            obj.plot(color, ax)
            counter += 1
            if counter == n:
                break
        set_subframe_size(ax)
        ax.set_axis_off()

    def plot_blotches(self, blotches=None, **kwargs):
        user_name = kwargs.pop('user_name', None)
        without_users = kwargs.pop('without_users', None)
        if blotches is None:
            blotches = [Blotch(i, self.scope) for _, i in
                        self.get_blotches(user_name, without_users).iterrows()]
        self.plot_objects(blotches, **kwargs)

    def plot_fans(self, fans=None, **kwargs):
        """Plotting fans using Fans class and self.subframe."""
        user_name = kwargs.pop('user_name', None)
        without_users = kwargs.pop('without_users', None)
        if fans is None:
            fans = [Fan(i, self.scope) for _, i in
                    self.get_fans(user_name, without_users).iterrows()]
        self.plot_objects(fans, **kwargs)

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
    scope : {'planet4', 'hirise'}
        string that decides between using x/y or image_x/image_y as center corods
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

    def __init__(self, data, scope, color='b'):
        self.data = data
        self.scope = scope
        if scope not in ['hirise', 'planet4']:
            raise TypeError('Unknown scope')
        try:
            self.x = data.x if scope == 'planet4' else data.image_x
            self.y = data.y if scope == 'planet4' else data.image_y
        except AttributeError:
            print("No x and y attributes in data:\n{}"
                  .format(data))
            raise AttributeError
        # default member number is 1. This is set to the cluster member inside
        # clustering execution.
        self._n_members = 1
        super(Blotch, self).__init__((self.x, self.y),
                                     data.radius_1 * 2, data.radius_2 * 2,
                                     data.angle, alpha=0.65,
                                     fill=False, linewidth=1, color=color)
        self.data = data

    def is_equal(self, other):
        if self.data.x == other.data.x and\
           self.data.y == other.data.y and\
           self.data.image_x == other.data.image_y and\
           self.data.image_y == other.data.image_y and\
           self.data.radius_1 == other.data.radius_1 and\
           self.data.radius_2 == other.data.radius_2 and\
           self.data.angle == other.data.angle:
            return True
        else:
            return False

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

    @property
    def n_members(self):
        return self._n_members

    @n_members.setter
    def n_members(self, value):
        self._n_members = value

    def plot(self, color='blue', ax=None):
        if ax is None:
            _, ax = plt.subplots()

        self.set_color(color)
        ax.add_patch(self)
        self.plot_center(ax, color=color)

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
        out['n_members'] = self.n_members
        return out

    def __str__(self):
        s = "markings.Blotch object. Input data:\n"
        s += self.data.__str__()
        s += '\n'
        s += "N_members: {}".format(self.n_members)
        return s

    def __repr__(self):
        return self.__str__()


class Fan(lines.Line2D):

    """Fan management class for P4.

    Parameters
    ----------
    data : object with fan data attributes
        object has to provide [`x`, `y`, `angle`, `spread`, `distance`]
    scope : {'planet4', 'hirise'}
        string that decides between using x/y or image_x/image_y as base coords
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

    def __init__(self, data, scope, linewidth=0.5, **kwargs):
        self.data = data
        self.scope = scope
        if scope not in ['hirise', 'planet4']:
            raise TypeError('Unknown scope')
        # first coordinate is the base of fan
        actual_x = 'x' if scope == 'planet4' else 'image_x'
        actual_y = 'y' if scope == 'planet4' else 'image_y'
        try:
            self.base = self.data.loc[[actual_x, actual_y]].values.astype('float')
        except KeyError:
            print("No x and y in the data:\n{}".format(data))
            raise KeyError
        # default n_members value (property)
        self._n_members = 1
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

    def is_equal(self, other):
        if self.data.x == other.data.x and\
           self.data.y == other.data.y and\
           self.data.image_x == other.data.image_x and\
           self.data.image_y == other.data.image_y and\
           self.data.angle == other.data.angle and\
           self.data.spread == other.data.spread and\
           self.data.distance == other.data.distance:
            return True
        else:
            return False

    @property
    def n_members(self):
        return self._n_members

    @n_members.setter
    def n_members(self, value):
        self._n_members = value

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
    def area(self):
        tr_h = np.sqrt(self.armlength**2 - self.radius**2)
        tr_area = tr_h*self.radius
        half_circ_area = 0.5 * pi * self.radius**2
        return tr_area + half_circ_area

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

    def plot(self, color='blue', ax=None):
        if ax is None:
            _, ax = plt.subplots()
        self.set_color(color)
        ax.add_line(self)
        self.add_semicircle(ax, color=color)
        self.plot_center(ax, color=color)

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
        out['n_members'] = self.n_members
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
    def from_series(cls, series, scope):
        "Create Fnotch instance from series with fan_ and blotch_ indices."
        fan = Fan(series.filter(regex='fan_').rename(lambda x: x[4:]),
                  scope=scope)
        blotch = Blotch(series.filter(regex='blotch_').rename(lambda x: x[7:]),
                        scope=scope)
        return cls(series.fnotch_value, fan, blotch, scope)

    def __init__(self, value, fan, blotch, scope):
        self.value = value
        self.fandata = fan.data
        self.blotchdata = blotch.data
        self.scope = scope

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
            return self.blotch
        else:
            return self.fan

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
    def from_df(cls, df, scope):
        rows = [i for _, i in df.iterrows()]
        return cls(rows, scope)

    @classmethod
    def from_fname(cls, fname, scope):
        if str(fname).endswith('.hdf'):
            readfunc = pd.read_hdf
        elif str(fname).endswith('.csv'):
            readfunc = pd.read_csv
        else:
            raise TypeError("Can only work with '.csv' or '.hdf' files.")
        df = readfunc(str(fname))
        return cls.from_df(df, scope)


class FanContainer(Container):

    def __init__(self, iterable, scope):
        self.content = [Fan(item, scope) for item in iterable]


class BlotchContainer(Container):

    def __init__(self, iterable, scope):
        self.content = [Blotch(item, scope) for item in iterable]


# class FnotchContainer(Container):
#     pass
#     def __init__(self, iterable, scope):
#         super().__init__(iterable, Fnotch.from_series)


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
