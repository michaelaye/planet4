#!/usr/bin/env python
import argparse
from itertools import cycle
from math import cos, degrees, radians, sin

import matplotlib.lines as lines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from numpy import linalg as LA
from numpy import arctan2

from . import p4io

data_root = p4io.data_root

img_x_size = 840
img_y_size = 648


img_shape = (img_y_size, img_x_size)

colors = cycle('rgbcym')

gold_members = ['michaelaye', 'mschwamb', 'Portyankina', 'CJ-DPI']
gold_plot_colors = list('cmyg')


def calc_fig_size(width):
    """Calc figure height in ratio of subframes."""
    ratio = img_x_size / img_y_size
    return (width, width / ratio)


def calc_4_3(width):
    return (width, width / (4 / 3.0))


def gold_legend(ax):
    colors = list('cmyg')
    line1 = plt.Line2D(range(10), range(10), marker='o', color=colors[0])
    line2 = plt.Line2D(range(10), range(10), marker='o', color=colors[1])
    line3 = plt.Line2D(range(10), range(10), marker='o', color=colors[2])
    line4 = plt.Line2D(range(10), range(10), marker='o', color=colors[3])
    ax.legend((line1, line2, line3, line4),
              gold_members, numpoints=2, loc='best', fontsize=7)


def set_upper_left_corner(ul_x, ul_y):
    """Only works with PyQT this way!"""
    mngr = plt.get_current_fig_manager()
    # to put it into the upper left corner for example:
    geom = mngr.window.geometry()
    x, y, dx, dy = geom.getRect()
    mngr.window.setGeometry(ul_x, ul_y, dx, dy)


def rotate_vector(v, angle):
    """Rotate vector by angle given in degrees."""
    rangle = radians(angle)
    rotmat = np.array([[cos(rangle), -sin(rangle)],
                       [sin(rangle), cos(rangle)]])
    return rotmat.dot(v)


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

    """Manage Planet 4 Image ids, getting data, plot stuff etc."""
    imgid_template = "APF0000000"

    def __init__(self, imgid, database_fname=None, data=None):
        super(ImageID, self).__init__()
        if len(imgid) < len(self.imgid_template):
            imgid = self.imgid_template[:-len(imgid)] + imgid
        self.imgid = imgid
        if data is not None:
            self.data = data
        else:
            if database_fname is None:
                database_fname = p4io.get_current_database_fname()
            self.data = pd.read_hdf(database_fname, 'df',
                                    where='image_id==' + imgid)

    @property
    def subframe(self):
        url = self.data.iloc[0].image_url
        return p4io.get_subframe(url)

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
            fig, ax = plt.subplots(figsize=calc_4_3(8))
        ax.imshow(self.subframe, origin='upper', aspect=aspect)
        if fig is not None:
            return fig

    def plot_blotches(self, n=None, img=True, user_name=None, ax=None,
                      user_color=None, without_users=None, blotches=None):
        """Plotting blotches using Blotch class and self.subframe."""
        fig = None
        if blotches is None:
            blotches = self.get_blotches(user_name, without_users)
        if ax is None:
            fig, ax = plt.subplots(figsize=calc_fig_size(8))
        if img:
            self.show_subframe(ax, aspect='equal')
        if n is None:
            n = len(blotches)
        for i, color in zip(range(len(blotches)), colors):
            if user_color is not None:
                color = user_color
            blotch = Blotch(blotches.iloc[i])
            blotch.set_color(color)
            ax.add_artist(blotch)
            # blotch.plot_center(ax, color=color)
            if i == n - 1:
                break
        set_subframe_size(ax)
        ax.set_axis_off()

    def plot_fans(self, n=None, img=True, user_name=None, ax=None,
                  user_color=None, without_users=None, fans=None):
        """Plotting fans using Fans class and self.subframe."""
        if fans is None:
            fans = self.get_fans(user_name, without_users)
        if ax is None:
            fig, ax = plt.subplots(figsize=calc_fig_size(8))
        if img:
            self.show_subframe(ax)
        if n is None:
            n = len(fans)
        for i, color in zip(range(len(fans)), colors):
            if user_color is not None:
                color = user_color
            fan = Fan(fans.iloc[i])
            fan.set_color(color)
            ax.add_line(fan)
            fan.add_semicircle(ax, color=color)
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
        plt.show()


class Blotch(Ellipse):

    """Blotch management class for P4.

    Parameters
    ----------
    json_row : object with blotch data attributes
        object should provide attributes [`x`, `y`, `radius_1`, `radius_2`, `angle`]
    color : str, optional
        to control the color of the mpl.Ellipse object

    Attributes
    ----------
    data : object with blotch data attributes, as provided by `json_row`
    """

    def __init__(self, json_row, color='b'):
        data = json_row
        try:
            self.x = data.x
            self.y = data.y
        except AttributeError:
            print("No x and y attributes in json_row:\n{}"
                  .format(json_row))
            raise AttributeError
        super(Blotch, self).__init__((self.x, self.y),
                                     data.radius_1 * 2, data.radius_2 * 2,
                                     data.angle, alpha=0.65,
                                     fill=False, linewidth=2, color=color)
        self.data = data

    def plot_center(self, ax, color='b'):
        ax.scatter(self.x, self.y, color=color,
                   s=20, c='b', marker='o')


class Fan(lines.Line2D):

    """Fan management class for P4.

    Parameters
    ----------
    json_row : object with fan data attributes
        object has to provide [`x`, `y`, `angle`, `spread`, `distance`]
    kwargs : dictionary, optional

    Attributes
    ----------
    data : object with fan data attributes
        as provided by `json_row`.
    base : tuple
        base coordinates `x` and `y`.
    inside_half : float
        `data` divided by 2.0.
    length : float
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

    def __init__(self, json_row, **kwargs):
        self.data = json_row
        # first coordinate is the base of fan
        try:
            self.base = self.data.loc[['x', 'y']].values
        except KeyError:
            print("No x and y in the json_row:\n{}".format(json_row))
            raise KeyError
        # angles
        self.inside_half = self.data.spread / 2.0
        alpha = self.data.angle - self.inside_half
        beta = self.data.angle + self.inside_half
        # length of arms
        self.length = self.get_arm_length()
        # first arm
        self.v1 = rotate_vector([self.length, 0], alpha)
        # second arm
        self.v2 = rotate_vector([self.length, 0], beta)
        # vector matrix, stows the 1D vectors row-wise
        self.coords = np.vstack((self.base + self.v1,
                                 self.base,
                                 self.base + self.v2))
        # init fan line, first column are the x-components of the row-vectors
        lines.Line2D.__init__(self, self.coords[:, 0], self.coords[:, 1],
                              alpha=0.65, linewidth=2, color='white',
                              **kwargs)

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
        endpoint = rotate_vector([5 * self.length, 0], self.data.angle)
        coords = np.vstack((self.base,
                            self.base + endpoint))
        pointer = lines.Line2D(coords[:, 0], coords[:, 1],
                               alpha=0.65, linewidth=3, linestyle=ls)
        pointer.set_color(color)
        ax.add_line(pointer)

    @property
    def midpoint(self):
        """Calculate vector to half total length.

        As total length, I define the arm-length + the radius of the semi-circle
        at the end.
        """
        mid_point_vec = rotate_vector([0.5 * (self.length + self.radius), 0],
                                      self.data.angle)
        return self.base + mid_point_vec

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
        fig, ax = plt.subplots()
        img = np.ones(img_shape)
        ax.imshow(img)
        ax.add_line(self)
        plt.show()

    def __str__(self):
        out = 'base: {0}\nlength: {1}\nv1: {2}\nv2: {3}'\
            .format(self.base, self.length, self.v1, self.v2)
        return out


class Fnotch(object):

    "Manage Fnotch by providing a cut during output."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('imgid',
                        help='zooniverse image id to plot')
    args = parser.parse_args()

    imgid = ImageID(args.imgid)
    imgid.plot_all()


if __name__ == '__main__':
    main()
