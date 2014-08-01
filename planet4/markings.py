from matplotlib.patches import Ellipse
import pandas as pd
import os
from math import sin, cos, radians, degrees
import sys
import urllib
import shutil
import numpy as np
from numpy import arctan2
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import matplotlib.lines as lines
import matplotlib.patches as mpatches
from itertools import cycle
import logging
import get_data

data_root = '/Users/maye/data/planet4'

img_x_size = 840
img_y_size = 648

colors = cycle('rgbcym')

gold_members = ['michaelaye', 'mschwamb', 'Portyankina', 'CJ-DPI']
gold_plot_colors = list('cmyg')


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
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
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
    def __init__(self, imgid, database_fname=None, data=None):
        super(ImageID, self).__init__()
        self.imgid = imgid
        if data is not None:
            self.data = data
        else:
            if database_fname is None:
                database_fname = get_data.get_current_database_fname()
            self.data = pd.read_hdf(database_fname, 'df',
                                    where='image_id=='+imgid)

    @property
    def subframe(self):
        url = self.data.iloc[0].image_url
        return get_data.get_subframe(url)

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
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(self.subframe(), origin='upper', aspect=aspect)

    def plot_blotches(self, n=None, img=True, user_name=None, ax=None,
                      user_color=None, without_users=None):
        """Plotting blotches using Blotch class and self.subframe."""
        blotches = self.get_blotches(user_name, without_users)
        if ax is None:
            _, ax = plt.subplots()
        if img:
            self.show_subframe(ax)
        if n is None:
            n = len(blotches)
        for i, color in zip(xrange(len(blotches)), colors):
            if user_color is not None:
                color = user_color
            blotch = Blotch(blotches.iloc[i])
            blotch.set_color(color)
            ax.add_artist(blotch)
            # blotch.plot_center(ax, color=color)
            if i == n-1:
                break
        set_subframe_size(ax)

    def plot_fans(self, n=None, img=True, user_name=None, ax=None,
                  user_color=None, without_users=None):
        """Plotting fans using Fans class and self.subframe."""
        fans = self.get_fans(user_name, without_users)
        if ax is None:
            fig, ax = plt.subplots()
        if img:
            self.show_subframe(ax)
        if n is None:
            n = len(fans)
        for i, color in zip(xrange(len(fans)), colors):
            if user_color is not None:
                color = user_color
            fan = Fan(fans.iloc[i])
            fan.set_color(color)
            ax.add_line(fan)
            fan.add_semicircle(ax, color=color)
            if i == n-1:
                break
        set_subframe_size(ax)


class Blotch(Ellipse):
    """Blotch management class for P4."""
    def __init__(self, json_row, color='b'):
        data = json_row
        super(Blotch, self).__init__((data.x, data.y),
                                     data.radius_1*2, data.radius_2*2,
                                     data.angle, alpha=0.65,
                                     fill=False, linewidth=2, color=color)
        self.data = data

    def plot_center(self, ax, color='b'):
        ax.scatter(self.data.x, self.data.y, color=color,
                   s=20, c='b', marker='o')


class Fan(lines.Line2D):
    """Fan management class for P4. """

    def __init__(self, json_row):
        self.data = json_row
        # first coordinate is the base of fan
        self.base = self.data[['x', 'y']].values
        # angles
        self.inside_half = self.data.spread
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
                              alpha=0.65, linewidth=2)

    def get_arm_length(self):
        half = radians(self.inside_half)
        return self.data.distance / (cos(half) + sin(half))

    def add_semicircle(self, ax, color='b'):
        circle_base = self.v1 - self.v2
        center = self.base + self.v2 + 0.5 * circle_base
        radius = 0.5 * LA.norm(circle_base)
        # reverse order of arguments for arctan2 input requirements
        theta1 = degrees(arctan2(*circle_base[::-1]))
        theta2 = theta1 + 180
        wedge = mpatches.Wedge(center, radius, theta1, theta2,
                               width=0.01*radius, color=color, alpha=0.65)
        ax.add_patch(wedge)

    def __str__(self):
        out = 'base: {0}\nlength: {1}\nv1: {2}\nv2: {3}'\
            .format(self.base, self.length, self.v1, self.v2)
        return out
