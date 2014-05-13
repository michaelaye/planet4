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

data_root = '/Users/maye/data/planet4'

img_x_size = 840
img_y_size = 648

colors = cycle('rgbcymk')


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


class P4_ImgID(object):
    """Manage Planet 4 Image ids, getting data, plot stuff etc."""
    def __init__(self, imgid):
        super(P4_ImgID, self).__init__()
        self.imgid = imgid
        self.data = pd.read_hdf(os.path.join(data_root, 'marked.h5'), 'df',
                                where='image_id=='+imgid)

    def get_subframe(self):
        """Download image if not there yet and return numpy array.

        Takes a data record (called 'line'), picks out the image_url.
        First checks if the name of that image is already stored in
        the image path. If not, it grabs it from the server.
        Then uses matplotlib.image to read the image into a numpy-array
        and finally returns it.
        """
        url = self.data.iloc[0].image_url
        targetpath = os.path.join(data_root, 'images', os.path.basename(url))
        if not os.path.exists(targetpath):
            print("Did not find image in cache. Downloading ...")
            sys.stdout.flush()
            path = urllib.urlretrieve(url)[0]
            print("Done.")
            shutil.move(path, targetpath)
        else:
            print("Found image in cache.")
        im = mplimg.imread(targetpath)
        return im

    def get_fans(self):
        """Return only data for fan markings."""
        return self.data[self.data.marking == 'fan']

    def get_blotches(self):
        """Return data for blotch markings."""
        return self.data[self.data.marking == 'blotch']

    def plot_blotches(self, n=None, img=True):
        """Plotting blotches using P4_Blotch class and self.get_subframe."""
        blotches = self.get_blotches()
        fig, ax = plt.subplots()
        if img:
            ax.imshow(self.get_subframe(), origin='upper')
        if n is None:
            n = len(blotches)
        for i, color in zip(xrange(len(blotches)), colors):
            blotch = P4_Blotch(blotches.iloc[i])
            blotch.set_color(color)
            ax.add_artist(blotch)
            blotch.plot_center(ax, color=color)
            if i == n-1:
                break
        set_subframe_size(ax)

    def plot_fans(self, n=None, img=True):
        """Plotting fans using P4_Fans class and self.get_subframe."""
        fans = self.get_fans()
        fig, ax = plt.subplots()
        if n is None:
            n = len(fans)
        if img:
            ax.imshow(self.get_subframe(), origin='upper')
        for i, color in zip(xrange(len(fans)), colors):
            fan = P4_Fan(fans.iloc[i])
            fan.set_color(color)
            ax.add_line(fan)
            fan.add_semicircle(ax, color=color)
            if i == n-1:
                break
        set_subframe_size(ax)


class P4_Blotch(Ellipse):
    """Blotch management class for P4."""
    def __init__(self, json_row, color='b'):
        data = json_row
        super(P4_Blotch, self).__init__((data.x, data.y),
                                        data.radius_1, data.radius_2,
                                        data.angle,
                                        fill=False, linewidth=1, color=color)
        self.data = data

    def plot_center(self, ax, color='b'):
        ax.scatter(self.data.x, self.data.y, color=color,
                   s=20, c='b', marker='o')


class P4_Fan(lines.Line2D):
    """Fan management class for P4. """

    def __init__(self, json_row):
        self.data = json_row
        # first coordinate is the base of fan
        self.x = self.data.x
        self.y = self.data.y
        self.base = np.array([self.x, self.y])
        # angles
        inside_half = self.data.spread / 2.0
        alpha = self.data.angle - inside_half
        beta = alpha + self.data.spread
        # length of arms
        length = self.get_arm_length()
        # first arm
        self.p1 = self.base + rotate_vector([length, 0], alpha)
        # second arm
        self.p2 = self.base + rotate_vector([length, 0], beta)
        # vector matrix
        self.vectors = np.vstack((self.p1, self.base, self.p2))
        # init fan line
        lines.Line2D.__init__(self, self.vectors[:, 0], self.vectors[:, 1])
        # grap the axis and set its view to subframe size
        set_subframe_size(plt.gca())

    def get_arm_length(self):
        half = radians(self.data.spread / 2.0)
        return self.data.distance / (cos(half) + sin(half))

    def add_semicircle(self, ax, color='b'):
        circle_base = self.p1 - self.p2
        center = self.p2 + 0.5 * circle_base
        radius = 0.5 * LA.norm(circle_base)
        theta1 = degrees(arctan2(*-circle_base))
        theta2 = theta1 + 180
        wedge = mpatches.Wedge(center, radius, theta1, theta2,
                               width=0.01*radius, color=color)
        ax.add_patch(wedge)

    def __str__(self):
        out = 'x: {0}\ny: {1}\nline_x: {2}\nline_y: {3}'\
            .format(self.x, self.y, self.line_x, self.line_y)
        return out

