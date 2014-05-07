from matplotlib.patches import Ellipse
import pandas as pd
import os
from math import sin, cos, radians
import sys
import urllib
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import matplotlib.lines as lines
from itertools import cycle

data_root = '/Users/maye/data/planet4'

img_x_size = 840
img_y_size = 648


def set_subframe_size(ax):
    ax.set_xlim(0, img_x_size)
    ax.set_ylim(0, img_y_size)


class P4_ImgID(object):
    """docstring for P4_Img_ID"""
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
        return self.data[self.data.marking == 'fan']

    def get_blotches(self):
        return self.data[self.data.marking == 'blotch']

    def plot_blotches(self):
        blotches = self.get_blotches()
        fig, ax = plt.subplots()
        ax.imshow(self.get_subframe())
        colors = cycle('rgbcymk')
        for i, color in zip(xrange(len(blotches)), colors):
            blotch = P4_Blotch(blotches.iloc[i])
            blotch.set_color(color)
            ax.add_artist(blotch)
            blotch.plot_center(ax, color=color)


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
        # angles
        inside_half = self.data.spread / 2.0
        alpha = self.data.angle - inside_half
        beta = alpha + self.data.spread
        # length of arms
        length = self.data.distance / cos(radians(inside_half))
        # first arm
        self.line_x = [self.x + length * cos(radians(alpha))] + [self.x]
        self.line_y = [self.y + length * sin(radians(alpha))] + [self.y]
        # second arm
        self.line_x += [self.x + length * cos(radians(beta))]
        self.line_y += [self.y + length * sin(radians(beta))]
        # init fan line
        lines.Line2D.__init__(self, self.line_x, self.line_y)
        # grap the axis and set its view to subframe size
        set_subframe_size(plt.gca())

    def __str__(self):
        out = 'x: {0}\ny: {1}\nline_x: {2}\nline_y: {3}'\
            .format(self.x, self.y, self.line_x, self.line_y)
        return out

