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

data_root = '/Users/maye/data/planet4'

img_x_size = 840
img_y_size = 648


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


class P4_Marking(object):
    """Base class for P4 markings."""
    def __init__(self, json_row):
        super(P4_Marking, self).__init__()
        self.data = json_row


class P4_Blotch(P4_Marking):
    """Blotch management class for P4."""

    def plot_center(self, ax, color='b'):
        ax.scatter(self.data.x, self.data.y, color=color,
                   s=20, c='b', marker='o')

    def get_ellipse(self, color='b'):
        return Ellipse((self.data.x, self.data.y),
                       self.data.radius_1, self.data.radius_2,
                       self.data.angle,
                       fill=False, linewidth=1, color=color)

    def add_ellipse(self, ax, color='b', ellipse=None):
        if ellipse is None:
            ellipse = self.get_ellipse()
        ellipse.set_color(color)
        ax.add_artist(ellipse)
        # do this better later in a plot controller
        ax.set_xlim(0, img_x_size)
        ax.set_ylim(0, img_y_size)


class P4_Fan(lines.Line2D):
    """Fan management class for P4. """

    def __init__(self, json_row):
        self.data = json_row
        # first coordinate is the base of fan
        self.x = self.data.x
        self.y = self.data.y
        inside_half = self.data.spread / 2.0
        alpha = self.data.angle - inside_half
        length = self.data.distance / cos(radians(inside_half))
        beta = alpha + self.data.spread
        # first arm
        self.line_x = [self.x + length * cos(radians(alpha))] + [self.x]
        self.line_y = [self.y + length * sin(radians(alpha))] + [self.y]
        # second arm
        self.line_x += [self.x + length * cos(radians(beta))]
        self.line_y += [self.y + length * sin(radians(beta))]
        lines.Line2D.__init__(self, self.line_x, self.line_y)
        ax = plt.gca()
        ax.set_xlim(0, img_x_size)
        ax.set_ylim(0, img_y_size)

    def __str__(self):
        out = 'x: {0}\ny: {1}\nline_x: {2}\nline_y: {3}'\
            .format(self.x, self.y, self.line_x, self.line_y)
        return out
