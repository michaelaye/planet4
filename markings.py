from matplotlib.patches import Ellipse
from itertools import cycle
import pandas as pd
import os
import matplotlib.pyplot as plt
data_root = '/Users/maye/data/planet4'


class P4_ImgID(object):
    """docstring for P4_Img_ID"""
    def __init__(self, imgid):
        super(P4_ImgID, self).__init__()
        self.imgid = imgid
        self.data = pd.read_hdf(os.path.join(data_root, 'marked.h5'), 'df',
                                where='image_id=='+imgid)

    def get_fans(self):
        return self.data[self.data.marking == 'fan']

    def get_blotches(self):
        return self.data[self.data.marking == 'blotch']


class P4_Blotch(object):
    """docstring for P4_Blotch"""

    def __init__(self, json_line):
        super(P4_Blotch, self).__init__()
        self.line = json_line

    def scatter(self, ax, color='b'):
        ax.scatter(self.line.x, self.line.y, color=color,
                   s=20, c='b', marker='o')

    def get_ellipse(self, color='b'):
        return Ellipse((self.line.x, self.line.y),
                       self.line.radius_1, self.line.radius_2,
                       self.line.angle,
                       fill=False, linewidth=1, color=color)

    def add_ellipse(self, ax, color='b', ellipse=None):
        if ellipse is None:
            ellipse = self.get_ellipse()
        ellipse.set_color(color)
        ax.add_artist(ellipse)
