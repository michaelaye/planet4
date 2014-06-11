import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from planet4 import markings
import os
import sys

# database name
fname = ('/Users/maye/data/planet4/'
         '2014-06-09_planet_four_classifications_queryable.h5')

# read the common gold_ids to check
with open('../data/gold_standard_commons.txt') as f:
    gold_ids = f.read()
gold_ids = gold_ids.split('\n')
del gold_ids[-1]  # last one is empty

gold_id = markings.P4_ImgID(gold_ids[2], fname)

try:
    start = int(sys.argv[1])
    end = int(sys.argv[2])
except IndexError:
    start = None
    end = None

for id in gold_ids[start:end]:
    print(id)
    gold_id = markings.P4_ImgID(id, fname)
    my_dpi = 96
    fig, axes = plt.subplots(2, 2,
                             figsize=(1280/my_dpi, 1024/my_dpi),
                             dpi=my_dpi)
    axes = axes.flatten()
    # markings.set_upper_left_corner(0, 0)
    for goldstar, color in zip(markings.gold_members,
                               markings.gold_plot_colors):
        gold_id.plot_fans(user_name=goldstar, ax=axes[0], user_color=color)
        gold_id.plot_blotches(user_name=goldstar, ax=axes[0], user_color=color)
    gold_id.plot_fans(ax=axes[2], without_users=markings.gold_members)
    gold_id.plot_blotches(ax=axes[2], without_users=markings.gold_members)
    for i in [1, 3]:
        gold_id.show_subframe(axes[i])
        axes[i].set_title("Original")
    axes[0].set_title("Gold stars")
    axes[2].set_title("Citizens")
    markings.gold_legend(axes[0])
    if not os.path.exists('plots'):
        os.mkdir('plots')
    fig.suptitle(id)
    plt.savefig('{folder}/{id}.png'.format(folder='plots', id=gold_id.imgid),
                dpi=2*my_dpi)
    plt.close(fig)
# plt.show()
