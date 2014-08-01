import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from planet4 import markings
import os
import sys
import get_data


def gold_star_plotter(gold_id, axis, fans=False):
    for goldstar, color in zip(markings.gold_members,
                               markings.gold_plot_colors):
        gold_id.plot_blotches(user_name=goldstar, ax=axis, user_color=color)
        if fans:
            gold_id.plot_fans(user_name=goldstar, ax=axis, user_color=color)


def main():
    # database name
    fname = get_data.get_current_database_fname()

    # read the common gold_ids to check
    with open('../data/gold_standard_commons.txt') as f:
        gold_ids = f.read()
    gold_ids = gold_ids.split('\n')
    del gold_ids[-1]  # last one is empty

    try:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
    except IndexError:
        start = None
        end = None

    for imgid in gold_ids[start:end]:
        print(imgid)
        gold_id = markings.ImageID(imgid)
        my_dpi = 96
        fig, axes = plt.subplots(2, 2,
                                 figsize=(1280/my_dpi, 1024/my_dpi),
                                 dpi=my_dpi)
        axes = axes.flatten()
        # markings.set_upper_left_corner(0, 0)
        gold_star_plotter(gold_id, axes[0], fans=True)
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
        fig.suptitle(imgid)
        plt.savefig('{folder}/{imgid}.png'.format(folder='plots',
                                                  imgid=gold_id.imgid),
                    dpi=2*my_dpi)
        plt.close(fig)
    # plt.show()

if __name__ == '__main__':
    main()
