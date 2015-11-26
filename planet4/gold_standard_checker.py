#!/usr/bin/env python
"""Checks if a given html file was processed by given user name using
the current available data file."""
from __future__ import print_function, division
try:
    from HTMLParser import HTMLParser
except ImportError:
    from html.parser import HTMLParser
import pandas as pd
import argparse
from . import markings
from . import io


class MyHTMLParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.container = []

    def handle_starttag(self, tag, attrs):
        if attrs and attrs[0][0] == 'href':
            url = attrs[0][1]
            if '#' in url:
                self.container.append(url.split('/')[-1])


def main(fname, user_name='michaelaye', datadir=None):
    parser = MyHTMLParser()

    with open(fname) as f:
        parser.feed(f.read())

    dbname = io.get_current_database_fname(datadir)
    df = pd.read_hdf(dbname, 'df', where='user_name={0}'.format(user_name))

    check = pd.DataFrame(parser.container, columns=['ids_to_test'])

    check['Done'] = check.ids_to_test.isin(df.image_id)

    if check.Done.all():
        print("All ids done.")
    else:
        print("\nSome are not done yet. Here's the status for all:\n")
        print(check)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # first argument should be the html file from Meg to check:
    parser.add_argument('fname', help='HTML filename from Meg to check.')
    parser.add_argument('--user', help='username to check for',
                        choices=markings.gold_members,
                        default='michaelaye')
    parser.add_argument('--datadir', help='folder where the csv and h5 files'
                        ' are stored.')
    args = parser.parse_args()
    main(args.fname, user_name=args.user, datadir=args.datadir)
