
"""Checks if a given html file was processed by given user name using
the current available data file."""
from HTMLParser import HTMLParser
import pandas as pd
import sys


class MyHTMLParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.container = []

    def handle_starttag(self, tag, attrs):
        if attrs and attrs[0][0] == 'href':
            url = attrs[0][1]
            if '#' in url:
                self.container.append(url.split('/')[-1])

parser = MyHTMLParser()
# first argument should be the html file from Meg to check:
fname = sys.argv[1]

with open(fname) as f:
    parser.feed(f.read())

user_name = 'michaelaye'
df = pd.read_hdf('/Users/maye/data/planet4/'
                 '2014-06-09_planet_four_classifications_queryable.h5',
                 'df', where='user_name={0}'.format(user_name))

check = pd.DataFrame(parser.container, columns=['ids_to_test'])

check['status'] = check.ids_to_test.isin(df.image_id)

if check.status.all():
    print("All ids done.")
else:
    print("The following still need to be done:")
    print(check[~check.status].ids_to_test)
