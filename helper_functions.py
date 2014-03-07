import pandas as pd
import numpy as np
import os
from math import pi

size_of_unique = lambda x: x.unique().size
data_root = '/Users/maye/data/planet4'
done_path = os.path.join(data_root, 'done.h5')

def classification_counts_per_user(df):
    return df.classification_id.groupby(df.user_name, sort=False).agg(size_of_unique)


def get_top_ten_users(df):
    users_work = classification_counts_per_user(df)
    return users_work.order(ascending=False)[:10]
    

def classification_counts_per_image(df):
    """Main function to help defining status of P4"""
    return df.classification_id.groupby(df.image_id, sort=False).agg(size_of_unique)


def get_status(df, limit=30):
    no_all = len(df.image_id.unique())
    counts = classification_counts_per_image(df)
    no_done = counts[counts >= limit].size
    return np.round(100.0 * no_done / no_all, 1)


def classification_counts_for_user(username, df):
    return df[df.user_name==username].classification_id.value_counts()


def no_of_classifications_per_user(df):
    return df.user_name.groupby(df.image_id, sort=False).agg(size_of_unique)


def get_blotch_area(record):
    if record.marking != 'blotch':
        return 0
    else:
        return pi*record.radius_1*record.radius_2