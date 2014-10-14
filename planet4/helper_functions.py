import pandas as pd
import numpy as np
import os
from math import pi

size_of_unique = lambda x: x.unique().size
data_root = '/Users/maye/data/planet4'
done_path = os.path.join(data_root, 'done.h5')


def classification_counts_per_user(df):
    return df.classification_id.groupby(df.user_name,
                                        sort=False).agg(size_of_unique)


def get_top_ten_users(df):
    users_work = classification_counts_per_user(df)
    return users_work.order(ascending=False)[:10]


def classification_counts_per_image(df):
    """Main function to help defining status of P4"""
    return df.classification_id.groupby(df.image_id,
                                        sort=False).agg(size_of_unique)


def get_no_done(df, limit=30):
    counts = classification_counts_per_image(df)
    no_done = counts[counts >= limit].size
    return no_done


def get_status(df, limit=30):
    no_all = len(df.image_id.unique())
    no_done = get_no_done(df, limit)
    return np.round(100.0 * no_done / no_all, 1)


def classification_counts_for_user(username, df):
    return df[df.user_name == username].classification_id.value_counts()


def no_of_classifications_per_user(df):
    return df.user_name.groupby(df.image_id,
                                sort=False).agg(size_of_unique)


def get_blotch_area(record):
    if record.marking != 'blotch':
        return 0
    else:
        return pi*record.radius_1*record.radius_2


###
### Season related stuff
###


def unique_image_ids_per_season(df):
    return df.image_id.groupby(df.season, sort=False).agg(size_of_unique)


def define_season_column(df):
    thousands = df.image_name.str[5:7].astype('int')
    df['season'] = 0
    df.loc[:, 'season'][df.image_name.str.startswith('PSP')] = 1
    df.loc[:, 'season'][(thousands > 10) & (thousands < 20)] = 2
    df.loc[:, 'season'][thousands > 19] = 3
