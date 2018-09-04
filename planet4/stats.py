import numpy as np
from math import pi
tau = 2 * pi


def get_fan_and_blotch_nunique_cids(data):
    f1 = data.marking == 'fan'
    f2 = data.marking == 'blotch'
    return data[f1 | f2].classification_id.nunique()


def get_fb_to_all_ratio(data):
    n_classifications = data.classification_id.nunique()
    n_class_fb = get_fan_and_blotch_nunique_cids(data)
    ratio = (n_class_fb / n_classifications)
    return ratio


def size_of_unique(x):
    return x.unique().size


def classification_counts_per_user(df):
    res = df.classification_id.groupby(df.user_name,
                                       sort=False).agg(size_of_unique).sort_values(ascending=False)
    return res


def get_top_ten_users(df):
    users_work = classification_counts_per_user(df)
    return users_work.order(ascending=False)[:10]


def classification_counts_per_image(df):
    """Main function to help defining status of P4"""
    return df.classification_id.groupby(df.image_id,
                                        sort=False).agg(size_of_unique)


def get_no_tiles_done(df, limit=30):
    counts = classification_counts_per_image(df)
    no_done = counts[counts >= limit].size
    return no_done


def get_status_per_classifications(df, limit=30):
    "Returns status in percent of limit*n_unique_image_ids."
    no_all = df.image_id.nunique()
    sum_classifications = classification_counts_per_image(df).sum()
    try:
        return np.round(100.0 * sum_classifications / (limit * no_all), 1)
    except ZeroDivisionError:
        return np.nan


def get_status_per_completed_tile(df, limit=30):
    no_all = len(df.image_id.unique())
    no_done = get_no_tiles_done(df, limit)
    try:
        return np.round(100.0 * no_done / no_all, 1)
    except ZeroDivisionError:
        return np.nan


def classification_counts_for_user(username, df):
    return df[df.user_name == username].classification_id.value_counts()


def no_of_classifications_per_user(df):
    return df.user_name.groupby(df.image_id,
                                sort=False).agg(size_of_unique)


def get_blotch_area(record):
    if record.marking != 'blotch':
        return 0
    else:
        return 0.5 * tau * record.radius_1 * record.radius_2


###
# Season related stuff
###


def unique_image_ids_per_season(df):
    return df.image_id.groupby(df.season, sort=False).agg(size_of_unique)


def define_season_column(df, colname='image_name'):
    """Create new column that indicates the MRO season.

    Seasons 1,2, and 3 are MY28, 29, and 30 respectively.

    Parameters:
    ----------
    df : {pandas.DataFrame}
        Dataframe that should have a column with name `colname` as deciding factor.
    colname : str
        Name of column to be used as HiRISE observation ID.
    """
    thousands = df[colname].str[5:7].astype('int')
    df['season'] = 0
    df.loc[df[colname].str.startswith('PSP'), 'season'] = 1
    df.loc[(thousands > 10) & (thousands < 15), 'season'] = 2
    df.loc[(thousands > 15) & (thousands < 25), 'season'] = 3
    df.loc[(thousands > 25) & (thousands < 35), 'season'] = 4
    df.loc[(thousands > 35), 'season'] = 5


# def get_markings_count_per_obsid(obsid):
#     for kind in ['fan', 'blotches']:
#
