import pandas as pd
import numpy as np
import os

size_of_unique = lambda x: x.unique().size
data_root = '/Users/maye/data/marszoo'
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

def get_and_save_done(df, limit=30):
    counts = classification_counts_per_image(df)
    ids_done = counts[counts >= limit].index
    df[df.image_id.isin(ids_done)].to_hdf(done_path, 'df')
    
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


###
### general database stuff
###

def get_current_database(with_tutorial=False):
    if with_tutorial:
        df = pd.read_hdf('/Users/maye/data/marszoo/2014-02-02_planet_four_classifications.h5',
                     'df')
    else:
        df = pd.read_hdf('/Users/maye/data/marszoo/2014-02-02_tutorial_id_removed.h5',
                            'df')
    define_season_column(df)
    return df


def get_current_done():
    return pd.read_hdf(done_path, 'df')

def clean_and_save_database(df):
    df = df[df.image_id != 'APF0000x3t']
    define_season_column(df)
    df.loc[:, 'marking'][df.marking.isnull()] = 'None'
    df.to_hdf('Users/maye/data/marszoo/current_cleaned.h5', 'df')


def get_current_cleaned():
    """docstring for get_current_cleaned"""
    return pd.read_hdf('/Users/maye/data/marszoo/current_cleaned.h5', 'df')
