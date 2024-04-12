from math import tau
from pathlib import Path

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

mars_years = {
    28: "2006-01-23",
    29: "2007-12-10",
    30: "2009-10-27",
    31: "2011-09-15",
    32: "2013-08-01",
    33: "2015-06-19",
    34: "2017-05-05",
    35: "2019-03-24",
    36: "2021-02-07",
    37: "2022-12-26",
    38: "2024-11-24",
    39: "2026-09-30",
    40: "2028-08-17",
}


def get_fan_and_blotch_nunique_cids(data):
    f1 = data.marking == "fan"
    f2 = data.marking == "blotch"
    return data[f1 | f2].classification_id.nunique()


def get_fb_to_all_ratio(data):
    n_classifications = data.classification_id.nunique()
    n_class_fb = get_fan_and_blotch_nunique_cids(data)
    ratio = n_class_fb / n_classifications
    return ratio


def size_of_unique(x):
    return x.unique().size


def classification_counts_per_user(df):
    res = (
        df.classification_id.groupby(df.user_name, sort=False)
        .agg(size_of_unique)
        .sort_values(ascending=False)
    )
    return res


def get_top_ten_users(df):
    users_work = classification_counts_per_user(df)
    return users_work.order(ascending=False)[:10]


def classification_counts_per_image(df):
    """Main function to help defining status of P4"""
    return df.classification_id.groupby(df.image_id, sort=False).agg(size_of_unique)


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
    return df.user_name.groupby(df.image_id, sort=False).agg(size_of_unique)


def get_blotch_area(record):
    if record.marking != "blotch":
        return 0
    else:
        return 0.5 * tau * record.radius_1 * record.radius_2


###
# Season related stuff
###


def unique_image_ids_per_season(df):
    return df.image_id.groupby(df.season, sort=False).agg(size_of_unique)


def define_season_column(df, colname="image_name"):
    """Create new column that indicates the MRO season.

    Seasons 1,2, and 3 are MY28, 29, and 30 respectively.

    Parameters:
    ----------
    df : {pandas.DataFrame}
        Dataframe that should have a column with name `colname` as deciding factor.
    colname : str
        Name of column to be used as HiRISE observation ID.
    """
    thousands = df[colname].str[5:7].astype("int")
    df["season"] = 0
    df.loc[df[colname].str.startswith("PSP"), "season"] = 1
    df.loc[(thousands > 10) & (thousands < 15), "season"] = 2
    df.loc[(thousands > 15) & (thousands < 25), "season"] = 3
    df.loc[(thousands > 25) & (thousands < 35), "season"] = 4
    df.loc[(thousands > 35), "season"] = 5


def define_martian_year(df, time_col_name):
    mars_timestamps = {k: pd.to_datetime(v) for k, v in mars_years.items()}
    df["MY"] = 0
    for yr, t in mars_timestamps.items():
        df.loc[df[time_col_name] > t, "MY"] = yr


def calculate_percent_lost(database_fname):
    ddf = dd.read_parquet(database_fname)
    image_names = ddf.image_name.unique().compute()

    keys = ["image_id", "user_name"]
    cols = ["created_at", "classification_id"]

    times = []
    percent_lost = []
    for image_name in tqdm(image_names):
        example = ddf[ddf.image_name == image_name].compute()
        old = example.shape[0]
        g = example[keys + cols].groupby(keys, sort=False)
        c_ids = g[cols].min().classification_id.values
        new = example[example.classification_id.isin(c_ids)].shape[0]
        percent_lost.append((old - new) / old * 100)
        times.append(example.created_at.mean())

    plt.figure()
    plt.plot(times, percent_lost, "*")
    plt.grid()
    plt.title("Duplicate loss over P4 time.")
    plt.xlabel("Data collection date")
    plt.ylabel("Duplicate data excluded [%]")
