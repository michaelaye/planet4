from scipy.stats import circmean


def get_average_object(df, kind):
    "Create the average object out of a cluster of data."
    # first filter for outliers more than 1 std away
    # for
    # reduced = df[df.apply(lambda x: np.abs(x - x.mean()) / x.std() < 1).all(axis=1)]
    meandata = df.mean()
    # this determines the upper limit for circular mean
    high = 180 if kind == 'blotch' else 360
    avg = circmean(df.angle, high=high)
    meandata.angle = avg
    return meandata
