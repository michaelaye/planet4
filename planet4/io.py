import configparser
import datetime as dt
import logging
import os
import shutil
import warnings
from pathlib import Path
from urllib.error import URLError

import dask.dataframe as dd
import matplotlib.image as mplimg
import pandas as pd
import pkg_resources as pr

from . import stats
from .exceptions import NoFilesFoundError

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

pkg_name = __name__.split(".")[0]

configpath = Path.home() / ".{}.ini".format(pkg_name)

LOGGER = logging.getLogger(__name__)


def get_config():
    """Read the configfile and return config dict.

    Returns
    -------
    dict
        Dictionary with the content of the configpath file.
    """
    if not configpath.exists():
        raise IOError("Config file {} not found.".format(str(configpath)))
    else:
        config = configparser.ConfigParser()
        config.read(str(configpath))
        return config


def set_database_path(dbfolder):
    """Use to write the database path into the config.

    Parameters
    ----------
    dbfolder : str or pathlib.Path
        Path to where planet4 will store clustering results by default.
    """
    try:
        d = get_config()
    except IOError:
        d = configparser.ConfigParser()
        d["planet4_db"] = {}
    d["planet4_db"]["path"] = dbfolder
    with configpath.open("w") as f:
        d.write(f)
    print("Saved database path into {}.".format(configpath))


def get_data_root():
    d = get_config()
    data_root = Path(d["planet4_db"]["path"]).expanduser()
    data_root.mkdir(exist_ok=True, parents=True)
    return data_root


def get_ground_projection_root():
    d = get_config()
    try:
        gp_root = Path(d["ground_projection"]["path"])
    except KeyError:
        warnings.warn(
            "ground_projection_root not set in config.\n"
            "Read/Write of projected data is disabled."
        )
        gp_root = None
    else:
        gp_root.mkdir(exist_ok=True)
    return gp_root


if not configpath.exists():
    print("No configuration file {} found.\n".format(configpath))
    savepath = input("Please provide the path where you want to store planet4 results:")
    set_database_path(savepath)
else:
    data_root = get_data_root()


def dropbox():
    return Path.home() / "Dropbox"


def p4data():
    return dropbox() / "data" / "planet4"


def analysis_folder():
    name = "p4_analysis"
    if p4data().exists():
        path = p4data() / name
    else:
        path = dropbox() / name
    return path


def check_and_pad_id(imgid):
    "Does NOT work with pd.Series item."
    if imgid is None:
        return None
    imgid_template = "APF0000000"
    if len(imgid) < len(imgid_template):
        imgid = imgid_template[: -len(imgid)] + imgid
    return imgid


def get_subframe(url):
    """Download image if not there yet and return numpy array.

    Takes a data record (called 'line'), picks out the image_url.
    First checks if the name of that image is already stored in
    the image path. If not, it grabs it from the server.
    Then uses matplotlib.image to read the image into a numpy-array
    and finally returns it.
    """
    targetpath = data_root / "images" / os.path.basename(url)
    targetpath.parent.mkdir(exist_ok=True)
    if not targetpath.exists():
        LOGGER.info("Did not find image in cache. Downloading ...")
        try:
            path = urlretrieve(url)[0]
        except URLError:
            msg = "Cannot receive subframe image. No internet?"
            LOGGER.error(msg)
            return None
        LOGGER.debug("Done.")
        shutil.move(path, str(targetpath))
    else:
        LOGGER.debug("Found image in cache.")
    im = mplimg.imread(targetpath)
    return im


class P4DBName(object):
    def __init__(self, fname):
        self.p = Path(fname)
        date = str(self.name)[:10]
        self.date = dt.datetime(*[int(i) for i in date.split("-")])

    def __getattr__(self, name):
        "looking up things in the Path object if not in `self`."
        return getattr(self.p, name)


def get_latest_file(filenames):
    fnames = list(filenames)
    if len(fnames) == 0:
        raise NoFilesFoundError
    retval = P4DBName(fnames[0])
    dtnow = retval.date
    for fname in fnames[1:]:
        dt_to_check = P4DBName(fname).date
        if dt_to_check > dtnow:
            dtnow = dt_to_check
            retval = P4DBName(fname)
    return retval.p


def get_latest_cleaned_db(datadir=None):
    datadir = data_root if datadir is None else Path(datadir)
    basestr = "201*_queryable_cleaned*"
    conf = get_config()
    dbformat = conf['options']['dbformat']
    parquet = True if dbformat == 'parquet' else False
    if not parquet:
        files = list(datadir.glob(basestr + ".h5"))
    else:
        files = list(datadir.glob(basestr + ".parquet"))
    if len(files) == 0:
        LOGGER.error("No files found. Searching in %s", str(datadir))
        raise NoFilesFoundError(f"No files found. Searching in {str(datadir)}")
    return get_latest_file(files)


def get_latest_season23_dbase(datadir=None):
    if datadir is None:
        datadir = data_root
    h5files = list(datadir.glob("201*_queryable_cleaned_seasons2and3.h5"))
    return get_latest_file(h5files)


def get_test_database():
    fname = pr.resource_filename("planet4", "data/test_db.csv")
    return pd.read_csv(fname)


def get_latest_tutorial_data(datadir=None):
    if datadir is None:
        datadir = data_root

    tut_files = datadir.glob("/*_tutorials.h5")
    tut_files = [i for i in tut_files if i.parent[:4].isdigit()]
    if not tut_files:
        raise NoFilesFoundError
    return pd.read_hdf(str(get_latest_file(tut_files)), "df")


def common_gold_ids():
    # read the common gold_ids to check
    with open("../data/gold_standard_commons.txt") as f:
        gold_ids = f.read()
    gold_ids = gold_ids.split("\n")
    del gold_ids[-1]  # last one is empty
    return gold_ids


def get_image_names_from_db(dbfname):
    """Return arrary of HiRISE image_names from database file.

    Parameters
    ----------
    dbfname : pathlib.Path or str
        Path to database file to be used.

    Returns
    -------
    numpy.ndarray
        Array of unique image names.
    """
    path = Path(dbfname)
    if path.suffix in [".hdf", ".h5"]:
        return dd.read_hdf(str(path), "df").image_name.unique()
    elif path.suffix == ".csv":
        return dd.read_csv(path).image_name.unique()
    elif path.suffix == ".parquet":
        return dd.read_parquet(path).image_name.unique()
    else:
        raise UserWarning(f"Unknown suffix: {path.suffix}")


def get_latest_marked():
    return pd.read_hdf(str(get_latest_cleaned_db()), "df", where="marking!=None")


def get_image_id_from_fname(fname):
    "Return image_id from beginning of Path(fname).name"
    fname = Path(fname)
    name = fname.name
    return name.split("_")[0]


def get_image_ids_in_folder(folder, extension=".csv"):
    fnames = Path(folder).glob("*" + extension)
    return [get_image_id_from_fname(i) for i in fnames]


class PathManager(object):

    """Manage file paths and folders related to the analysis pipeline.

    Level definitions:
    * L0 : Raw output of Planet Four
    * L1A : Clustering of Blotches and Fans on their own
    * L1B : Clustered blotches and fans combined into final fans, final blotches, and fnotches that
    need to have a cut applied for the decision between fans or blotches.
    * L1C : Derived database where a cut has been applied for fnotches to become either fan or
    blotch.

    Parameters
    ----------
    id_ : str, optional
        The data item id that is used to determine sub-paths. Can be set after
        init.
    datapath : str or pathlib.Path, optional
        the base path from where to manage all derived paths. No default assumed
        to prevent errors.
    suffix : {'.hdf', '.h5', '.csv'}
        The suffix that controls the reader function to be used.
    obsid : str, optional
        HiRISE obsid (i.e. P4 image_name), added as a folder inside path.
        Can be set after init.
    extra_path : str, pathlib.Path, optional
        Any extra path element that needs to be added to the standard path.

    Attributes
    ----------
    cut_dir : pathlib.Path
        Defined in `get_cut_folder`.
    """

    def __init__(
        self,
        id_="",
        datapath="clustering",
        suffix=".csv",
        obsid="",
        cut=0.5,
        extra_path="",
    ):
        self.id = id_
        self.cut = cut
        self._obsid = obsid
        self.extra_path = extra_path

        if datapath is None:
            # take default path if none given
            self._datapath = Path(data_root) / "clustering"
        elif Path(datapath).is_absolute():
            # if given datapath is absolute, take only that:
            self._datapath = Path(datapath)
        else:
            # if it is relative, add it to data_root
            self._datapath = Path(data_root) / datapath
        self.suffix = suffix

        # point reader to correct function depending on required suffix
        if suffix in [".hdf", ".h5"]:
            self.reader = pd.read_hdf
        elif suffix == ".csv":
            self.reader = pd.read_csv

        # making sure to warn the user here if the data isn't where it's expected to be
        if id_ != "":
            if not self.path_so_far.exists():
                raise FileNotFoundError(f"{self.path_so_far} does not exist.")

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        if value is not None:
            self._id = check_and_pad_id(value)

    @property
    def clustering_logfile(self):
        return self.fanfile.parent / "clustering_settings.yaml"

    @property
    def obsid(self):
        if self._obsid == "":
            if self.id != "":
                LOGGER.debug("Entering obsid search for known image_id.")
                db = DBManager()
                df = db.read(columns=['image_id', 'image_name'])
                obsid = df.query("image_id==@self.id").image_name.iloc[0]
                LOGGER.debug("obsid found: %s", obsid)
                self._obsid = obsid
        return self._obsid

    @obsid.setter
    def obsid(self, value):
        self._obsid = value

    @property
    def obsid_results_savefolder(self):
        subfolder = "p4_catalog" if self.datapath is None else self.datapath
        savefolder = analysis_folder() / subfolder
        savefolder.mkdir(exist_ok=True, parents=True)
        return savefolder

    @property
    def obsid_final_fans_path(self):
        return self.obsid_results_savefolder / f"{self.obsid}_fans.csv"

    @property
    def obsid_final_blotches_path(self):
        return self.obsid_results_savefolder / f"{self.obsid}_blotches.csv"

    @property
    def datapath(self):
        return self._datapath

    @property
    def path_so_far(self):
        p = self.datapath
        p /= self.extra_path
        p /= self.obsid
        return p

    @property
    def L1A_folder(self):
        "Subfolder name for the clustered data before fnotching."
        return "L1A"

    @property
    def L1B_folder(self):
        "Subfolder name for the fnotched data, before cut is applied."
        return "L1B"

    @property
    def L1C_folder(self):
        "subfolder name for the final catalog after applying `cut`."
        return "L1C_cut_{:.1f}".format(self.cut)

    def get_path(self, marking, specific=""):
        p = self.path_so_far
        # now add the image_id
        try:
            p /= self.id
        except TypeError:
            logging.warning("self.id not set. Storing in obsid level.")

        id_ = self.id if self.id != "" else self.obsid

        # add the specific sub folder
        p /= specific

        if specific != "":
            p /= f"{id_}_{specific}_{marking}{self.suffix}"
        else:
            # prepend the data level to file name if given.
            p /= f"{id_}_{marking}{self.suffix}"
        return p

    def get_obsid_paths(self, level):
        """get all existing paths for a given data level.

        Parameters
        ----------
        level : {'L1A', 'L1B', 'L1C'}
        """
        folder = self.path_so_far
        # cast to upper case for the lazy... ;)
        level = level.upper()
        image_id_paths = [item for item in folder.glob("*") if item.is_dir()]
        bucket = []
        for p in image_id_paths:
            try:
                bucket.append(next(p.glob(f"{level}*")))
            except StopIteration:
                continue
        return bucket

    def get_df(self, fpath):
        return self.reader(str(fpath))

    @property
    def fanfile(self):
        return self.get_path("fans", self.L1A_folder)

    @property
    def fandf(self):
        return self.get_df(self.fanfile)

    @property
    def reduced_fanfile(self):
        return self.get_path("fans", self.L1B_folder)

    @property
    def reduced_fandf(self):
        return self.get_df(self.reduced_fanfile)

    @property
    def final_fanfile(self):
        return self.get_path("fans", self.L1C_folder)

    @property
    def final_fandf(self):
        return self.get_df(self.final_fanfile)

    @property
    def blotchfile(self):
        return self.get_path("blotches", self.L1A_folder)

    @property
    def blotchdf(self):
        return self.get_df(self.blotchfile)

    @property
    def reduced_blotchfile(self):
        return self.get_path("blotches", self.L1B_folder)

    @property
    def reduced_blotchdf(self):
        return self.get_df(self.reduced_blotchfile)

    @property
    def final_blotchfile(self):
        return self.get_path("blotches", self.L1C_folder)

    @property
    def final_blotchdf(self):
        return self.get_df(self.final_blotchfile)

    @property
    def fnotchfile(self):
        return self.get_path("fnotches", self.L1B_folder)

    @property
    def fnotchdf(self):
        # the fnotchfile has an index, so i need to read that here:
        return pd.read_csv(self.fnotchfile, index_col=0)


class DBManager(object):

    """Access class for database activities.

    Provides easy access to often used data items.

    Parameters
    ----------
    dbname : str, optional
        Path to database file to be used. Default: use get_latest_cleaned_db() to
        find it.

    Attributes
    ----------
    image_names
    image_ids
    n_image_ids
    n_image_names
    obsids : Alias to image_ids
    season2and3_image_names

    """

    def __init__(self, dbname=None):
        """Initialize DBManager class.

        Parameters
        ----------
        dbname : <str>
            Filename of database file to use. Default: Latest produced full
            database.
        """
        if dbname is None:
            self.dbname = str(get_latest_cleaned_db())
        else:
            self.dbname = str(dbname)

    def __repr__(self):
        s = "Database root: {}\n".format(Path(self.dbname).parent)
        s += "Database name: {}\n".format(Path(self.dbname).name)
        return s

    def read(self, **kwargs):
        p = Path(self.dbname)
        if p.suffix.endswith("hdf"):
            return pd.read_hdf(p, **kwargs)
        elif p.suffix.endswith("parquet"):
            where = kwargs.pop('where', None)
            if where is not None:
                obsid = where.split('=')[-1].strip()
                folder = p.parent / p.stem
                fname = (folder / obsid).with_suffix('.parquet')
                return pd.read_parquet(fname)
            else:
                return pd.read_parquet(p, **kwargs)
        elif p.suffix.endswith("csv"):
            return pd.read_csv(p, **kwargs)

    @property
    def orig_csv(self):
        p = Path(self.dbname)
        return p.parent / (p.name[:38] + ".csv")

    def get_obsid_for_tile_id(self, tile_id):
        df = self.read(columns=['image_id', 'image_name'])
        obsid = df.query("image_id==@tile_id").image_name.iloc[0]
        return obsid

    def set_latest_with_dupes_db(self, datadir=None):
        datadir = data_root if datadir is None else Path(datadir)
        h5files = datadir.glob("201*_queryable.h5")
        dbname = get_latest_file(h5files)
        print("Setting {} as dbname.".format(dbname.name))
        self.dbname = str(dbname)

    @property
    def image_names(self):
        """Return list of unique obsids used in database.

        See also
        --------
        get_image_names_from_db
        """
        return get_image_names_from_db(self.dbname)

    @property
    def image_ids(self):
        "Return list of unique image_ids in database."
        with pd.HDFStore(self.dbname) as store:
            return store.select_column("df", "image_id").unique()

    @property
    def n_image_ids(self):
        return len(self.image_ids)

    @property
    def n_image_names(self):
        return len(self.image_names)

    @property
    def obsids(self):
        "Alias to self.image_names."
        return self.image_names

    def get_all(self, datadir=None):
        return self.read()

    def get_obsid_markings(self, obsid):
        "Return marking data for given HiRISE obsid."
        return self.read(where="image_name=" + obsid)

    def get_image_name_markings(self, image_name):
        "Alias for get_obsid_markings."
        return self.get_obsid_markings(image_name)

    def get_image_id_markings(self, image_id, obsid=None):
        "Return marking data for one Planet4 image_id"
        image_id = check_and_pad_id(image_id)
        if obsid is None:
            obsid = self.get_obsid_for_tile_id(image_id)
        data = self.get_image_name_markings(obsid)
        return data.query("image_id==@image_id")

    def get_data_for_obsids(self, obsids):
        bucket = []
        for obsid in obsids:
            bucket.append(self.get_obsid_markings(obsid))
        return pd.concat(bucket, ignore_index=True)

    def get_classification_id_data(self, class_id):
        "Return data for one classification_id"
        return self.read(where="classification_id=='{}'".format(class_id))

    @property
    def season2and3_image_names(self):
        "numpy.array : List of image_names for season 2 and 3."
        image_names = self.image_names
        metadf = pd.DataFrame(
            pd.Series(image_names).astype("str"), columns=["image_name"]
        )
        stats.define_season_column(metadf)
        return metadf[(metadf.season > 1) & (metadf.season < 4)].image_name.unique()

    def get_general_filter(self, f):
        return self.read(where=f)
