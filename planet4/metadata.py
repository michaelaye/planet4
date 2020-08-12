"""Provides tools to create the required metadata for the analysis of P4 data.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pvl

from pyrise import downloads, labels, products

from . import io

logger = logging.getLogger(__name__)

try:
    from osgeo import gdal
except ImportError:
    logger.warn("GDAL not available.")

REGIONS = ['Inca', 'Ithaca', 'Giza', 'Manhattan2', 'Starburst', 'Oswego_Edge',
           'Maccelsfield', 'BuenosAires', 'Potsdam']


# This function is currently not in use due to some issues:
# Lack of precision, and missing cut-off part for P4 data ingestion
def get_fraction_of_black_pixels(savepath):
    "Currently not in use"
    ds = gdal.Open(str(savepath))
    data = ds.ReadAsArray()
    fractions = []
    for band in data:
        nonzeros = band.nonzero()[0]
        fractions.append((band.size - nonzeros.size) / band.size)
    return np.array(fractions).mean()


class MetadataReader:
    proj_folder = io.get_ground_projection_root()

    def __init__(self, obsid):
        self.obsid = obsid
        self.prodid = products.PRODUCT_ID(obsid)
        self.prodid.kind = 'COLOR'
        if not self.labelpath.exists():
            self.download_label()

    def read_edr_index(self):
        edrindex = pd.read_hdf("/Volumes/Data/hirise/EDRCUMINDEX.hdf")
        return edrindex

    @property
    def labelpath(self):
        return downloads.labels_root() / self.prodid.label_fname

    def download_label(self):
        inpath = Path(self.prodid.label_path)
        downloads.download_product(inpath, downloads.labels_root())

    @property
    def label(self):
        return labels.HiRISE_Label(self.labelpath)

    @property
    def campt_out_path(self):
        return self.proj_folder / self.obsid / f"{self.obsid}_campt_out.csv"

    @property
    def campt_out_df(self):
        return pd.read_csv(self.campt_out_path)

    def get_data_dic(self):
        """Defines a dictionary with metadata

        Uses mostly the mosaic label data, but adds the previously
        SPICE-calculated NorthAzimuth angle from `campt` ISIS tool.

        Prerequisite for this call is, that the campt files have all been
        created for the obsids to be done.
        This is usually the case after all tile coordinates have been created
        using projection.TileCalculator.

        FIXME
        """
        # edrindex = pd.read_hdf("/Volumes/Data/hirise/EDRCUMINDEX.hdf")
        # p4_edr = edrindex[edrindex.OBSERVATION_ID.isin(obsids)].query(
        #     'CCD_NAME=="RED4"').drop_duplicates(subset='OBSERVATION_ID')

        # label = self.label
        # labelpath = self.labelpath
        # d = dict(obsid=self.obsid,
        #          l_s=label.l_s, line_samples=label.line_samples,
        #          lines=label.lines, map_scale=label.map_scale)
        # d['north_azimuth'] = self.campt_out_df['NorthAzimuth'].median()
        # return d


def get_north_azimuths_from_SPICE(obsids):
    NAs = []
    for obsid in obsids:
        meta = MetadataReader(obsid)
        NAs.append(meta.campt_out_df['NorthAzimuth'].median())
    return pd.DataFrame(dict(OBSERVATION_ID=obsids, north_azimuth=NAs))

#     savedir = downloads.hirise_dropbox() / 'browse'
#     savepath = savedir / prodid.browse_path.name
#     if not savepath.exists():
#         ht.download_product(prodid.browse_path, savedir)
#     black_fraction = get_fraction_of_black_pixels(savepath)
#     all_area = label.line_samples*label.lines * label.map_scale**2
#     real_area = (1-black_fraction)*all_area
#     d = dict(obsid=obsid, path=labelpath, binning=label.binning,
#              l_s=label.l_s, line_samples=label.line_samples,
#              lines=label.lines, map_scale=label.map_scale)
#     invalids=black_fraction, real_area=real_area)
    # self calculated north azimuths
#     folder = io.get_ground_projection_root()
#     path = folder / obsid / f"{obsid}_campt_out.csv"
#     df = pd.read_csv(path)
#     d['north_azimuth'] = df.NorthAzimuth.median()
#     return d
