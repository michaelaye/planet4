"""
This module contains the functions required to ground project the PlanetFour data.
Pipeline was initially developed by Meg Schwamb and her student.
Adapted to provide more functionality and code clarity by K.-Michael Aye.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvl

from hirise_tools.products import RED_PRODUCT_ID
from planet4 import io
from pysis import CubeFile
from pysis.exceptions import ProcessError
logger = logging.getLogger(__name__)



try:
    from pysis.isis import (campt, cubenorm, getkey, handmos, hi2isis, histitch,
                        spiceinit)
except ImportError:
    logger.warning("ISIS commands not found.")

img_x_size = 840
img_y_size = 648


def nocal_hi(source_product):
    """Import HiRISE product into ISIS and spice-init it.

    Parameters
    ----------
    source_product : hirise_tools.SOURCE_PRODUCT_ID
        Class object managing the precise filenames and locations for HiRISE source products
    """
    logger.info("hi2isis and spiceinit for %s", source_product)
    img_name = source_product.local_path
    cub_name = source_product.local_cube
    try:
        hi2isis(from_=str(img_name), to=str(cub_name))
        spiceinit(from_=str(cub_name))
    except ProcessError as e:
        logger.error("Error in nocal_hi. STDOUT: %s", e.stdout)
        logger.error("STDERR: %s", e.stderr)
        return


def stitch_cubenorm(spid1, spid2):
    "Stitch together the 2 CCD chip images and do a cubenorm."
    logger.info("Stitch/cubenorm %s and %s", spid1, spid2)
    cub = spid1.local_cube.with_name(spid1.stitched_cube_name)
    norm = cub.with_suffix('.norm.cub')
    try:
        histitch(from1=str(spid1.local_cube), from2=str(spid2.local_cube),
                 to=cub)
        cubenorm(from_=cub, to=norm)
    except ProcessError as e:
        print(e.stdout)
        print(e.stderr)
        sys.exit()
    for spid in [spid1, spid2]:
        spid.local_cube.unlink()
    cub.unlink()
    return norm


def get_RED45_mosaic_inputs(obsid, saveroot):
    """Create list with filenames for RED4 and RED5 CCD chips 0 and 1, respectively.

    Parameters
    ----------
    obsid : str
        HiRISE observation id, e.g. ESP_011350_0945
    saveroot : str, pathlib.Path
        Path to where the data is stored

    Example
    -------
    ESP_011350_0945 returns a list of hirise_tools.RED_PRODUCT_ID objects, that represent
    themselves in the notebook as:
    [RED_PRODUCT_ID: ESP_011350_0945_RED4_0, .... RED4_1, .... RED5_0, .... RED5_1]

    Returns
    -------
    list
        List of 4 hirise_tools.RED_PRODUCT_IDs
    """
    inputs = []
    for channel in [4, 5]:
        for chip in [0, 1]:
            inputs.append(RED_PRODUCT_ID(obsid, channel, chip, saveroot=saveroot))
    return inputs


def create_RED45_mosaic(obsid, overwrite=False):
    gp_root = io.get_ground_projection_root()

    logger.info('Processing the EDR data associated with ' + obsid)

    mos_path = gp_root / obsid / f'{obsid}_mosaic_RED45.cub'

    # bail out if exists:
    if mos_path.exists() and not overwrite:
        print(f'{mos_path} already exists and I am not allowed to overwrite.')
        return obsid, True

    products = get_RED45_mosaic_inputs(obsid, gp_root)

    for prod in products:
        prod.download()
        nocal_hi(prod)

    norm_paths = []
    for channel_products in [products[:2], products[2:]]:
        norm_paths.append(stitch_cubenorm(*channel_products))

    # handmos part
    norm4, norm5 = norm_paths
    im0 = CubeFile.open(str(norm4))  # use CubeFile to get lines and samples
    # get binning mode from label
    bin_ = int(getkey(from_=str(norm4), objname="isiscube", grpname="instrument",
                      keyword="summing"))

    # because there is a gap btw RED4 & 5, nsamples need to first make space
    # for 2 cubs then cut some overlap pixels
    try:
        handmos(from_=str(norm4), mosaic=str(mos_path), nbands=1, outline=1, outband=1,
                create='Y', outsample=1, nsamples=im0.samples * 2 - 48 // bin_,
                nlines=im0.lines)
    except ProcessError as e:
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)

    im0 = CubeFile.open(str(norm5))  # use CubeFile to get lines and samples

    # deal with the overlap gap between RED4 & 5:
    handmos(from_=str(norm5), mosaic=str(mos_path), outline=1, outband=1, create='N',
            outsample=im0.samples - 48 // bin_ + 1)
    for norm in [norm4, norm5]:
        norm.unlink()
    return obsid, True


def cleanup(data_dir, img):
    # do some cleanup removing temporary files
    # removing ISIS cubes made during processing that aren't needed
    fs = data_dir.glob(f'{img}_RED*.cub')

    print(fs)
    for p in fs:
        p.unlink()

    # removing the normalized files

    fs = data_dir.glob(f'{img}_RED*.norm.cub')
    for p in fs:
        p.unlink()

    # remove the raw EDR data

    fs = data_dir.glob(f'{img}_RED*.IMG')
    for p in fs:
        p.unlink()


def get_campt_label(frompath, sample, line):
    try:
        group = pvl.load(campt(from_=str(frompath),
                               sample=sample,
                               line=line)).get('GroundPoint')
    except ProcessError as e:
        print(e.stdout)
        print(e.stderr)
        raise e
    else:
        return group


def do_campt(mosaicname, savepath, temppath):
    print("Calling do_campt")
    try:
        campt(from_=mosaicname, to=savepath, format='flat', append='no',
              coordlist=temppath, coordtype='image')
    except ProcessError as e:
        print(e.stderr)
        return mosaicname, False


def xy_to_hirise(x, y, xtile, ytile):
    "Convert pixels from tile (x,y) to hirise pixels."
    x_offset = img_x_size - 100
    y_offset = img_y_size - 100
    x_HiRISE = x + ((x_offset) * (xtile - 1))  # **formula
    y_HiRISE = y + ((y_offset) * (ytile - 1))  # **formula
    return x_HiRISE, y_HiRISE


def p4pix_to_hirise_pix(p4pix, tile, x_or_y):
    """This convert either x or y coordinate of a planet4 pixel to Hirise coordinate.

    Parameters
    ----------
    p4pix : int
        Coordinate value for either x or y dimension of P4 pixel
    tile : int
        x or y tile coordinate of PlanetFour
    x_or_y : {'x','y'}
        Switch between different coordinate transformations
    """
    offset = dict(x=740, y=548)  # image width/height - 100
    return p4pix + offset[x_or_y] * (np.array(tile) - 1)

def p4tile_center_to_hirise_pix(tile, x_or_y):
    p4pix = dict(x=420, y=324)  # half image sizes
    return p4pix_to_hirise_pix(p4pix[x_or_y], tile, x_or_y)

def tilecenter_to_hirise(x_tile, y_tile=None):
    "get HiRISE pixels for tile center"
    if y_tile is None:
        x_tile, y_tile = x_tile
    return xy_to_hirise(img_x_size/2, img_y_size/2,
                        x_tile, y_tile)


class TileCalculator(object):
    def __init__(self, cubepath, read_data=True):
        self.cubepath = Path(cubepath)
        db = io.DBManager()
        if read_data:
            self.data = db.get_image_name_markings(self.img_name)

    @property
    def img_name(self):
        s = Path(self.cubepath).stem
        return s[:15]

    @property
    def x_tile_max(self):
        return self.data.x_tile.max()

    @property
    def y_tile_max(self):
        return self.data.y_tile.max()

    @property
    def campt_results_path(self):
        savename = f"{self.img_name}_campt_out.csv"
        return self.cubepath.parent / savename

    def get_xy_tiles(self):
        return np.mgrid[1:self.x_tile_max + 1,
                        1:self.y_tile_max + 1]

    def get_campt_input_coords(self):
        xtiles, ytiles = self.get_xy_tiles()
        df = pd.DataFrame(dict(x_tile=xtiles.ravel(), y_tile=ytiles.ravel()))
        df['x_hirise'] = p4tile_center_to_hirise_pix(xtiles.ravel(), 'x')
        df['y_hirise'] = p4tile_center_to_hirise_pix(ytiles.ravel(), 'y')
        return df

    @property
    def temppath(self):
        return self.cubepath.with_suffix('.tocampt')

    @property
    def final_path(self):
        final_fname = f"{self.img_name}_tile_coords.csv"
        final_path = self.cubepath.parent / final_fname
        return final_path

    @property
    def tile_coords_df(self):
        df = pd.read_csv(self.final_path)
        df['obsid'] = self.img_name
        return df

    def calc_tile_coords(self):
        df = self.get_campt_input_coords()
        df[['x_hirise', 'y_hirise']].to_csv(self.temppath, header=False, index=False)
        do_campt(self.cubepath, self.campt_results_path, self.temppath)
        results = pd.read_csv(self.campt_results_path)
        subdf = results[['Sample', 'Line',
                         'PlanetocentricLatitude',
                         'PlanetographicLatitude',
                         'PositiveEast360Longitude',
                         'BodyFixedCoordinateX',
                         'BodyFixedCoordinateY',
                         'BodyFixedCoordinateZ']]
        joined = df.merge(subdf, left_on=['x_hirise', 'y_hirise'],
                          right_on=['Sample', 'Line'])

        # now correlate tiles with image_id
        subset = self.data[['image_id', 'x_tile', 'y_tile']]
        # # this subset is not unique because it comes from marking data,
        # # there are many markings per tile, but i only need one line per tiles
        subset = subset.drop_duplicates()
        # df.merge will find the columns with same names for merging
        finaldf = joined.merge(subset)
        finaldf.to_csv(self.final_path, index=False)
        print("Created", self.final_path)
