"""
This module contains the functions required to ground project the PlanetFour data.
Pipeline was initially developed by Meg Schwamb and her student.
Adapted to provide more functionality and code clarity by K.-Michael Aye.
"""
import sys
from pathlib import Path

from hirise_tools.products import RED_PRODUCT_ID
from planet4 import io
from pysis import CubeFile
from pysis.exceptions import ProcessError
from pysis.isis import cubenorm, getkey, handmos, hi2isis, histitch, spiceinit
from pysis.util import file_variations
import logging

logger = logging.getLogger(__name__)


def nocal_hi(source_product):
    """Import HiRISE product into ISIS and spice-init it.

    Parameters
    ----------
    source_product : hirise_tools.SOURCE_PRODUCT_ID
        Class object managing the precise filenames and locations for HiRISE source products
    """
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
    cub = spid1.local_cube.with_name(spid1.stitched_name)
    norm, = file_variations(cub, ['.norm.cub'])
    try:
        histitch(from1=str(spid1.local_cube), from2=str(spid2.local_cube),
                 to=cub)
        cubenorm(from_=cub, to=norm)
    except ProcessError as e:
        print(e.stdout)
        print(e.stderr)
        sys.exit()


def get_RED45_mosaic_inputs(obsid, saveroot):
    inputs = []
    for channel in [4, 5]:
        for chip in [0, 1]:
            inputs.append(RED_PRODUCT_ID(obsid, channel, chip, saveroot=saveroot))
    return inputs


def hi2mos(obsid, overwrite=False):
    gp_root = io.get_ground_projection_root()

    print('Processing the EDR data associated with ' + obsid)

    mos_path = gp_root / obsid / f'{obsid}_mosaic_RED45.cub'

    # bail out if exists:
    if mos_path.exists() and not overwrite:
        print(f'{mos_path} already exists and I am not allowed to overwrite.')
        return obsid, False

    products = get_RED45_mosaic_inputs(obsid, gp_root)

    for prod in products:
        prod.download()
        nocal_hi(prod)

    for channel_products in [products[:2], products[2:]]:
        stitch_cubenorm(*channel_products)

    return
    # handmos part
    im0 = CubeFile.open(f'{name}4.norm.cub')  # use CubeFile to get lines and samples
    # use linux commands to get binning mode
    bin_ = int(getkey(from_=name + '4.norm.cub', objname="isiscube", grpname="instrument",
               keyword="summing"))

    # because there is a gap btw RED4 & 5, nsamples need to first make space
    # for 2 cubs then cut some overlap pixels
    try:
        handmos(from_=f'{name}4.norm.cub', mosaic=mos_path, nbands=1, outline=1, outband=1,
                create='Y', outsample=1, nsamples=im0.samples*2 - 48//bin_,
                nlines=im0.lines)
    except ProcessError as e:
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)

    im0 = CubeFile.open(f'{name}5.norm.cub')  # use CubeFile to get lines and samples

    # deal with the overlap gap between RED4 & 5:
    handmos(from_=f'{name}5.norm.cub', mosaic=mos_path, outline=1, outband=1, create='N',
            outsample=im0.samples - 48//bin_ + 1)
    return name, True


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
