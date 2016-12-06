import glob
import os
import sys

from pysis import CubeFile
from pysis.exceptions import ProcessError
from pysis.isis import cubenorm, getkey, handmos, hi2isis, histitch, spiceinit
from pysis.util import file_variations


# case 2: turn IMG to cub, give information of it (spiceinit), don't calibrate it
def nocal_hi(img_name):
    (img_name, cub_name) = file_variations(img_name, ['.IMG', '.cub'])
    # something wrong here, if just put one extension in it, the name will contain brankets
    try:
        hi2isis(from_=img_name, to=cub_name)
        spiceinit(from_=cub_name)
    except ProcessError as e:
        print(e.stdout)
        print(e.stderr)
        sys.exit()


# stitch _0 and _1 together and normalize it
def stit_norm_hi(img1, img2):
    cub = img1[0:20]+'.cub'
    (cub, norm) = file_variations(cub, ['.cub', '.norm.cub'])   # same here
    try:
        histitch(from1=img1, from2=img2, to=cub)
        cubenorm(from_=cub, to=norm)
    except ProcessError as e:
        print(e.stdout)
        print(e.stderr)
        sys.exit()


def hi2mos(nm):
    os.chdir('/Users/klay6683/data/planet4/season2_3_EDRs')
    print('Processing the EDR data associated with '+nm)

    mos_name = 'redMosaic'+nm+'.cub'
#     status = os.path.isfile(mos_name)
    status = False
    if status is True:
        print('skip processing '+nm+'...')
        return nm, False
    else:
        nm = nm+'_RED'
        channel = [4, 5]
        ccd = [0, 1]

        for c in channel:
            for chip in ccd:
                nocal_hi(nm+str(c)+'_'+str(chip)+'.IMG')

            stit_norm_hi(nm+str(c)+'_0.cub', nm+str(c)+'_1.cub')

        # handmos part
        im0 = CubeFile.open(nm+'4.norm.cub')  # use CubeFile to get lines and samples
        # use linux commands to get binning mode
        bin = int(getkey(from_=nm+'4.norm.cub', objname="isiscube", grpname="instrument",
                  keyword="summing"))

        # because there is a gap btw RED4 & 5, nsamples need to first make space
        # for 2 cubs then cut some overlap pixels
        try:
            handmos(from_=nm+'4.norm.cub', mosaic=mos_name, nbands=1, outline=1, outband=1,
                    create='Y', outsample=1, nsamples=im0.samples*2-48//bin,
                    nlines=im0.lines)
        except ProcessError as e:
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
        im0 = CubeFile.open(nm+'5.norm.cub')  # use CubeFile to get lines and samples

        # deal with the overlap gap between RED4 & 5:
        handmos(from_=nm+'5.norm.cub', mosaic=mos_name, outline=1, outband=1, create='N',
                outsample=im0.samples-48//bin+1)
        return nm, True


def cleanup(data_dir, img):
    # do some cleanup removing temporary files
    # removing ISIS cubes made during processing that aren't needed
    fs = glob.glob(data_dir+'/'+img+'_RED*.cub')

    print(fs)
    for f in fs:
        os.remove(f)

    # removing the normalized files

    fs = glob.glob(data_dir+'/'+img+'_RED*.norm.cub')
    for f in fs:
        os.remove(f)

    # remove the raw EDR data

    fs = glob.glob(data_dir+'/'+img+'_RED*.IMG')
    for f in fs:
        os.remove(f)
