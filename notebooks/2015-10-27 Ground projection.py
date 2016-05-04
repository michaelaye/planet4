
# coding: utf-8

# In[ ]:

from __future__ import print_function, division
from pysis.util import file_variations
import os
from pysis import CubeFile
from pysis.isis import cubenorm, handmos, hi2isis, hical, histitch, spiceinit, getkey
import subprocess as sp
from pysis.exceptions import ProcessError
import sys

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



# In[ ]:

img_names = pd.read_table('/Users/klay6683/Dropbox/data/planet4/season2_3_image_names.txt',
                          header=None, squeeze=True)
img_names.head()


# In[ ]:

cd ~/data/planet4/season2_3_EDRs/


# In[ ]:

from ipyparallel import Client
c = Client()

lbview = c.load_balanced_view()
dview = c.direct_view()


# In[ ]:

get_ipython().run_cell_magic('px', '', 'from __future__ import print_function,division\nfrom pysis.util import file_variations\nimport os\nfrom pysis import CubeFile\nfrom pysis.isis import cubenorm, handmos, hi2isis, hical, histitch, spiceinit, getkey\nimport subprocess as sp\nfrom pysis.exceptions import ProcessError\nimport sys')


# In[ ]:

dview.push({'nocal_hi':nocal_hi,
            'stit_norm_hi':stit_norm_hi})


# In[ ]:

results = lbview.map_async(hi2mos, img_names)


# In[ ]:

from iuvs.multitools import nb_progress_display


# In[ ]:

nb_progress_display(results, img_names)


# In[ ]:

for res in results:
    print(res)


# ## cleanup

# In[ ]:

get_ipython().system('rm -f *RED*.cub')


# # xy2latlon

# In[ ]:

from pysis.isis import campt
from pysis.exceptions import ProcessError
import pvl


# In[ ]:

from pathlib import Path
edrpath = Path('/Users/klay6683/data/planet4/season2_3_EDRs/')
clusterpath = Path('/Users/klay6683/Dropbox/data/planet4/inca_s23_0.5cut_applied/')


# In[ ]:

obsids = get_ipython().getoutput('cat /Users/klay6683/Dropbox/data/planet4/season2_3_image_names.txt')


# In[ ]:

fpaths = [item for obsid in obsids for item in (clusterpath.glob("{}_*.hdf".format(obsid)))]


# In[ ]:

blotch_coords = ['', 'p1', 'p2', 'p3', 'p4']
fan_coords = ['', 'arm1', 'arm2']


# In[ ]:

from ipyparallel import Client
c = Client()

lbview = c.load_balanced_view()
dview = c.direct_view()


# In[ ]:

with dview.sync_imports():
    from pysis.isis import campt
    from pysis.exceptions import ProcessError
    from pathlib import Path
    from ipyparallel import CompositeError


# In[ ]:

def do_campt(mosaicname, savepath, temppath):
    try:
        campt(from_=mosaicname, to=savepath, format='flat', append='no',
              coordlist=temppath, coordtype='image')
    except ProcessError as e:
        print(e.stderr)
        return obsid, False


def process_inpath(inpath, marking, mosaicpath):
    coords_switch = dict(blotches=blotch_coords,
                         fans=fan_coords)
    
    df = pd.read_hdf(str(inpath), 'df')
    for coord in coords_switch[marking]:
        print("Coord", coord)
        if coord == '':
            name = 'base'
            tempcoords = ['x', 'y']
        else:
            name = coord
            tempcoords = [coord + '_x', coord + '_y']
        print("Tempcoords", tempcoords)
        temppath = inpath.with_suffix('.tocampt')
        try:
            df[tempcoords].to_csv(str(temppath), header=False, index=False)
        except KeyError:
            return False
        print("name", name)
        savename = "{stem}_{c}_campt_out.csv".format(stem=inpath.stem, c=name)
        print("savename", savename)
        savepath = clusterpath / savename
        try:
            do_campt(mosaicpath, savepath, temppath)
        except:
            return False
    return True

def xy2latlon(inpath):
    d = dict(inpath=inpath)
    edrpath = Path('/Users/klay6683/data/planet4/season2_3_EDRs/')
    tokens = inpath.stem.split('_')
    obsid = '_'.join(tokens[:3])
    marking = tokens[-1]
    mosaicname = 'redMosaic' + obsid + '.cub'
    mosaicpath = edrpath / mosaicname
    ok = process_inpath(inpath, marking, mosaicpath)
    d['ok'] = ok
    return d


# In[ ]:

dview.push(dict(process_inpath=process_inpath,
                do_campt=do_campt,
                blotch_coords=blotch_coords,
                fan_coords=fan_coords,
                clusterpath=Path('/Users/klay6683/Dropbox/data/planet4/'
                                 'inca_s23_0.5cut_applied/')))


# In[ ]:

xy2latlon(fpaths[1])


# In[ ]:

results = lbview.map_async(xy2latlon, fpaths)


# In[ ]:

from iuvs.multitools import nb_progress_display

nb_progress_display(results, fpaths)


# In[ ]:

res = pd.DataFrame(results.result)


# In[ ]:

res.ok.value_counts()


# In[ ]:

res[res.ok==False].inpath.values


# # Combining campt results

# In[ ]:

p = fpaths[0]


# In[ ]:

class GroundMarking(object):
    def __init__(self, resultfile):
        self.p = Path(resultfile)
        
        # this loop creates filename paths for all coords campt output files
        # and assigns them to object attributes, like
        # self.basefile, self.p1file, etc.
        self.paths = []
        self.mapped_coords = []
        for coord in self.coords:
            path = self.campt_fname(coord)
            setattr(self, coord+'file', path)
            self.paths.append(path)
            self.store_mapped_coords(coord, path)
        self.mapped_coords = pd.concat(self.mapped_coords, axis=1)
        newpath = self.p.with_name(self.p.stem+'_latlons.csv')
        self.mapped_coords.to_csv(str(newpath), index=False)
        self.coordspath = newpath

    def campt_fname(self, coordname):
        return self.p.with_name(self.p.stem + '_{}_campt_out.csv'.format(coordname))
    
    def store_mapped_coords(self, coord, path):
        df = pd.read_csv(path)
        subdf = df[['PlanetographicLatitude',
                    'PositiveEast360Longitude']]
        subdf.columns = [coord+'_lat', coord+'_lon']
        self.mapped_coords.append(subdf)

class GroundBlotch(GroundMarking):
    coords = ['base', 'p1', 'p2', 'p3', 'p4']
    kind = 'blotch'


class GroundFan(GroundMarking):
    coords = ['base', 'arm1', 'arm2']
    kind = 'fan'

    
def get_ground_marking(fname):
    tokens = Path(fname).stem.split('_')
    if tokens[-1] == 'blotches':
        return GroundBlotch(fname)
    else:
        return GroundFan(fname)


# In[ ]:

for path in fpaths:
    print(path.stem)
    get_ground_marking(path)


# In[ ]:



