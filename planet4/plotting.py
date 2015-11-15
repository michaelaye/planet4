import pandas as pd
from pathlib import Path

from . import markings


class ResultManager:

    def __init__(self, id_, datapath, suffix='.hdf'):
        self.id_ = id_
        self.inpath = Path(datapath)
        self.suffix = suffix
        self.unfnotched_path = self.inpath.with_name(self.inpath.stem+'_unfnotched')

    def fanfile(self, unfnotched=False):
        if unfnotched:
            path = self.unfnotched_path
        else:
            path = self.inpath
        return path / (self.id_ + '_fans' + self.suffix)

    def fandf(self, unfnotched=False):
        return pd.read_hdf(str(self.fanfile(unfnotched)))

    def fans(self, unfnotched=False):
        fandf = self.fandf(unfnotched)
        return [markings.Fan(i) for _, i in fandf.iterrows()]

    def blotchfile(self, unfnotched=False):
        if unfnotched:
            path = self.unfnotched_path
        else:
            path = self.inpath
        return path / (self.id_ + '_blotches' + self.suffix)

    def blotchdf(self, unfnotched=False):
        return pd.read_hdf(str(self.blotchfile(unfnotched)))

    def blotches(self, unfnotched=False):
        blotchdf = self.blotchdf(unfnotched)
        return [markings.Blotch(i) for _, i in blotchdf.iterrows()]

    @property
    def fnotchfile(self):
        return self.inpath / (self.id_ + '_fnotches' + self.suffix)

    @property
    def fnotchdf(self):
        return pd.read_hdf(str(self.fnotchfile))
