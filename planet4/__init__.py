import importlib
import logging
from pathlib import Path

__version__ = '0.7.3'

importlib.reload(logging)
logpath = Path.home() / 'p4reduction.log'
logging.basicConfig(filename=str(logpath), filemode='w',
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
