# Planet Four

[![Join the chat at https://gitter.im/michaelaye/planet4](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/michaelaye/planet4?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Software to support the analyis of planetfour data.

[![Build Status](https://travis-ci.org/michaelaye/planet4.svg?branch=master)](https://travis-ci.org/michaelaye/planet4)
[![Coverage Status](https://coveralls.io/repos/github/michaelaye/planet4/badge.svg?branch=master)](https://coveralls.io/github/michaelaye/planet4?branch=master)
[![Code Health](https://landscape.io/github/michaelaye/planet4/master/landscape.svg?style=plastic)](https://landscape.io/github/michaelaye/planet4/master)

See more on the Wiki [here](https://github.com/michaelaye/planet4/wiki)

Release link [![DOI](https://zenodo.org/badge/15486/michaelaye/planet4.svg)](https://zenodo.org/badge/latestdoi/15486/michaelaye/planet4)

Poster at the DPS 2015 conference:
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.34114.svg)](http://dx.doi.org/10.5281/zenodo.34114)

## Requirements

I develop now exclusively on Python 3.5. If you are still on Python 2, you really should upgrade. Ana/Miniconda from continuum.io makes it easy to have both versions though, in case you need it.

## Install:

```bash
cd <where_u_store_software_from_others>
# next command will create folder `planet4`
git clone https://github.com/michaelaye/planet4.git
cd planet4
python setup.py install
```

This will add the module `planet4` to your importable list of modules. (Without the need of adapting PYTHONPATH)
