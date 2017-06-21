# To use a consistent encoding
from codecs import open
from os import path

from setuptools import setup, find_packages

DISTNAME = 'planet4'
DESCRIPTION = "Software for the reduction and analysis of PlanetFour data."
AUTHOR = "K.-Michael Aye"
AUTHOR_EMAIL = "michael.aye@lasp.colorado.edu"
MAINTAINER_EMAIL = AUTHOR_EMAIL
URL = "https://github.com/michaelaye/planet4"
LICENSE = "ISC"
KEYWORDS = ['Mars', 'science', 'MRO', 'imaging']
DOWNLOAD_URL = "https://github.com/michaelaye/planet4"

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()


setup(
    name=DISTNAME,
    version='0.10.0',
    packages=find_packages(),

    install_requires=['pandas', 'numpy', 'matplotlib', 'pyaml'],
    tests_require=['pytest'],
    setup_requires=['pytest-runner'],
    package_data={
        'planet4': ['data/*']
    },

    entry_points={
        "console_scripts": [
            'p4reduction = planet4.reduction:main',
            'plot_p4_imageid = planet4.markings:main',
            'create_season2and3 = planet4.reduction:create_season2_and_3_database',
            'p4catalog_production = planet4.catalog_production:main',
            'hdf2csv = planet4.hdf2csv:main',
            'cluster_image_id = planet4.dbscan:main'
        ]
    },

    # metadata
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    license=LICENSE,
    keywords=KEYWORDS,
    url=URL,
    download_url=DOWNLOAD_URL,

    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers'
        'License :: OSI Approved :: ISC License (ISCL)',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ]
)
