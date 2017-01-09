# To use a consistent encoding
from codecs import open
from os import path

from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name="planet4",
    version='0.7.3',
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'matplotlib'],
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
            'hdf2csv = planet4.hdf2csv:main'
        ]
    },

    # metadata
    author="K.-Michael Aye",
    author_email="kmichael.aye@gmail.com",
    description="Software for the reduction and analysis of Planet4 data.",
    long_description=long_description,
    license="ISC",
    keywords="Mars Planet4 Zooniverse",
    url="http://github.com/michaelaye/planet4",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: ISC License (ISCL)',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ]
)
