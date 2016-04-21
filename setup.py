#!/usr/bin/env python

import sys
# To use a consistent encoding
from codecs import open
from os import path

import versioneer
from setuptools import setup
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['-v']
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name="planet4",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=['planet4'],
    install_requires=['pandas>=0.17.1'],
    tests_require=['pytest', 'pandas', 'pytables'],
    extras_require={
        'test': ['coverage'],
    },
    package_data={
        'planet4': ['data/*']
    },

    cmdclass={'test': PyTest},

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
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: ISC License (ISCL)',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ]
)
