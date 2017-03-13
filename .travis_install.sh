#!/bin/bash
# this script is called by the install step in .travis.yml

set -e

echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip

# deactivating the travis provided virtual env
deactivate

pushd .
cd
mkdir -p download
cd download
echo "Cached in $HOME/download :"
ls -l
echo
if [[ ! -f miniconda.sh ]]
  then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
            -O miniconda.sh
  fi
chmod +x miniconda.sh && ./miniconda.sh -b
cd ..
export PATH=/home/travis/miniconda/bin:$PATH
conda config --set always_yes yes --set changeps1 no
conda update conda
conda info -a
popd
conda create -q -n test-environment python=3
source activate test-environment
conda install --file requirements.txt
conda install pytest
pip install coveralls
python setup.py install
python -c "from planet4 import io;io.set_database_path('~')"
