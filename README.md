<!-- ## Badges

(Customize these badges with your own links, and check https://shields.io/ or https://badgen.net/ to see which other badges are available.)

| fair-software.eu recommendations | |
| :-- | :--  |
| (1/5) code repository              | [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/QuantumApplicationLab/wntr-quantum) |
| (2/5) license                      | [![github license badge](https://img.shields.io/github/license/QuantumApplicationLab/wntr-quantum)](https://github.com/QuantumApplicationLab/wntr-quantum) |
| (3/5) community registry           | [![RSD](https://img.shields.io/badge/rsd-wntr_quantum-00a3e3.svg)](https://www.research-software.nl/software/wntr_quantum) [![workflow pypi badge](https://img.shields.io/pypi/v/wntr_quantum.svg?colorB=blue)](https://pypi.python.org/project/wntr_quantum/) |
| (4/5) citation                     | [![DOI](https://zenodo.org/badge/DOI/<replace-with-created-DOI>.svg)](https://doi.org/<replace-with-created-DOI>) |
| (5/5) checklist                    | [![workflow cii badge](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>/badge)](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>) |
| howfairis                          | [![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu) |
| **Other best practices**           | &nbsp; |
| Static analysis                    | [![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=QuantumApplicationLab_wntr-quantum&metric=alert_status)](https://sonarcloud.io/dashboard?id=QuantumApplicationLab_wntr-quantum) |
| Coverage                           | [![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=QuantumApplicationLab_wntr-quantum&metric=coverage)](https://sonarcloud.io/dashboard?id=QuantumApplicationLab_wntr-quantum) |
| Documentation                      | [![Documentation Status](https://readthedocs.org/projects/wntr-quantum/badge/?version=latest)](https://wntr-quantum.readthedocs.io/en/latest/?badge=latest) |
| **GitHub Actions**                 | &nbsp; |
| Build                              | [![build](https://github.com/QuantumApplicationLab/wntr-quantum/actions/workflows/build.yml/badge.svg)](https://github.com/QuantumApplicationLab/wntr-quantum/actions/workflows/build.yml) |
| Citation data consistency          | [![cffconvert](https://github.com/QuantumApplicationLab/wntr-quantum/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/QuantumApplicationLab/wntr-quantum/actions/workflows/cffconvert.yml) |
| SonarCloud                         | [![sonarcloud](https://github.com/QuantumApplicationLab/wntr-quantum/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/QuantumApplicationLab/wntr-quantum/actions/workflows/sonarcloud.yml) |
| MarkDown link checker              | [![markdown-link-check](https://github.com/QuantumApplicationLab/wntr-quantum/actions/workflows/markdown-link-check.yml/badge.svg)](https://github.com/QuantumApplicationLab/wntr-quantum/actions/workflows/markdown-link-check.yml) | -->

## WNTR Quantum

WNTR Quantum, builds on the python package WNTR to create a quantum enabled water nework management toolkit.


## Installation

To install wntr_quantum from GitHub repository, do:

```console
git clone git@github.com:QuantumApplicationLab/wntr-quantum.git
cd wntr-quantum
python -m pip install .
```

## Installation of EPANET Quantum

WNTR Quantum can use a dedicated EPANET solver that allows to offload calculation to quantum linear solvers. This custom EPANET code can be found at : https://github.com/QuantumApplicationLab/EPANET. To install this sover follow the instructions below:


```
# clone EPANET
git clone https://github.com/QuantumApplicationLab/EPANET

# build EPANET
cd EPANET
mkdir build
cd build 
cmake .. 
cmake --build . --config Release

# copy the shared lib
cp lib/libepanet2.so <path to wntr-quantum>/wntr-quantum/wntr_quantum/epanet/Linux/libepanet22_amd64.so

# export environment variable
export EPANET_TMP=<path to tmp dir>/.epanet_quantum 
export EPANET_QUANTUM = <path to EPANET_QUANTUM>
```

## Documentation

Include a link to your project's full documentation here.

## Contributing

If you want to contribute to the development of wntr_quantum,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
