![build](https://github.com/reiterlab/reiterlablib/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/reiterlab/reiterlablib/branch/main/graph/badge.svg?token=3OJS5MCMSC)](https://codecov.io/gh/reiterlab/reiterlablib)
![PyPI](https://github.com/reiterlab/reiterlablib/workflows/PyPI/badge.svg)

# reiterlablib (rll)
A collection of different modules providing methods for frequently used computations and visualizations used in the reiterlab. 
See some of our developed software on our Github page [https://github.com/reiterlab]().


## <a name="installation"> Installation 
To install the latest release of rll, run ```pip install rll```


*If you want to customize some functions, it is best to:
1. Clone the repository from Github with ```git clone https://github.com/reiterlab/reiterlablib.git```
1. Create distribution package by going to the main folder with ```cd reiterlablib``` and run ```python setup.py clean sdist bdist_wheel``` 
1. Install *rll* to your python environment by running ```pip install -e .```
1. Test installation with ```python -c 'import rll'``` and ```pytest tests/```

### <a name="releases"> Releases
* rll v0.1.0 2020-12-11: Initial release of package.
* rll v0.1.2 2020-12-11: Added flexibility with legend display.

### License
These methods are licensed under the GNU General Public License, Version 3. The methods are free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License. There is no warranty for this free software.

Translational Cancer Evolution Laboratory, Canary Center for Cancer Early Detection, Stanford University, https://reiterlab.stanford.edu