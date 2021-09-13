# Astro-Fix: Correcting Astronomical Bad Pixels in Python
**Authors:** Hengyue Zhang, Timothy D. Brandt

## Description
**astrofix** is an astronomical image correction algorithm based on Gaussian Process Regression. It trains itself to apply the optimal interpolation kernel for each image, performing multiple times better than median replacement and interpolation with a fixed kernel.

Please cite our original paper at [Zhang, H. & Brandt, T. D. 2021, AJ, 162, 139](https://doi.org/10.3847/1538-3881/ac1348).  

## Installation
To install, git clone the repo by running `git clone https://github.com/HengyueZ/astrofix`.  
Then, run: `cd astrofix`. You will be under the root directory of this repo.   
Then, run: `pip install -e .` to finally install **astrofix**.  

### Tests
To test the installation, you will need to download [a sample image of NGC 104](https://archive.lco.global/?q=a&RLEVEL=&PROPID=&INSTRUME=&OBJECT=&SITEID=&TELID=&FILTER=&OBSTYPE=&EXPTIME=&BLKUID=&REQNUM=&basename=cpt0m407-kb84-20200917-0147-e91&start=2020-09-17%2000%3A00&end=2020-09-18%2000%3A00&id=&public=true) from the LCO archive, and put it under your local astrofix/astrofix/tests folder. Then, you can run the tests by `cd astrofix` (if you have not done so in the installation step) and `pytest -sv`. There are three tests in total and they should take about 30 seconds. If any of the tests fail, please let us know by submitting an issue ticket!  

## Usage
A sample Jupyter notebook showing the basic usage of **astrofix** is attached. The images used in the example are available from the LCO archive at the links below:  
[NGC104](https://archive.lco.global/?q=a&RLEVEL=&PROPID=&INSTRUME=&OBJECT=&SITEID=&TELID=&FILTER=&OBSTYPE=&EXPTIME=&BLKUID=&REQNUM=&basename=cpt0m407-kb84-20200917-0147-e91&start=2020-09-17%2000%3A00&end=2020-09-18%2000%3A00&id=&public=true)  
[M15](https://archive.lco.global/?q=a&RLEVEL=&PROPID=&INSTRUME=&OBJECT=&SITEID=&TELID=&FILTER=&OBSTYPE=&EXPTIME=&BLKUID=&REQNUM=&basename=cpt0m407-kb84-20201021-0084-e91&start=2020-10-21%2000%3A00&end=2021-10-22%2000%3A00&id=&public=true)

## License
The project is licensed under the terms of the BSD 3-clause license. See the License file for details.
