# Astro-Fix: Correcting Astronomical Bad Pixels in Python
**Authors:** Hengyue Zhang, Timothy D. Brandt

## Description
**astrofix** is an astronomical image correction algorithm based on Gaussian Process Regression. It trains itself to apply the optimal interpolation kernel for each image, performing multiple times better than median replacement and interpolation with a fixed kernel.

Please cite our original paper at [].  

## Installation
To install, git clone the repo by running `git clone https://github.com/HengyueZ/astrofix`.  
Then, run: `cd astrofix`. You will be under the root directory of this repo.   
Then, run: `pip install -e .` to finally install **astrofix**.  
If something goes wrong with the installation, please let us know by submitting an issue ticket!

## Usage
A sample Jupiter notebook showing the basic usage of **astrofix** is attached. The image used in the example is available at []  

## License
The project is licensed under the terms of the BSD 3-clause license. See the License file for details.
