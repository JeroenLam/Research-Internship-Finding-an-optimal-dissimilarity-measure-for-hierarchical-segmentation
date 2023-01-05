# Automated segmentation of VHR images using alpha-trees
This repository contains the code related to the corresponding reasearch internship project. The corresponding report can be found [here](https://fse.studenttheses.ub.rug.nl/29064/).

### Installation
Note that the software has only been tested on Linux and WSL.

## Install 
Install [OpenCV](https://opencv.org/)

## Download the repository
`git clone https://github.com/JeroenLam/Automated-segmentation-VHR-images-using-Alpha-trees`

## Create build directory and compile code
`mkdir build` \
`cd build` \
`cmake ../` \
`cmake --build .` 

## Running the code
`./attree.out -h`

## Changing the number of alpha values considered in each run
You can change the number fo alpha values considered by changing the `SIM_STEPS` variable in `ATlib/at-fitness.h`.

## Running the code
Below you can find example commands to compute the score of a given set ground truth images and a command that can be used to filter the alpha tree. \
`./atree.out Difference ../img/Campus/zernike180701p11.png 3 ../img/Campus/GT1p11.png ../img/Campus/GT2p11.png ../img/Campus/GT3p11.png 2 1 2 1 2 1` \
`./atree.out Difference ../img/Campus/zernike180701p11.png -3 4 8 12 2 1 2 1 2 1` 

## Running the optimisation script
Make sure to run an instance of the [MOE Baysian Optimisation toolbox](https://github.com/Yelp/MOE) in Docker. \
Move to the `MOEParameterOpt` folder. \
`cp example_env .env` \ 
Update `.env` based on your parameters. \
run `start.sh <num>` where `<num> = {11,12,21,22}` \
This will spawn an instance for each method as a background task


## References
The alpha tree code is heavily based on the implementation of Felix Zailskas https://github.com/felix-zailskas/alpha-trees
