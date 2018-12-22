# Learning Monocular Stereo
This project was created for CSCI5980 at University of Minnesota.

A CNN that predicts depth and motion from an image pair. Based on the DeMoN network (https://github.com/lmb-freiburg/demon).

## Installation
This package requires

    tensorflow
    python 3.5
    cuda 8

You also have to manually build these packages and place them in the parent directory to this project. Follow their respective readme for build instructions.

    Multi View H5 Data Reader  (https://github.com/lmb-freiburg/demon/tree/master/multivih5datareaderop)
    lmbspecialops - (https://github.com/lmb-freiburg/lmbspecialops)


I recommend installing everything in a virtualenv.
