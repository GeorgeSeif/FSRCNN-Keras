# Keras FSRCNN

## Overview
An implementation of the Super Resolution CNN proposed in:

Chao Dong, Chen Change Loy, Xiaoou Tang. Accelerating the Super-Resolution Convolutional Neural Network, in Proceedings of European Conference on Computer Vision (ECCV), 2016

Written in Python using Keras on top of TensorFlow

- [The author's project page](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html)
- To download the required data for training/testing, please refer to the README.md at data directory.

## Files
- FSRCNN.py : main training file

## Usage
To download the required data for training/testing, please refer to the README.md at data directory.

The training images come from the 291 images data set. Simply run the "FSRCNN.py" script to begin training. 

**NOTE:** 32x32 HR images are used in the program rather than the size suggested in the paper. 