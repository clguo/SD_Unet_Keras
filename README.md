# SD_Unet_Keras



# [SD-Unet: A Structured Dropout U-Net for Retinal Vessel Segmentation (IEEE BIBE)](https://ieeexplore.ieee.org/document/8942005)

## Overview

At present, artificial visual diagnosis of fundus diseases has low manual reading efficiency and strong subjectivity, which easily causes false and missed detections. Automatic segmentation of retinal blood vessels in fundus images is very effective for early diagnosis of diseases such as the hypertension and diabetes. In this paper, we utilize the U-shaped structure to exploit the local features of the retinal vessels and perform retinal vessel segmentation in an end-to-end manner. Inspired by the recently DropBlock, we propose a new method called Structured Dropout U-Net (SD-Unet), which abandons the traditional dropout for convolutional layers, and applies the structured dropout to regularize U-Net. Compared to the state-of-the-art methods, we demonstrate the superior performance of the proposed approach.

This code is for the paper: SD-Unet: A Structured Dropout U-Net for Retinal Vessel Segmentation

Code written by Changlu Guo, Budapest University of Technology and Economics(BME).


We train and evaluate on Ubuntu 16.04, it will also work for Windows and OS.



### Datasets
#### Data augmentation:
Please refer to [SA-UNet](https://github.com/clguo/SA-UNet)

if you do not want to do above augmentation,just download it from my link.

[DRIVE](https://drive.google.com/file/d/1t_UxlVWZXBtJQQNxW0vPdwrnqcdYdrRs/view?usp=sharing)
[CHASE_DB1](https://drive.google.com/file/d/1RnPR3hpKIHnu0e3y9DBOXKPXuiqPN8hg/view?usp=sharing)




## Environments
Keras 2.3.1  <br>
Tensorflow==1.14.0 <br>

## About Keras

Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
supports both convolutional networks and recurrent networks, as well as combinations of the two.
supports arbitrary connectivity schemes (including multi-input and multi-output training).
runs seamlessly on CPU and GPU.
Read the documentation [Keras.io](http://keras.io/)

Keras is compatible with: Python 2.7-3.5.



# If you are inspired by our work, please cite these papers.


@INPROCEEDINGS{8942005,  <br>
author={C. {Guo} and M. {Szemenyei} and Y. {Pei} and Y. {Yi} and W. {Zhou}}, <br> 
booktitle={2019 IEEE 19th International Conference on Bioinformatics and Bioengineering (BIBE)},   <br>
title={SD-Unet: A Structured Dropout U-Net for Retinal Vessel Segmentation},   <br>
year={2019},  <br>
volume={},  <br>
number={},  <br>
pages={439-444},}<br>

OR


C. Guo, M. Szemenyei, Y. Pei, Y. Yi and W. Zhou, "SD-Unet: A Structured Dropout U-Net for Retinal Vessel Segmentation," 2019 IEEE 19th International Conference on Bioinformatics and Bioengineering (BIBE), Athens, Greece, 2019, pp. 439-444, doi: 10.1109/BIBE.2019.00085.












