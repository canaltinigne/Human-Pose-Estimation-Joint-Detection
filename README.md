# Human Pose Estimation & Joint Detection:

## Semester Project - MSc in Computer Science @ EPFL

by <br>
_Can Yilmaz Altinigne_ <br>
under the supervision of <br>
_Radhakrishna Achanta_ and _Dorina Thanou_ <br>
**@ Swiss Data Science Center** <br>

## Abstract:

In this project, we aim to train a fully convolutional network for human pose segmentation. For this purpose, we use U-Net 
architecture which is mainly used in segmentation tasks. Additionally, we change the convolution layers with the harmonic 
convolution layers to test if we can make learning faster. Then we train a combined model based on the best segmentation 
model to find accurate human pose segmentation masks and joint locations on human body concurrently.

## Dataset:

Data samples that we use in order to create our own datasets are from Leeds Sports Images Dataset [1], MPII Human Pose 
Dataset [2] and Fashion Dataset [3]. Since it is nearly 10 GB, the dataset couldn't be shared here.

## Creating Target Values:

- Masks - Mask-RCNN: https://github.com/matterport/Mask_RCNN
- Joint Locations - OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose

## Models:

We implemented U-Net models using Keras framework. Also we use harmonic networks in order to check if the learning process 
is faster than regular convolutions, and also we may utilize rotational invariance feature of harmonic networks even though 
the most images do not suffer from the rotation problem using PyTorch and a GitHub library [4].

## Environment:

You can create the same Conda environment using .yaml file in /env folder. Also you can find the Python libraries in 
/env/requirements.txt

## References:

[1] S. Johnson and M. Everingham, “Clustered pose and nonlinear appearance models for human pose estimation.” in BMVC, 
vol. 2, no. 4, 2010, p. 5.

[2] M. Andriluka, L. Pishchulin, P. Gehler, and B. Schiele, “2d human pose estimation: New benchmark and state of the 
art analysis,” in Proceedings of the IEEE Conference on computer Vision and Pattern Recognition, 2014, pp. 3686–3693.

[3] X. Liang, C. Xu, X. Shen, J. Yang, S. Liu, J. Tang, L. Lin, and S. Yan, “Human parsing with contextualized convolutional 
neural network,” in Proceedings of the IEEE International Conference on Computer Vision, 2015, pp. 1386–1394.

[4] https://github.com/jatentaki/harmonic
