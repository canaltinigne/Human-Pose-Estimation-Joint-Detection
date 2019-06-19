# HUMAN POSE ESTIMATION & JOINT DETECTION:

## Semester Project - MSc in Computer Science @ EPFL
---
by
Can Yilmaz Altinigne
under the supervision of 
Radhakrishna Achanta and Dorina Thanou 
@ Swiss Data Science Center
---
## ABSTRACT:

In this project, we aim to train a fully convolutional network for human pose segmentation. For this purpose, we use U-Net 
architecture which is mainly used in segmentation tasks. Additionally, we change the convolution layers with the harmonic 
convolution layers to test if we can make learning faster. Then we train a combined model based on the best segmentation 
model to find accurate human pose segmentation masks and joint locations on human body concurrently.
---
## DATASET:

Data samples that we use in order to create our own datasets are from Leeds Sports Images Dataset [1], MPII Human Pose 
Dataset [2] and Fashion Dataset [3].
---
## CREATING TARGET VALUES:

MASKS - Mask-RCNN: https://github.com/matterport/Mask_RCNN
JOINT LOCATIONS - OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose
---
## MODELS:

We implemented U-Net models using Keras framework. Also we use harmonic networks in order to check if the learning process 
is faster than regular convolutions, and also we may utilize rotational invariance feature of harmonic networks even though 
the most images do not suffer from the rotation problem using PyTorch and a GitHub library [4].
---
## ENVIRONMENT:

You can create the same Conda environment using .yaml file in /env folder. Also you can find the Python libraries in 
/env/requirements.txt
---
## DIRECTORIES:

* /
|----->	* /data
|	|----->	/BIG_18K_DATASET
|	|	|----->	/BIG_18K_IMAGES: 18.6k single-person full-body images.
|	|	|----->	/BIG_18K_MASKS: Mask RCNN outputs (binary image) for mask target values for 18.6k single-person full-body images.
|	|	|----->	/BIG_18K_OPENPOSE: OpenPose outputs (JSON files) for joint location target values for 18.6k single-person full-body images.
|	|
|	|----->	/ORIGINAL_IMAGES: Original 18.6k images from 3 different datasets
|	|
|	|----->	/SMALL_5K_DATASET
|		|----->	/SMALL_5K_HEATMAPS: Regression values (real values) between 0-1 for joint prediction in Experiment 1 in 2D matrix
|		|----->	/SMALL_5K_HEATMAPSOFTMAX: Saved numpy arrays to indicate the gaussian pixel locations w/ threshold (0.01) for joints in 2D matrix.
|		|----->	/SMALL_5K_IMAGES: Rescaled (shorter edge is 128px) 5k single-person full-body images.
|		|----->	/SMALL_5K_JOINTCENTERPIXELS: Saved numpy arrays to indicate the position of the center pixel location for joints in 2D matrix
|		|----->	/SMALL_5K_JOINTGAUSSIANS: Saved numpy arrays to indicate the gaussian pixel w/o threshold locations for joints in 2D matrix.
|		|----->	/SMALL_5K_MASKS: Target mask values in binary images (output of Mask-RCNN)
|		|----->	/SMALL_5K_OPENPOSE: Target joint locations in JSON files (output of OpenPose)
|
|----->	* /env
|	|----->	environment.yaml: Create Conda Environment using this file
|	|----->	requirements.txt: Used Python libraries
|
|----->	* /harmonic: Harmonic Network Library (PyTorch Code)
|
|----->	* /presentations: All presentations during the project progress
|
|----->	* /saved_models
|	|----->	EXP1-100ep-heatmap-mask-model-newest-08996-161.h5: Saved model for Experiment 1, Mask and Regression Heatmap Model w/ UNet
|	|----->	EXP2-100ep-heatmap-mask-model-val_dice_9221-val_cce_00564.h5: Saved model for Experiment 2, Mask and Softmax Heatmap Model w/ UNet
|	|----->	EXP3-model-088-0.093179.h5: Saved model for Experiment 3, Mask and Regression Softmax Heatmap Model w/ UNet
|	|----->	harmonic_100ep_adam2e5_3232_512512_drop_bnorm.pt: Saved model for Only Mask Prediction w/ Harmonic Networks
|	|----->	harmonic_100ep_adam2e5_3232_512512_drop_bnorm.pth: Same as above
|	|----->	UNet-OnlyHumanPoseMasks-model-089-0.919240-0.080760.h5: Saved model for Only Mask Prediction w/ UNet
|	|----->	UNet-OnlyRegressionHeatmaps-model-heatmap-009-13.708091.h5: Saved model for only Joint Detection (Regression) w/ UNet
|
|----->	* /src
|	|----->	/dataset-prepare
|	|	|----->	CheckImageJointNumbers.ipynb: Example code for finding joint numbers in each image using OpenPose outputs.
|	|	|----->	HeatmapCreator.ipynb: Example code for creating heatmaps for the simultaneous mask and joint location prediction experiments
|	|	|----->	SmallDatasetScaling.ipynb: Example code for rescaling training set images using Bicubic Interpolation
|	|	|----->	TargetMaskValues-MaskRCNN.ipynb: Example MaskRCNN code to create target mask values
|	|----->	/models
|	|	|----->	Exp1-UNet-Mask-Regression.ipynb: EXP-1 Mask and Regression Heatmap Model: A combined model which outputs human pose masks and predicts the heatmaps
|	|	|----->	Exp2-UNet-Mask-Softmax.ipynb: EXP-2 Mask and Softmax Heatmap Model: Human pose masks and joint locations (1's in channels)
|	|	|----->	Exp3-UNet-Mask-SoftmaxRegression.ipynb: EXP-3 Mask and Regression Softmax Model: Human pose masks and joint locations (01's in channels)
|	|	|----->	Exp4-TripleModel.ipynb: Combination of EXP-1, EXP-2 for joint prediction and also Mask Prediction, three outputs
|	|	|----->	HarmonicNet-OnlyHumanPoseMasks.ipynb: Training Harmonic Network for only Human Pose Estimation Experiment
|	|	|----->	HarmonicNet-OnlyRegressionHeatmaps.ipynb: Training Harmonic Network for only Joint Detection (Regression) Experiment
|	|	|----->	UNet-Mask-JointCenterPixels.ipynb: Training U-Net for Mask Prediction and Joint Center Pixel Estimation Experiment
|	|	|----->	UNet-OnlyHumanPoseMasks.ipynb: Training U-Net for only Human Pose Estimation Experiment
|	|	|----->	UNet-OnlyRegressionHeatmaps.ipynb: Training U-Net for only Joint Detection (Regression) Experiment
|	|----->	/python
|		|----->	hnet_datagenerator.py: Data generator function for HNet training
|		|----->	hnet_loss.py: Loss functions that was used in HNet Experiments
|		|----->	hnet_onlymask.py: HNet model for only Mask prediction
|		|----->	skeleton_draw.py: Joint connector and limb drawer functions
|		|----->	unet_datagenerator.py: Data generator function for UNet training
|		|----->	unet_exp2model.py: UNet Model in Exp2-UNet-Mask-Softmax.ipynb (Experiment 2 in Report)
|		|----->	unet_loss.py: Loss functions that was used in UNet Experiments
|	
|----->	* /torch-dimcheck: Used libraries for Harmonic Network Implementation
|
|----->	* /torch-localize: Used libraries for Harmonic Network Implementation
|
|----->	* harmonic_network_lite.py: Harmonic Network Operations File in PyTorch
|
|----->	* harmonic_network_ops.py: Harmonic Network Operations File in PyTorch
|
|----->	* report.pdf: Final Project Report

---
## REFERENCES:

[1] S. Johnson and M. Everingham, “Clustered pose and nonlinear appearance models for human pose estimation.” in BMVC, 
vol. 2, no. 4, 2010, p. 5.

[2] M. Andriluka, L. Pishchulin, P. Gehler, and B. Schiele, “2d human pose estimation: New benchmark and state of the 
art analysis,” in Proceedings of the IEEE Conference on computer Vision and Pattern Recognition, 2014, pp. 3686–3693.

[3] X. Liang, C. Xu, X. Shen, J. Yang, S. Liu, J. Tang, L. Lin, and S. Yan, “Human parsing with contextualized convolutional 
neural network,” in Proceedings of the IEEE International Conference on Computer Vision, 2015, pp. 1386–1394.

[4] https://github.com/jatentaki/harmonic
---
