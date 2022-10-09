# CS5330 - Computer Vision
# Project 5: Recognition using Deep Networks
# April 1, 2022
# Sida Zhang and Hongyu Wan
# Please visit my Wiki report to see more contents and the video capture extension: 
# https://wiki.khoury.northeastern.edu/display/~zhangsida1997/CS5330%3A+Project+5%3A+Recognition+using+Deep+Networks
# Environment: 
#	OS: Windows 11
#	IDE: Pycharm
#	All Imports:
os, cv2, sys, torch, torchvision, torch.nn as nn, torch.nn.functional as F,
torch.optim as optim, matplotlib.pyplot as plt, from torch.utils import data,
from torch.utils.data import DataLoader, from PIL import Image, import numpy as np
random, math. from torchvision import models, from torchvision import transforms
from collections import Counter
#	Project Structure:
  - Project 5:
	- data
	   - greek_data
		- 27 (3x9) png images
	   - greek_input
		- 9 (3x3) png images
	   - handwritting
		- 9 (1x9) png images
	- extension
	   - extension_test
		- 12 (2x6) png images
	   - extension_train
		- 66 (6x11) png images
	- results
	   - extension_csv
		- dataset.csv
		- label.csv
	   - greek_symbol_csv
		- dataset.csv
		- label.csv
	   - networkTrained
		- model.pth
		- optimizer.pth
	   - conv_multi3.txt
	   - conv_original.txt
	   - customized_results.csv
	- src
	   - extension1_classifier.py
	   - extension2_livevideo.py
	   - extension3_resnet.py
	   - task1_handwritting.py
	   - task1_main.py
	   - task1_testset.py
	   - task2_analysis.py
	   - task3_main.py
	   - task4_main.py
	   - task4_multiconv.py



# For Wiki and video demonstration please visit: https://wiki.khoury.northeastern.edu/display/~zhangsida1997/CS5330%3A+Project+5%3A+Recognition+using+Deep+Networks
 Please feel free to comment on our Wiki Page or leave a like!

--------------------------------------------------------------------------------------------------
Instructions:
 The project was divided into three programs
	1st. livevid_I.cpp:
		Detect and Extract Chessboard Corners
		Select Calibration Images
		Calibrate the Camera
	This program saves camera matrix and distortion coefficients to /data/.
	2st. livevid_II.cpp:
		Calculate Current Position of the Camera
			Loads camera matrix and distortion coefficients data
		Project Outside Corners or 3D Axes (press A or A)
		Create virtual Object
			Diamond (press v or V)
			Icosahedron (press i or I)
			Aircraft (press e or E)
		Detect Robust Features
			Harris Corners (press h or H)
	3rd. livevid_III.cpp:
		ArUco functionalities
	calibration.cpp:
		All functionalities
---------------------------------------------------------------------------------------------------------------------------------------------------
Travel day usage and re-submission:

 We had to use one travel day for this project because we want to do a couple extensions
	and we didn't have the time to hand it in on time.

 We have talked to Dr. Maxwell about the situation and he has approved.

 Thank you so much for your understanding.
---------------------------------------------------------------------------------------------------------------------------------------------------

Tasks:
 All tasks' results are available on the Wiki Page.
 Task1: Build and  train a network to do digit
        recognition using the MNIST database.
	  Read the network from task1_main and
        run it on the test set.
	  Created Handw-ritting data sets and
        test the network on new Handwrittings
 Task2: Examine network by analyze the layers
        and make effect of the filters. We have also built
        a truncated model to load the state dictionary
        that we have read from the file.
 Task3: Create greek letter embedding space
    	  and compute the distance in the embedding space
 Task4: 
	  1. We have designed our new train with
    		batch_size_list of [64, 128, 256],
		epochs_list of [1, 5, 10],
    		learning_rate_list of [0.001, 0.01, 0.1], and
    		dropout_rate_list of [0.125, 0.25, 0.5],
    		for this program..
	  2. We have designed our new train with 3
		convolutional layers for this program.

Extensions:
 Several extensions are included in this project.
 Please visit the Wiki Page for more information.
	Extension 1: For this program, we have wrote our on database
    		with three new greek letters "Pi, Theta, and Mu"
    		Each of the greek letter have 11 training sets and
		2 test sets
    		We have built an actual KNN classifier that can take
		in any square image and classify it.
	Extension 2: A live video digit recognition application
    		using the trained network.
	Extension 3: The progoram downloads ResNetx from pytorch
    		resnext50_32x4d and run the submodel(torchvision)

---------------------------------------------------------------------------------------------------------------------------------------------------

Visit Wiki Report for details: https://wiki.khoury.northeastern.edu/display/~zhangsida1997/CS5330%3A+Project+5%3A+Recognition+using+Deep+Networks

---------------------------------------------------------------------------------------------------------------------------------------------------
Please feel free to comment on our Wiki Page or leave a like!