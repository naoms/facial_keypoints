# Facial keypoints detection and filter application


In this repository, we implement end-to-end face to face detection and filter application using Google Colab. 

![Altered video screenshot](https://github.com/sarahj134/facial_keypoints/blob/main/demo_img.jpg?raw=true)

Our filters are applied with growing difficulty : from static webcam images and screenshots to videos, both online and offline. 
We successfully implemented this project thanks to two key references : 

- [Balraj98's winning solution for the facial keypoint Kaggle Competition](https://www.kaggle.com/balraj98/data-augmentation-for-facial-keypoint-detection).   
- [Rohit Agrawal's work on filter application](https://www.codementor.io/@rohitagrawalofficialmail/how-and-why-i-built-snapchat-filter-system-x5p95x8i0 ).   

The project consists in two parts : 

## Model training and basic inferences : Facial_keypoints.ipynb
In this notebook, we perform data augmentation as per Balraj98's solution: our dataset quadruples in size, from ~7,000 images to over 28,000 images for training.
Data augmentation operations include : 
- Horizontal flip
- Rotation (12,-12)
- Brightness augmentation 
- Shifting 
- Random noise augmentation

We then implement two models using this augmented dataset : 
- a CNN architecture with LeakyReLU activations and batch normalization. We exactly replicate Balraj's architecture, and re-load his model with an additional training of 50 epochs. This model has a total of over 7,000,000 parameters to train. 
- a lighter version based on transfer learning : the MobileNet model trained on the ImageNet dataset, of which we freeze the weights and add our custom output layers. This model had half as many parameters, around 3,500,000, of which most were frozen. 


We saved our best models for both tentatives into the "models" directory because they were too heavy. 

In the last part of the notebook, you will be able to visualise keypoint prediction for both models, as well as filter application for one and several people. For this part we tweaked Rohit Agrawal's filter implementation, adding our custom scaling factors to improve filter robustness across various image scales. 



## Model inference on webcam pictures, screenshots, and videos, both offline and online : Filter_application.ipynb

In this notebook, we use our best model for filter application in various situations. The code to process a video online, offline, and make a GIF out of it is included. 



