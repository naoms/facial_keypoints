# Eleven2020 

## Project Presentation 

The purpose of this project is to leverage construction works camera images to identify best performing concreting sites for our client.  
First, Machine Learning models are trained to identify objects of interest : cementing area, workers, pumps...
Then, we define an algorithm to identify whether a pump is active and how engaged workers are with the concreting zone, to ultimately derive useful and useable monitoring information for our client's operation teams. 

This project is seperated into 2 main phases :  
    **-  MODEL 1 - Detection model**
        This model 1, detects various objects on construction site images  
    **-  MODEL 2 - Analytics model**
        This model 2, analyses the results from model 1 and returns an aggregated and synthetised table.  

Don't hesitate to go back to our presentation for more information.   


## Step by Step 

### 1. Detection model for bounding boxes - Yolov5
We trained Yolov5 on 1,500+ construction sites images to identify people. We got a validation mAP of 77%. 

### 2. Detection model for polygons - Mask R-CNN
We trained Mask R-CNN model on construction sites images in order to detect concrete pumps. We opt for transfer learning using the pre-trained COCO weights, training top layers on 30 epochs. 
We achieved the following results on the validation set: 53% IoU and 68% maP@0.5

### 3. Identification of concreting zone 
Thanks to the polygon encompassing the concrete pump given by Mask R-CNN we were able to deduce the zone where the concreting operation took place on the construction site. We followed several steps to delimit this zone.
#### a. Find the extremity of concrete pump
First we needed to find the extremity of the concrete pump. The extremity is generally the thinnest part of the polygon.We implemeted a rule finding the two closest points of the polygon and taking the mid-point between them.
We coded two other rules that we did not implemented yet: 
- Take the lowest point of the polygon. On most images, the pump is working towards camera and therefore the extremity is the lowest point of the polygon on the image
- Find the most vertical part of the polygon and take the end point. The pump is divided into 3 arms. Usually the last arm that is linked to the extremity of the pump is vertical. 

#### b. Determine the concreting time window
To delimit the concreting zone we needed to know the time frame during which the concreting work was occuring. 
To do so our algorithm checks when the concrete pumps appears and when it disappears from the successive image frames we have. 

#### c. Delimit the concreting zone 
We then track the pump extremity over the concreting operation window. This gives us a polygon that determines the concreting zone. 

### 4. Analysis of working force efficiency
Once we have the polygon we simplify it to a rectangle. We assume that workers who are not too far from this rectangle are involved in the concreting task. We expand the rectangle proportionnaly to its diagonal length to take perspective into account. Knowing the number of workers during the concreting period thanks to Yolov5 predictions, we can analyse the working force efficiency. We take the ratio between workers inside the delimited working zone and the total number of worker identified during the concreting task. 

## Installation

1. Clone this repository

2. Install dependencies

The main application can be found in the main.py file.  
To run it, type in the terminal :  

```bash
python main.py <IMAGES_PATH> <ACTION>  
```
  
The action argument is not mandatory and can take 4 values : ['yolo', 'mask', 'model1', 'model2']  
If action is yolo, the application will return the output of the images in IMAGES_PATH run on our yolo model.   
The output will be stored in XXXX  
      
If action is mask, the application will return the output of the images in IMAGES_PATH run on our mask model.    
The output will be stored in XXXX.  
      
If action is model1, the application will return the output of the images in IMAGES_PATH run on both our yolo and mask model.   
The output will be stored in a json with the COCO format in  : XXXX.  
      
If action is model2, the application will return the final aggregated and synthesised table.  

## Requirements

pip install keras==2.1.5
pip install tensorflow-gpu==1.15.2
pip install tensorflow==1.15.2
pip install pycocotools

You can also find this in the requirements.txt file

## Slides
[Slides](https://docs.google.com/presentation/d/1UHg2GKkDu0jf3gcDpQGbwKZuWa7OT7MYBbmeZJ41jsg/edit?usp=sharing)


