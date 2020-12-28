"""
    This file will contain a set of useful functions that we will use in our
    project for the analytics model
"""

import os
import json
import glob
import json
import sys
from math import sqrt
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter
import random
import cv2
import numpy as np
import pandas as pd
import shutil  

random.seed(300)

DIR_IMAGES = '../inputs/images'
DIR_LABELS = '../inputs/labels'

# Functions related to showing images

def show_image(img_path, figsize=(10,10)):
    """this method diplays the image img_path

    Parameters
    ----------
    img_path : str
        input image path
    figsize : tuple, optional
        size of the figure, by default (10,10)
    """
    img = mpimg.imread(img_path)
    plt.figure(figsize = figsize)
    plt.imshow(img)
    plt.show()


def show_image_sample(n, img_dir):
    """display as sample of images of size n from img_dir

    Parameters
    ----------
    n : int
        size of the sample
    img_dir : str
        images directory
    """
    figsize = (10*n,10)
    _, axs = plt.subplots(1, n, figsize=figsize)
    images_paths = sorted(glob.glob(f"{img_dir}*jpg"))
    img_sample_paths = random.choices(images_paths, k=n)
    for index, img_path in enumerate(img_sample_paths):
        img = mpimg.imread(img_path)
        axs[index].imshow(img)
    plt.show()


def draw_labels(img, label):
    """Draw the labels boudingboxes / polygon on the input image img
    and return the obtain image

    Parameters
    ----------
    img : str
        input image path
    label : str
        image label(json file) path

    Returns
    -------
    numpy.ndarray
        output image with boundingbo/polygon of all the objects in label
    """
    class_colors = {'Vertical_formwork': (255,0,255),
                    'People': (255, 255, 85),
                    'Mixer_truck': (0, 233, 0),
                    'Concrete_pump_hose': (0,0,255)}
    img = plt.imread(img)
    img_copy = img.copy()
    with open(label, 'r') as file:
        data = json.load(file)
    for obj in data["objects"]:
        if obj.get('geometryType') == 'rectangle':
            pt1 = tuple(obj.get('points').get('exterior')[0])
            pt2 = tuple(obj.get('points').get('exterior')[1])
            cv2.rectangle(img_copy, pt1, pt2, class_colors[obj.get('classTitle')], 2)
        elif obj.get('geometryType') == 'polygon':
            cv2.fillPoly(img, pts= [np.array(obj.get('points').get('exterior'))], color=class_colors[obj.get('classTitle')])
    out = cv2.addWeighted(img_copy, .6, img, 0.4, 1)
    return out


def show_image_labeled_sample(n, images_dir, labels_dir):
    """display a sample of images with labels boundingboxes/polygons

    Parameters
    ----------
    n : int
        size of the sample
    images_dir : str
        inmput images directory
    labels_dir : str
        iput labels directory
    """
    figsize = (20*n,20)
    _, axs = plt.subplots(1, n, figsize=figsize)
    images_paths = np.array(sorted(glob.glob(f"{images_dir}*jpg")))
    labels_paths = np.array(sorted(glob.glob(f"{labels_dir}*json")))
    index_sample = random.choices(range(len(images_paths)), k=n)
    print(index_sample)
    for index, data in enumerate(zip(images_paths[index_sample], labels_paths[index_sample])):
        img = draw_labels(data[0], data[1])
        axs[index].imshow(img)
    plt.show()

    
def show_image_with_extremity (img_path, extremity, label, figsize=(10,10)):
    img = mpimg.imread(img_path)
    img = draw_labels(img_path, label)
    img_copy = img.copy()
    plt.figure(figsize = figsize)
    plt.scatter(extremity[0], extremity[1], s=100, c='red', marker='X')

    plt.imshow(img_copy)
    plt.show()


# Functions related to analytics

def has_concrete_pump(df):
    """
    Segment the dataframe across time.
    """
    df["has_concrete_pump"] = df.classTitle.apply(lambda x : x=="Concrete_pump_hose")*1
    return pd.DataFrame(df.groupby("date_time").mean()["has_concrete_pump"]>0)


def locate_pump_tip():
    pass

def get_concreting_zone(df):
    # Concreting period
    # 
    return get_max_min_coordinates(my_series)


def get_max_min_coordinates(my_dict):
    """
    :param my_dict: key is the timeframe for the pump, values are the corresponding pump coordinates
    :return:
    """
    my_series = [elem for elem in my_dict.values()]
    x_points = [x[0] for x in my_series[0]] # indice 0 to get rid of the double brackets
    y_points = [x[1] for x in my_series[0]] # indice 0 to get rid of the double brackets
    x_min, x_max = np.min(x_points), np.max(x_points)
    y_min, y_max = np.min(y_points), np.max(y_points)
    bottom_left = tuple((x_min, y_min))
    bottom_right = tuple((x_max, y_min))
    top_left = tuple((x_min, y_max))
    top_right = tuple((x_max, y_max))
    return list(my_dict.keys())[0],[bottom_left, bottom_right, top_left, top_right]

def get_num_workers(df):
    """
    Retrieve the number of workers per time step.
    df: dataframe with one object per row
    """
    count_objects = df.groupby(["date_time", "classTitle"]).count().reset_index()
    count_objects["is_people"] = count_objects.classTitle.apply(lambda x : x=="People")*1
    count_objects["nb_workers"] = count_objects["is_people"] * count_objects["index"]
    return pd.DataFrame(count_objects.groupby("date_time").sum().loc[:,"nb_workers"])

def draw_pump_worker_anlysis(data_df, img_path, label, figsize=(10, 10)):
    """ prints an image and draws its label and the point given

    Parameters
    ----------
    data_df : pd.DataFrame 
        dataframe with images and objects
    img_path: str
        input image path
    label: str
        input label path
    figsize : tuple, optional
        size of the figure, by default (10,10)
    """
    img = mpimg.imread(img_path)
    img_copy = img.copy()
    json_name = img_path.split('/')[4]+'.json'
    
    #Draw zone and extended zone
    pts_zone = list(data_df[(data_df['filename']==json_name) & (data_df['classTitle']=='Concrete_pump_hose')]['pump_polygon'])
    pts_extended_zone = list(data_df[(data_df['filename']==json_name) & (data_df['classTitle']=='Concrete_pump_hose')]['extended_polygon'])
    cv2.fillPoly(img_copy, pts=[np.array(pts_extended_zone)], color = (75, 37, 109))
    cv2.addWeighted(img_copy, .6, img, 0.4, 1)
    cv2.fillPoly(img_copy, pts=[np.array(pts_zone)], color=(239, 62, 91))
    cv2.addWeighted(img_copy, 0.6, img, 0.4, 1)
    
    #Draw pump
    pts_exterior = list(data_df[(data_df['filename']==json_name) & (data_df['classTitle']=='Concrete_pump_hose')]['ext_points'])
    cv2.fillPoly(img_copy, pts=[np.array(pts_exterior)], color=(75, 37, 109))
    cv2.addWeighted(img_copy, 0.6, img, 0.4, 1)
    
    #Draw workers
    workers_working = data_df[(data_df['filename']==json_name) & (data_df['classTitle']=='People') & (data_df['working_status']=='Working')]['ext_points']
    workers_not_working = data_df[(data_df['filename']==json_name) & (data_df['classTitle']=='People') & (data_df['working_status']=='Not_working')]['ext_points']
    for pts_worker in workers_not_working:
        cv2.rectangle(img_copy, tuple(pts_worker[0]), tuple(pts_worker[1]), (239,132,91), 3)
    for pts_worker in workers_working:
        cv2.rectangle(img_copy, tuple(pts_worker[0]), tuple(pts_worker[1]), (149,212,122), 3)

    
    out_2 = cv2.addWeighted(img_copy, 0.6, img, 0.4, 1)
    
    plt.figure(figsize=figsize)
    plt.imshow(out_2)
    plt.show()

    
# Funtions related to finding extremity

def find_thinnest_part(polygon):
    """Returns the coordinates of the middle point of the thinnest part of the polygon

    Parameters
    ----------
    polygon : list
        list of coordinates of the polygon
    """
    shortest_dist = 10000
    for point_A in polygon:
        for point_B in polygon: 
            dist = sqrt( (point_B[0] - point_A[0])**2 + (point_B[1] - point_A[1])**2 )
            if dist != 0 and dist < shortest_dist:
                shortest_dist = dist
                shortest_point_A = point_A
                shortest_point_B = point_B
                mid_point_A_B = [(point_A[0] + point_B[0])/2, (point_A[1] + point_B[1])/2]
    return mid_point_A_B


def find_lowest_point(polygon):
    """Returns the coordinates of the lowest point of the polygon

    Parameters
    ----------
    polygon : list
        list of coordinates of the polygon
    """
    max_y = 0
    for x, y in polygon:
        if y > max_y : 
            max_y = y 
            lowest_point = [x,y]
    return lowest_point
