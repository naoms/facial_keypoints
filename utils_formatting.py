"""
    This file will contain a set of useful functions that we will use in our project to download and format our data
"""

import os
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter
from datetime import datetime
import random
import cv2
import numpy as np
import pandas as pd

DIR_IMAGES = '../inputs/images'
DIR_LABELS = '../inputs/labels'

random.seed(300)

# Functions related to downloading data

def get_name(file_path):
    """extract the file name from the path

    Parameters
    ----------
    file_path : string
        file path

    Returns
    -------
    string
        file name
    """
    return file_path.name.split('/')[-1]


def download_file(container, blob_path, destination_dir):
    """This method copy a single file from online container localy
    in destination_dir. The nameon the file is kept

    Parameters
    ----------
    container : azure.storage.blob.ContainerClient
        online container where taget file is stored
    blob_path : string
        online file path
    destination_dir : string
        destination folder path
    """
    blob_name = get_name(blob_path)
    Path(f"{destination_dir}/{blob_name}").parent.mkdir(
        parents=True, exist_ok=True)
    with open(f"{destination_dir}/{blob_name}", "wb") as data:
        download_stream = container.get_blob_client(blob_path).download_blob()
        data.write(download_stream.readall())


def download_files(container, destination_dir):
    """This fuction copies all the files stored in a Azure
    online container and strore them in destination_dir

    Parameters
    ----------
    container : azure.storage.blob.ContainerClient
        online container where taget file is stored
    destination_dir : string
        destination folder path
    """
    blobs = list(container.list_blobs())
    for blob in blobs:
        download_file(container, blob, destination_dir)



# Functions related to table formating

def load_json_into_table(folder, nb_labels = -1):
    """Creates a dataframe with all the data from a folder with json files

    Parameters
    ----------
    folder: str
        folder to retrieve the labels from
    nb_labels: int
        number of labels to retrieve

    Returns
    -------
    pd.DataFrame
        dataframe with all the data from the json files in the folder
    """
    if nb_labels == -1 :
        nb_labels = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
    list_data=[]
    i = 0
    for filename in os.listdir(folder):
        if i <= nb_labels-1:
            i +=1
            with open(os.path.join(folder,filename)) as json_data:
                lbl = json.load(json_data)
            if lbl is not None:
                for obj in lbl.get('objects'):
                    ext_points = obj.get('points').get('exterior')
                    int_points = obj.get('points').get('interior')
                    nb_ext = len(ext_points) if ext_points is not None else 0 
                    nb_int = len(int_points) if ext_points is not None else 0 
                    row = {
                        'filename' : filename,
                        'height' : lbl.get('size').get('height'),
                        'width' : lbl.get('size').get('width'),
                        'geometryType' : obj.get('geometryType'), 
                        'classTitle' : obj.get('classTitle'),
                        'ext_points' : list(ext_points),
                        'int_points' : list(int_points),
                        'nb_exterior' : nb_ext,
                        'nb_interior' : nb_int
                    }
                    list_data.append(row)
        data_df = pd.DataFrame(list_data) 
    return data_df

def get_ext_points(row):
    if row['classTitle'] in ['People', 'Mixer_truck']:
        return row['bbox']
    elif row['classTitle'] in ['Vertical_formwork', 'Concrete_pump_hose']:
        return row['segmentation']
    
def unflatten_by_2(lst):
    res = []
    for i, el in enumerate(lst):
        if i%2 == 1: 
            continue
        else:
            res.append([lst[i], lst[i+1]])
    return res

def get_len(row):
    if row['ext_points'] is not None:
        return len(row['ext_points'])
    else:
        return 0
def load_coco_json_into_table(file_path):
    """Creates a dataframe with all the data from a folder with json files

    Parameters
    ----------
    file_path: str
        file_path to retrieve the labels from
    nb_labels: int
        number of labels to retrieve

    Returns
    -------
    pd.DataFrame
        dataframe with all the data from the json files in the folder
    """
    with open(os.path.join('',file_path)) as json_data:
        lbl = json.load(json_data)
    if lbl is not None:
        list_category=[]
        for category in lbl.get('categories'):
            row = {
                'id' : category.get('id'),
                'name' : category.get('name')
            }
            list_category.append(row)
        category_df = pd.DataFrame(list_category)
        
        list_images=[]
        for image in lbl.get('images'):
            row = {
                'id' : image.get('id'),
                'filename' : image.get('file_name').split('/')[5],
                'height' : image.get('height'),
                'width' : image.get('width')
            }
            list_images.append(row)
        images_df = pd.DataFrame(list_images)
        
        list_annotations=[]
        for annotation in lbl.get('annotations'):
            if (annotation.get('segmentation') not in [None, []] ):
                segmentation = unflatten_by_2(annotation.get('segmentation')[0])
                #We keep only the first element of the segmentation as in our case we only have one polygon per object
            else:
                segmentation = None
            if (annotation.get('bbox') not in [None, []]):
                bbox = unflatten_by_2(annotation.get('bbox'))
            else:
                bbox = None
            row = {
                'segmentation' : segmentation, 
                'bbox' : bbox,
                'image_id' : annotation.get('image_id'),
                'category_id' : annotation.get('category_id')
            }
            list_annotations.append(row)
        annotations_df = pd.DataFrame(list_annotations)
        
        df = annotations_df.merge(category_df, left_on='category_id', right_on='id')\
                            .merge(images_df, left_on='image_id', right_on='id')
        df_final  = df[['filename', 'height', 'width', 'name', 'bbox', 'segmentation']]
        df_final = df_final.rename(columns = {'name':'classTitle'})
        df_final['ext_points'] = df_final.apply(lambda row : get_ext_points(row), axis=1)
        df_final['nb_exterior'] = df_final.apply(lambda row : get_len(row), axis=1)
        
        
    return df_final

def get_date_time(row):
    date = row['filename']
    if date.split('-')[0]=='2020':
        return datetime.strptime(date[:-4], '%Y-%m-%d-%H-%M-%S')
    
def get_year(row): 
    date = row['filename'].split('_')
    if date[0]=='2020':
        return date[0]
    
def get_date(row):
    date = row['filename']
    if date.split('-')[0]=='2020':
        return row['date_time'].date()
        
def get_time(row):
    date = row['filename']
    if date.split('-')[0]=='2020':
        return row['date_time'].time()

def add_date_time(data_df):
    data_df['date_time'] = data_df.apply(lambda row : get_date_time(row), axis=1)
    data_df['date'] = data_df.apply(lambda row : get_date(row), axis=1)
    data_df['time'] = data_df.apply(lambda row : get_time(row), axis=1)
    
    #Sort dataframe by time
    data_df = data_df.sort_values(by='date_time')
    
    return data_df
