"""
    This file will contain a set of useful functions that we will use in our project
"""


import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import cv2
import numpy as np
import os
random.seed(300)


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

def show_image(img_path, figsize=(10, 10)):
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
    figsize = (10*n, 10)
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
    class_colors = {'Vertical_formwork': (255, 0, 255),
                    'People': (255, 255, 85),
                    'Mixer_truck': (0, 233, 0),
                    'Concrete_pump_hose': (0, 0, 255)}
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
            cv2.fillPoly(img, pts=[np.array(obj.get('points'
                                                    ).get('exterior'))],
                         color=class_colors[obj.get('classTitle')])
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
    figsize = (20*n, 20)
    _, axs = plt.subplots(1, n, figsize=figsize)
    images_paths = np.array(sorted(glob.glob(f"{images_dir}*jpg")))
    labels_paths = np.array(sorted(glob.glob(f"{labels_dir}*json")))
    index_sample = random.choices(range(len(images_paths)), k=n)
    print(index_sample)
    for index, data in enumerate(zip(images_paths[index_sample], labels_paths[index_sample])):
        img = draw_labels(data[0], data[1])
        axs[index].imshow(img)
    plt.show()


def load_labels_from_folder(label_paths):
    """
    Load and pre-format labels to feed to ConstructionDataset.
    folder: folder to retrieve the labels from
    """
    labels = list()
    for label_path in label_paths:
        with open(label_path) as json_data:
            lbl = reformat_label(json.load(json_data), label_path)
            if lbl is not None:
                labels.append(lbl)
    return labels


def reformat_label(label, label_path):
    """Reformat label to correspond to VGG annotator standard.

    Parameters
    ----------
    label : json
        label json file
    label_path : str
        path to the label file

    Returns
    -------
    set
        output dictionary
    """
    objects = label["objects"]
    filename = os.path.split(label_path)[-1] # retrieve filename from full path
    object_list = []
    objects = list(filter(is_polygon, objects))
    for i in range(len(objects)):
        object_list.append(reshape_poly_coordinates(objects[i]))
    return {
            'filename': filename,
            'objects': {"region_attributes": {},
                        "shape_attributes": object_list},
            'size': label["size"]
            }


def is_polygon(obj):
    """Boolean determining whether an object is a polygon or not

    Parameters
    ----------
    obj : dic
        a label object pertaining to an image

    Returns
    -------
    bool
        True if obj is a polygon
    """
    if obj['geometryType'] == 'polygon':
        return True
    else:
        return False


def reshape_poly_coordinates(object_label):
    """Reshape polygons points to generate the mask.

    Parameters
    ----------
    object_label : dict
        object from the json labels file's objects list

    Returns
    -------
    dict
        reformated polygons
    """
    ext_points = object_label["points"]["exterior"]
    return {
            "all_points_x": [x[0] for x in ext_points],
            "all_points_y": [x[1] for x in ext_points],
            "name": "polygon",
            "class": object_label["classTitle"]
        }


def filter_on_image_id(image_info_list, image_id):
    """
    Return the image_info dictionary corresponding to the specified image_id.
    """
    all_ids = [my_dict["id"] for my_dict in image_info_list]
    index = all_ids.index(image_id)
    if image_info_list[index]:
        return image_info_list[index]
    else:
        return None


def draw_polygon(image_dict:  dict):
    """return an image with drawn polygons

    Parameters
    ----------
    image_dict : dict
        image data

    Returns
    -------
    np.ndarray
        output image
    """
    class_colors = {'Vertical_formwork': (255, 0, 255),
                    'Concrete_pump_hose': (0, 0, 255)}
    img = plt.imread(image_dict["path"])
    img_copy = img.copy()
    for shape in image_dict["shapes"]:
        points = np.array(tuple([x, y] for x, y in zip(shape["all_points_x"], shape["all_points_y"])))
        cv2.fillPoly(img, pts=[points], color=class_colors[shape.get('class')])
        out = cv2.addWeighted(img_copy, .6, img, 0.4, 1)
    plt.imshow(out)
    return out