import os
import re
import datetime
import numpy as np
from itertools import groupby
from skimage import measure
from PIL import Image
from pycocotools import mask
import json
import glob



category_info = {0: {"category_id": 1, "iscrowd": 0, 'id':0},
                 1: {"category_id": 2, "iscrowd": 0, 'id':1},
                 2: {"category_id": 3, "iscrowd": 0, 'id':2},
                 3: {"category_id": 4, "iscrowd": 0, 'id':3}}


convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

def resize_binary_mask(array, new_size):
    image = Image.fromarray(np.array(array).astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def create_image_info(image_id, file_name, image_size, 
                      ):

    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
    }

    return image_info


def create_class_info(classes_list
                      ):
    res = []
    for i in range(len(classes_list)):
        class_info = {
            "Supercategory": None,
            "name": classes_list[i],
            "id": i
                }
        res.append(class_info)
    return res



def create_annotation_info(annotation_id, image_id, category_info, binary_mask,
                           image_size=None, tolerance=2, bounding_box=None):

    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask, image_size)

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    if area < 1 and bounding_box is None:
        return None

    if bounding_box is None:
        bounding_box = np.int32(mask.toBbox(binary_mask_encoded))


    if category_info["iscrowd"]:
        is_crowd = 1
        segmentation = binary_mask_to_rle(binary_mask)
    else :
        is_crowd = 0
        segmentation = binary_mask_to_polygon(binary_mask, tolerance)

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": is_crowd,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
    }

    return annotation_info



def to_coco(yolo_file, rcnn_file, image_id, category_info,anno_id):
    img_size = rcnn_file["orig_image_shape"] #rcnn_file['size']
    h, w = img_size[0],img_size[1]
    lines=yolo_file.readlines()
    res = []
    for line in lines:
        line_l = line.split(" ")
        label = int(line_l[0])
        bbx = np.array([int((float(line_l[1]) - float(line_l[3])/2)*w) , int((float(line_l[2]) + float(line_l[3])/2)*h),
               int((float(line_l[1]) + float(line_l[3])/2)*w ), int((float(line_l[2]) - float(line_l[3])/2)*h)])
        res.append(create_annotation_info(anno_id, image_id, category_info[label], [[]],
                               image_size=img_size, bounding_box=bbx))
        anno_id += 1
    for i in range(len(rcnn_file["class_ids"])):
        label = rcnn_file["class_ids"][i] + 1
        # print(np.array(rcnn_file["masks"])[:, :, i].sum())
        # print(np.array(rcnn_file["masks"])[:, :, i])
        a = create_annotation_info(anno_id, image_id, category_info[label], np.array(rcnn_file["masks"])[:, :, i],
                               image_size=img_size)
        # print(a)
        if a is not None:
            res.append(a)
        anno_id += 1
    return res


def run_to_coco_folder(img_dir, yolo_dir, mrcnn_dir, category_info):
    images_paths = sorted(glob.glob(f"{img_dir}/*jpg"))
    yolo_paths= sorted(glob.glob(f"{yolo_dir}/*txt"))
    mrcnn_paths = sorted(glob.glob(f"{mrcnn_dir}/*json"))

    print(len(images_paths), len(yolo_paths), len(mrcnn_paths))
    out = {"categories": [],
           "info": {},
           "licences": [],
           "images": [],
           "annotations": []}

    out["categories"] = create_class_info(["Truck",
    "People", "Vertical_formwork", "Concrete_pump_hose"])
    out['info'] = {"description": "Chronsite dataset", "year": 2020}
    anno_id = 0
    for img_id in range(len(images_paths)):
        yolo = yolo_paths[img_id]
        mrcnn = mrcnn_paths[img_id]

        yolo_file = open(yolo, "r")
        with open(mrcnn, 'r') as file:
            rcnn_file = json.load(file)
        # print(np.sum(rcnn_file["masks"]))
        img_size = rcnn_file["orig_image_shape"] #( 1080, 1080) rcnn_file['size']
        out["images"].append(create_image_info(img_id, images_paths[img_id], img_size))
        out["annotations"].extend(to_coco(yolo_file, rcnn_file, img_id, category_info, anno_id))
    return out

if __name__ == "__main__":
    output_coco_dir = "../inputs/"
    anno_id = 0
    category_info = {0: {"category_id": 1, "iscrowd": 0, 'id':0},
                 1: {"category_id": 2, "iscrowd": 0, 'id':1},
                 2: {"category_id": 3, "iscrowd": 0, 'id':2},
                 3: {"category_id": 4, "iscrowd": 0, 'id':3}}
    # adapter les inputs de run_to_coco_folder
    out = run_to_coco_folder("../../inputs/datasets/images/Test/", "test_labels/", "Mask_RCNN/samples/construction", category_info)

    with open(f'{output_coco_dir}pred_eleven_g3.json', 'w') as json_file:
        json.dump(out, json_file)
