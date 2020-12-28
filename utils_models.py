import warnings
warnings.filterwarnings("ignore")

import sys,os
import logging
import cv2
import json
import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

def run_yolo_model(IMAGES_PATH):
    os.system('models/yolov5/detect.py --weights models/yolov5/runs/exp1_eleven_50/weights/best.pt --img 1280 --conf 0.4 --source IMAGE_PATH --out outputs --save-txt')
    
def run_mask_model(IMAGES_PATH):
    logging.info("Making predictions with Mask_RCNN.")
    #os.chdir("models/mask_rcnn/samples")
    MODEL_DIR = "../models/mask_rcnn/logs"
    from models.mask_rcnn.samples.construction import construction
    from models.mask_rcnn.mrcnn.utils import Dataset
    from models.mask_rcnn.samples.construction.construction import ConstructionConfig
    from models.mask_rcnn.mrcnn import model as modellib
    # Inference config
    ## Inherits our custom ConstructionConfig 
    logging.info("Instantiating Mask model with inference config.")
    class InferenceConfig(ConstructionConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    # Define inference configuration
    test_config = InferenceConfig()
    # Create model in inference mode
    DEVICE = "/cpu:0" 
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(
            mode="inference", 
            model_dir=MODEL_DIR,
            config=test_config
        )
    # Load weights of your final model
    logging.info("Loading pre-trained model for 30 epochs.")
    weights_path = "../models/mask_rcnn/logs/mask_rcnn_construction_0030.h5"
    logging.info("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    # Create and prepare the dataset 
    logging.info("Preparing the data.")
    dataset_test = Dataset()
    ## Add classes
    dataset_test.add_class("construction", 1, "Vertical_formwork")
    dataset_test.add_class("construction", 2, "Concrete_pump_hose")                   
    ## Add images
    IMG_NAMES = sorted(os.listdir(IMAGES_PATH))
    image_paths = [os.path.join(IMAGES_PATH, image_name) for image_name in IMG_NAMES]
    i = 0
    for image_path in image_paths: 
        if ".ipynb.checkpoints" in image_path:
            print(image_path)
            continue
        img = plt.imread(image_path)
        height = img.shape[0]
        width = img.shape[1]
        dataset_test.add_image("construction", i, image_path, width=width,height=height)
        i += 1
    dataset_test.prepare()
    # Make your predictions
    ## Image per image prediction
    ## Otherwise we had errors (even when changing config to BATCH_SIZE=100)
    results = []
    logging.info("Rendering prediction results.")
    for image_id in dataset_test.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset_test, 
            test_config, 
            image_id, 
            use_mini_mask=False
        )
        result = model.detect([image], verbose=1)
        ## Keep original image shape
        ## To resize square image prediction boxes to actual image size
        result[0]["image_orig_shape"] = (
            dataset_test.image_info[image_id]["height"],
            dataset_test.image_info[image_id]["width"]
        )
        results.extend(result)
    return results