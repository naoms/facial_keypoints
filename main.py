import sys
import logging
## Sarah imports
import tensorflow as tf
import os
import matplotlib.pyplot as plt

def main(IMAGES_PATH, LABELS_PATH, action='mask'):
    if action == 'data_prep':
        pass
    elif action == 'yolo' :
        pass
        #inference yolo on images and labels
        #return_yolo_ouput
        
    elif action == 'mask' :
        logging.info("Making predictions with Mask_RCNN.")
        #os.chdir("models/mask_rcnn/samples")
        MODEL_DIR = "models/mask_rcnn/logs"
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
        weights_path = "models/mask_rcnn/logs/mask_rcnn_construction_0030.h5"
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
    elif action == 'model1' :
        pass
        #inference yolo on images and labels
        #inference mask on images and labels
        #aggregate output yolo and output mask
        #returns aggregation --> loads output to json

    elif action == 'model2' :
        pass
        #1. read_coco_format_json
        #2. Get_all extremities
        #3. Get_the extented_ working_zone
        #4. Get the workers_status 
        #5. Get final table --> returns final table with efficiency ratio
        #6. Saves graphs somewhere
        
    print(action)
    
    
if __name__ == "__main__":
        possible_actions = ['data_prep', 'yolo', 'mask', 'model1', 'model2']
        if len(sys.argv)<3:
            print("Please input main.py <IMAGES_PATH> <LABELS_PATH>")
            print("You can also add an <ACTION>")
        elif len(sys.argv)>4:
            print("Please input main.py <IMAGES_PATH> <LABELS_PATH> <ACTION>")
        else:
            IMAGES_PATH = sys.argv[1]
            LABELS_PATH = sys.argv[2]
            if (len(sys.argv)==4 and sys.argv[3] in possible_actions ) :
                action = sys.argv[3]
                main(IMAGES_PATH, LABELS_PATH, action)
            elif len(sys.argv)==3: 
                main(IMAGES_PATH, LABELS_PATH)
            elif len(sys.argv)==4 and sys.argv[3] not in possible_actions:
                print(f"{sys.argv[3]} is not an action. The parameter action can only be : data_prep, yolo, mask, model1 or model2")