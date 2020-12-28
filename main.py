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
sys.path.insert(0, "..")
from utils_formatting import * 
from utils_analytics import * 
from utils_concretting import * 
from utils_models import * 



def main(IMAGES_PATH, action='model2'):
    """
    Gives a list that has same length as concreting_df of the starting time of concreting periods. 
    Parameters
    ----------
    IMAGES_PATH : str
        path of directory where all the images are stored

    Returns
    -------
    pd.DataFrame
        the final dataframe with all the aggregate results
    """
    if action == 'yolo':
        # Run yolo model on images in the given folder
        run_yolo_model(IMAGES_PATH)
        
    elif action == 'mask':
        # Run mask model on images in the given folder
        run_mask_model(IMAGES_PATH)
    
    elif action == 'model1':
        # Run yolo model on images in the given folder
        run_yolo_model(IMAGES_PATH)
        
        # Run mask model on images in the given folder
        run_mask_model(IMAGES_PATH)
        
        # Aggregate output of yolo and output of mask
        #yolo_coco.py

    elif action == 'model2':
        # For sake of time saving, we will run model2 on a previously saved table instead of the output of model1. 
    
        #data_df = load_coco_json_into_table('yolo_mask_to_coco/Predictions_group3.json')
        #data_df = add_date_time(data_df)
        
        data_df = pd.read_pickle('inputs/table_labels_new.pkl')
        
        data_df.reset_index(drop=True, inplace=True)
        
        complete_concreting_df = get_concreting_periods(data_df)
        
        final_concreting_df = get_concreting_zone_and_workers_infos(complete_concreting_df)
        
        df_efficiency = efficiency_of_site(final_concreting_df)
        
        return df_efficiency
    
    
if __name__ == "__main__":
        possible_actions = ['yolo', 'mask', 'model1', 'model2']
        if len(sys.argv)<2:

            logging.info("Please input main.py <IMAGES_PATH>")
            logging.info("You can also add an <ACTION>")
        elif len(sys.argv)>3:
            logging.info("Please input main.py <IMAGES_PATH> <ACTION>")
        else:
            IMAGES_PATH = sys.argv[1]
            if (len(sys.argv)==3 and sys.argv[2] in possible_actions ) :
                action = sys.argv[2]
                df_final = main(IMAGES_PATH, action)
                df_final.to_csv('outputs/final_out.csv')
            elif len(sys.argv)==2: 
                df_final = main(IMAGES_PATH)
                df_final.to_csv('outputs/final_out.csv')
            elif len(sys.argv)==3 and sys.argv[2] not in possible_actions:
                print(f"{sys.argv[3]} is not an action. The parameter action can only be : yolo, mask, model1 or model2")