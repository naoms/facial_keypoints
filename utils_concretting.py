import sys
import os
import cv2
import json
import pandas as pd
import random
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import sqrt
sys.path.insert(0, "..")
#from utils import show_image_labeled_sample, show_image_sample,load_json_into_table, draw_labels
import shutil
import glob
import numpy as np
from shapely.geometry import Point, Polygon

def is_concreting_day(labels_df):
    """
    Gives the labels dataframe grouped by date and with a new column "has_concrete_pump" indicating 
    if there was some concreting work done on this particular day.
    
    Parameters
    ----------
    labels_df : df
        input dataframe gathering the labels information of the images 

    Returns
    -------
    dataframe
        output the labels dataframe grouped by dates and with an additional has_concreting_pump column
    """
    labels_df["has_concreting_pump"] = labels_df.classTitle.apply(lambda x : x=="Concrete_pump_hose")*1
    return pd.DataFrame(labels_df.groupby("date").mean()["has_concreting_pump"]>0)

def get_concreting_df(labels_df):
    """
    Returns the labels dataframe keeping only the days when there was concreting work.
    Parameters
    ----------
    labels_df : df
        input dataframe gathering the labels information of the images 

    Returns
    -------
    dataframe
        output the labels dataframe containing only rows corresponding to days with concreting work.
    """    
    concreting_days = is_concreting_day(labels_df)
    concreting_days_indexes = concreting_days[concreting_days["has_concreting_pump"]==True].index
    concreting_df = labels_df[labels_df.date.apply(lambda x: x in concreting_days_indexes)]
    return concreting_df

def delete_duplicated_pumps(concreting_df):
    """
    Allows to keep only one concrete pump object per image
    Parameters
    ----------
    concreting_df : df
        dataframe gathering information on images that correspond to concreting days

    Returns
    -------
    dataframe
        output the same dataframe but containing only one concrete pump object per image
    """
    date_list = list(concreting_df[concreting_df["has_concreting_pump"]>=2]["date_time"])
    df_no_duplicates = concreting_df[concreting_df["date_time"].isin(date_list)]
    df_no_duplicates = df_no_duplicates[["filename", "nb_exterior"]].groupby("filename", as_index=False).agg({"nb_exterior":"min"}, as_index=False)
    df_no_duplicates["pumps_to_delete"] = 0
    concreting_df = pd.merge(concreting_df, df_no_duplicates, left_on=["filename", "nb_exterior"], right_on=["filename", "nb_exterior"], how='left')
    concreting_df = concreting_df[concreting_df["pumps_to_delete"]!=0]
    concreting_df.drop("pumps_to_delete", axis=1,inplace=True)
    concreting_df = concreting_df.groupby("date_time", as_index=False).sum()
    return concreting_df

def get_change_in_concreting(concreting_df):
    """
    Allows to see when the concrete pump appears and when it disappears on successive images to identify the
    time window of the concreting work. 
    
    Parameters
    ----------
    concreting_df : dataframe
        dataframe gathering information on images that correspond to concreting days

    Returns
    -------
    dataframe
        output the same dataframe with an additional column called next_timestep_has_pump and indicating if the 
        concreting pump is still on the image or if it has disappeared (meaning concreting work is finished)
    """
    concreting_df["next_timestep_has_pump"] = concreting_df["has_concreting_pump"].shift(-1)
    concreting_df["next_timestep_has_pump"].fillna(0, inplace=True)
    concreting_df["change_in_concreting"] = concreting_df["has_concreting_pump"]*2 - concreting_df["next_timestep_has_pump"]
    return concreting_df

def get_start_time_list(concreting_df):
    """
    Gives a list that has same length as concreting_df of the starting time of concreting periods. 
    Parameters
    ----------
    concreting_df : dataframe
        dataframe gathering information on images that correspond to concreting days

    Returns
    -------
    list
        containing datetime objects
    """
    global date_time_list
    global change_concreting_list
    global start_time_list
    date_time_list = list(concreting_df["date_time"])
    change_concreting_list = list(concreting_df["change_in_concreting"])
    start_time_list = [0] * len(date_time_list)
    start_time_list[0] = date_time_list[0]

    for i in range(1,len(start_time_list)):
        if change_concreting_list[i] >= 1:
            start_time_list[i] = start_time_list[i-1]
        elif change_concreting_list[i] == 0:
            start_time_list[i] == 0
        elif change_concreting_list[i] == -1:
            start_time_list[i] = date_time_list[i+1]
    return start_time_list


def get_end_time_list():
    """
    Gives a list that has same length as concreting_df of the end time of concreting periods. 

    Returns
    -------
    list
        containing datetime objects
    """
    global end_time_list
    end_time_list = [0] * len(date_time_list)
    last_num = len(end_time_list)-1
    end_time_list[last_num] = date_time_list[last_num]

    for i in range(1, len(end_time_list)):
        if change_concreting_list[last_num-i] == 2:
            end_time_list[last_num-i] = date_time_list[last_num-i]
        else:
            end_time_list[last_num-i] = end_time_list[last_num-i+1]
    return end_time_list


def get_df_with_periods(concreting_df):
    """
    Gives a dataframe that repeats the start and end time of the concreting period for each row corresponding to 
    the same concreting work.
    
    Parameters
    ----------
    concreting_df : dataframe
        dataframe gathering information on images that correspond to concreting days

    Returns
    -------
    dataframe
        output the same dataframe with two additional columns "start_time" and "end_time".
    """
    pump_list = list(concreting_df["has_concreting_pump"])
    for i in range(len(pump_list)):
        if pump_list[i] == 0:
            start_time_list[i] = 0
            end_time_list[i] = 0
    concreting_df["start_time"] = start_time_list
    concreting_df["end_time"] = end_time_list
    return concreting_df


def mapping_concreting_periods(concreting_df):
    """
    Gives a dataframe mapping each start-end time couples to a period number (1,2,3,etc)
    
    Parameters
    ----------
    concreting_df : dataframe
        dataframe gathering information on images that correspond to concreting days

    Returns
    -------
    dataframe
        output a df with 2 columns period and start_time
    """
    concreting_periods = concreting_df[["start_time"]].drop_duplicates()
    concreting_periods["start_time"] = [str(x) for x in list(concreting_periods["start_time"])]
    concreting_periods =concreting_periods.sort_values("start_time").reset_index(drop=True)
    concreting_periods["period"] = [i for i in range(len(concreting_periods))]
    return concreting_periods

def get_complete_concreting_df(labels_df, concreting_df, concreting_periods):
    """
    Gives a dataframe with all the information of the labels_df, including all days (even those with no concreting work) 
    and all types of objects (not only concrete pumps).
    
    Parameters
    ----------
    labels_df : dataframe
        input dataframe gathering information on images
    concreting_df: dataframe
        dataframe gathering information on images that correspond to concreting days 
    concreting_periods: dataframe
        mapping each start-end time couples to a period number

    Returns
    -------
    dataframe
        output labels_df but including new columns "start_time", "end_time" and period"
    
    """
    concreting_df["start_time"] = [str(x) for x in list(concreting_df["start_time"])]
    concreting_df = pd.merge(concreting_df, concreting_periods, left_on=["start_time"], right_on=["start_time"], how="left")
    concreting_df = concreting_df[["date_time", "start_time", "end_time", "period"]]
    complete_concreting_df = pd.merge(labels_df, concreting_df, left_on=["date_time"], right_on=["date_time"], how="left").fillna(0)
    return complete_concreting_df

def find_thinnest_part(polygon): 
    shortest_dist = 1000
    for point_A in polygon:
        for point_B in polygon: 
            dist = sqrt( (point_B[0] - point_A[0])**2 + (point_B[1] - point_A[1])**2 )
            if dist != 0 and dist < shortest_dist:
                shortest_dist = dist
                shortest_point_A = point_A
                shortest_point_B = point_B
                mid_point_A_B = [(point_A[0] + point_B[0])/2, (point_A[1] + point_B[1])/2]
    return mid_point_A_B


def get_df_with_extremities(complete_concreting_df):
    """
    Applies a function to find the extremity of the concrete pump and include this information in the dataframe
    
    Parameters
    ----------
    complete_concreting_df : df
        dataframe gathering information on images that correspond to concreting days

    Returns
    -------
    dataframe
        output a df keeping only rows corresponding to concrete pump hose and people objects and with a
        new column giving the extremity of the pump when relevant.
    
    """
    global df_pump
    df_pump = complete_concreting_df[complete_concreting_df['classTitle'] == 'Concrete_pump_hose']
    df_pump["extremity_point"] = df_pump['ext_points'].apply(find_thinnest_part)
    complete_concreting_df = complete_concreting_df[complete_concreting_df["classTitle"].isin(["Concrete_pump_hose", "People"])]
    complete_concreting_df = pd.merge(complete_concreting_df, df_pump[["filename", "extremity_point"]], left_on=["filename"], right_on=["filename"], how="left").fillna(0)
    return complete_concreting_df


def get_pump_polygon_for_concreting_period_df(df_pump):
    """
    Creates a list of extrmity points in a list of concreting period and transform it in a dataframe. 
    
    Parameters
    ----------
    df_pump : df
        input dataframe keeping only concrete pump hose objects

    Returns
    -------
    dataframe
        output a dataframe with two columns, one for the concreting period and one gathering all the extremity pump points
        in a list
    
    """
    period_list = list(df_pump["period"])
    global period_unique
    period_unique = list(df_pump["period"].unique())
    extremities_list = list(df_pump["extremity_point"])
    global all_points_list
    all_points_list = [[]]*len(period_unique)
    
    for i in range(len(period_list)):
        if period_list[i] != 0:
            sub_list = period_unique.index(period_list[i])
            list_to_fill = all_points_list[sub_list].copy()
            list_to_fill.append(extremities_list[i])
            list_to_fill = list_to_fill
            all_points_list[sub_list] = list_to_fill
    pump_polygon_df = pd.DataFrame(data={"period":period_unique, "pump_polygon":all_points_list})
    return pump_polygon_df

def get_max_min_coordinates(my_dict, key):
    """
    Retrieves the farthest points among all the extrimity points found for the concreting period.
    
    Parameters
    ----------
    my_dict : dictionary
        dictionary of concreting period and associated with extrimity points of the concrete pump

    Returns
    -------
    key and list
        the key an a list of four points
    
    """
    my_series = my_dict[key]
    x_points = [x[0] for x in my_series] 
    y_points = [x[1] for x in my_series]
    x_min, x_max = np.min(x_points), np.max(x_points)
    y_min, y_max = np.min(y_points), np.max(y_points)
    bottom_left = tuple((x_min, y_min))
    bottom_right = tuple((x_max, y_min))
    top_left = tuple((x_min, y_max))
    top_right = tuple((x_max, y_max))
    return key,[bottom_left, bottom_right, top_left, top_right]


def get_quadrilateral_concreting_zone_df(all_points_list):
    """
    Retrieves the farthest points among all the extrimity points found for the concreting period.
    
    Parameters
    ----------
    all_points_list : list of list
        list of all extrimity points for each concreting period

    Returns
    -------
    dataframe
        with 2 columns: "period" corresponding to the concreting work period and "pump_polygon" corresponding to the 
        rectangle created with the extrimity points of the pump.
    
    """
    period_extremities_dict = {period_unique[i]: all_points_list[i] for i in range(len(period_unique))}
    list_quad = []
    quad_dict = {}
    for i in range(len(period_unique)-1):
        key,points = get_max_min_coordinates(period_extremities_dict, i+1)
        quad_dict[key] = points
    pump_rectangle_df = pd.DataFrame(list(quad_dict.keys()))
    pump_rectangle_df["points"] = pd.Series(list(quad_dict.values()))
    pump_rectangle_df = pump_rectangle_df.rename(columns={0:"period", "points":"pump_polygon"})
    return pump_rectangle_df

def get_final_concreting_df(complete_concreting_df, pump_rectangle_df):
    """
    Retrieves the farthest points among all the extrimity points found for the concreting period.
    
    Parameters
    ----------
    complete_concreting_df : dataframe
        dictionary of concreting period and associated with extrimity points of the concrete pump
    
    pump_rectangle_df: dataframe
        dataframe including a column with points delimiting the rectangle concreting zone

    Returns
    -------
    dataframe
        df including all columns related to the concreting period and to the concreting zone. 
    
    """
    final_concreting_df = pd.merge(complete_concreting_df, pump_rectangle_df, left_on=["period"], right_on=["period"], how="left").fillna(0)
    return final_concreting_df


# Finding the center of the polygon or of the worker 

def centroid(polygon):
    """Finds the coordinates of the center of some coordinates

    Parameters
    ----------
    polygon: list
        concreting zone delimitation 
    
    Returns
    -------
    list
        coordinates of the center 
    """
    x_list = [x[0] for x in polygon]
    y_list = [y[1] for y in polygon]
    len_polygon = len(polygon)
    x = sum(x_list) / len_polygon
    y = sum(y_list) / len_polygon
    center = [x,y]
    return center

def extended_polygon(polygon):
    """Finds the working zone for one concreting zone 

    Parameters
    ----------
    polygon: list
        concreting zone delimitation 
    
    Returns
    -------
    list
        working zone delimitation 
    """
    center = centroid(polygon) 
    extended_poly = [] 
    for coord in polygon:
        new_x = round(2*coord[0] - center[0])
        new_y = round(2*coord[1] - center[1])
        extended_poly.append([new_x, new_y]) 
    return extended_poly


def building_extended_polygon(df):
    """Builds a new column containing the working zone 

    Parameters
    ----------
    df: pd.DataFrame
        whole dataframe with one object per row 
    
    Returns
    -------
    pd.DataFrame
        same dataframe with an additional column for the working zone 
    """
    df = df[df['period'] !=0].reset_index(drop=True)
    list_pump_polygon = list(df['pump_polygon'])
    global extended_poly
    extended_poly = []
    for pump_poly in list_pump_polygon:
        extended_poly.append(extended_polygon(pump_poly))
    df['extended_polygon'] = extended_poly
    return df
    

def worker_working_or_not(df):
    """Finds if the worker is in or outside of the working zone 

    Parameters
    ----------
    df: pd.DataFrame
        whole dataframe with one object per row 
    
    Returns
    -------
    pd.DataFrame
        same dataframe with an additional column for the working status 
    """
    list_objects = list(df['classTitle'])
    list_coords = list(df['ext_points'])
    # extended_poly : extracted from the function above 
    list_working = []
    for i in range(len(list_objects)):
        if list_objects[i] != 'People':
            list_working.append('NA')
        else:
            worker = Point(centroid(list_coords[i]))
            polygon = Polygon(extended_poly[i])
            if worker.within(polygon) == True:
                list_working.append('Working')
            else:
                list_working.append('Not_working')
    df['working_status'] = list_working
    return df

# Functions used in main.py

def get_concreting_periods(labels_df):
    """Builds a table with delimitated concreting periods in time

    Parameters
    ----------
    labels_df: pd.DataFrame
        dataframe with all objects identified, date and time of the pictures  
    
    Returns
    -------
    pd.DataFrame
        dataframe with unique code corresponding to each concreting work
    """    
    concreting_df = get_concreting_df(labels_df)
    concreting_df = delete_duplicated_pumps(concreting_df)
    concreting_df = get_change_in_concreting(concreting_df)
    start_time_list = get_start_time_list(concreting_df)
    end_time_list = get_end_time_list()
    concreting_df = get_df_with_periods(concreting_df)
    concreting_periods = mapping_concreting_periods(concreting_df)
    complete_concreting_df = get_complete_concreting_df(labels_df, concreting_df, concreting_periods)
    return complete_concreting_df

def get_concreting_zone_and_workers_infos(complete_concreting_df):
    """For each concreting work, delimitation of the concreting zone, working zone and status of the workers 

    Parameters
    ----------
    complete_concreting_df: pd.DataFrame
        dataframe with unique code corresponding to each concreting work
    
    Returns
    -------
    pd.DataFrame
        complete dataframe with the status eg working on concreting or not of each worker 
    """  
    complete_concreting_df = get_df_with_extremities(complete_concreting_df)
    df_pump = complete_concreting_df[complete_concreting_df['classTitle'] == 'Concrete_pump_hose']
    df_polygon = get_pump_polygon_for_concreting_period_df(df_pump)
    pump_rectangle_df = get_quadrilateral_concreting_zone_df(all_points_list)
    final_concreting_df = get_final_concreting_df(complete_concreting_df, pump_rectangle_df)
    final_concreting_df = building_extended_polygon(final_concreting_df)
    final_concreting_df = worker_working_or_not(final_concreting_df)
    return final_concreting_df

def efficiency_of_site(df):
    """Builds the final table with efficiency ratio of each concreting site 

    Parameters
    ----------
    df: pd.DataFrame
        whole dataframe with one object per row 
        
    Returns
    -------
    pd.DataFrame
        new dataframe with one concreting work per row 
    """
    df_efficiency = df.copy()
    df_efficiency = df_efficiency[df_efficiency['classTitle']=='People']
    df_efficiency['working_people'] = df_efficiency['working_status'].apply(lambda x: 1 if x=='Working' else 0)
    df_efficiency['non_working_people'] = df_efficiency['working_status'].apply(lambda x: 1 if x=='Not_working' else 0)
    df_efficiency["period"] = [str(x) for x in list(df_efficiency["period"])]
    df_efficiency = df_efficiency[['filename','period', 'working_people', 'non_working_people']].groupby(['filename', 'period']).sum().reset_index()
    df_efficiency = df_efficiency.groupby('period', as_index=False).mean()
    df_efficiency['working_people'] = df_efficiency['working_people'].apply(lambda x: round(x, 1))
    df_efficiency['non_working_people'] = df_efficiency['non_working_people'].apply(lambda x: round(x, 1))
    df_efficiency['efficiency_ratio'] = round(df_efficiency['working_people'] / 
                                              (df_efficiency['working_people'] + df_efficiency['non_working_people']), 2)
    df_efficiency["period"] = [int(float(x)) for x in list(df_efficiency["period"])]
    df_efficiency = df_efficiency.sort_values("period")
    
    # Getting dates for period
    df_merge = df.copy()
    df_merge = df_merge[["period", "start_time", "end_time"]].drop_duplicates()
    df_merge["period"] = [int(x) for x in list(df_merge["period"])]
    
    df_efficiency = pd.merge(df_efficiency, df_merge, left_on=["period"], right_on=['period'], how="left")
    df_efficiency["start_time"] = pd.to_datetime(df_efficiency["start_time"])
    df_efficiency["end_time"] = pd.to_datetime(df_efficiency["end_time"])
    df_efficiency["concreting_time"] = df_efficiency["end_time"] - df_efficiency["start_time"]
    
    df_efficiency = df_efficiency[["period", "start_time", "end_time", "concreting_time", "working_people", "non_working_people", "efficiency_ratio"]]
    
    return(df_efficiency)