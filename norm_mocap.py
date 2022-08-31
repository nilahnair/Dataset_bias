# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:26:32 2021

@author: nilah
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import csv_reader

SCENARIO = {'R01': 'L01', 'R02': 'L01', 'R03': 'L02', 'R04': 'L02', 'R05': 'L02', 'R06': 'L02', 'R07': 'L02',
            'R08': 'L02', 'R09': 'L02', 'R10': 'L02', 'R11': 'L02', 'R12': 'L02', 'R13': 'L02', 'R14': 'L02',
            'R15': 'L02', 'R16': 'L02', 'R17': 'L03', 'R18': 'L03', 'R19': 'L03', 'R20': 'L03', 'R21': 'L03',
            'R22': 'L03', 'R23': 'L03', 'R24': 'L03', 'R25': 'L03', 'R26': 'L03', 'R27': 'L03', 'R28': 'L03',
            'R29': 'L03', 'R30': 'L03'}

labels_persons = {"S01": 0, "S02": 1, "S03": 2, "S04": 3, "S05": 4, "S06": 5, "S07": 6, "S08": 7, "S09": 8,
                  "S10": 9, "S11": 10, "S12": 11, "S13": 12, "S14": 13, "S15": 14, "S16": 15}

annotator = {"S01": "A17", "S02": "A03", "S03": "A08", "S04": "A06", "S05": "A12", "S06": "A13",
             "S07": "A05", "S08": "A17", "S09": "A03", "S10": "A18", "S11": "A08", "S12": "A11",
             "S13": "A08", "S14": "A06", "S15": "A05", "S16": "A05"}

repetition = ["N01", "N02"]


def select_columns_opp(data):
    """
    Selection of the columns employed in the MoCAP
    excluding the measurements from lower back,
    as this became the center of the human body,
    and the rest of joints are normalized
    with respect to this one

    :param data: numpy integer matrix
        Sensor data (all features)
    :return: numpy integer matrix
        Selection of features
    """

    #included-excluded
    features_delete = np.arange(66, 72)
    
    return np.delete(data, features_delete, 1)


def statistics_measurements():
    '''
    Computes some statistics over the channels for the entire training data

    returns a max_values, min_values, mean_values, std_values
    '''

    #dataset_path_imu = "/vol/actrec/DFG_Project/2019/LARa_dataset/Motionminers/2019/flw_recordings_12000/"
    dataset_path_imu = "/vol/actrec/DFG_Project/2019/LARa_dataset/MoCap/recordings_2019/16_Annotated_Dataset/"

    persons = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]
    train_ids = ["R01", "R02", "R03", "R04", "R05", "R06","R07", "R08", "R09", "R10", "R11", "R12", "R13", "R14", "R15", "R16", "R17",
                 "R18", "R19", "R20", "R21", "R22", "R23", "R24", "R25", "R26", "R27", "R28", "R29", "R30"]
    
   
    accumulator_measurements = np.empty((0, 126))
    for P in persons:
          for R in train_ids:
                S = SCENARIO[R]
                for N in repetition:
                    annotator_file = annotator[P]
                    if P == 'S07' and SCENARIO[R] == 'L01':
                        annotator_file = "A03"
                    if P == 'S14' and SCENARIO[R] == 'L03':
                        annotator_file = "A19"
                    if P == 'S11' and SCENARIO[R] == 'L01':
                        annotator_file = "A03"
                    if P == 'S11' and R in ['R04', 'R08', 'R09', 'R10', 'R11', 'R12', 'R13', 'R15']:
                        annotator_file = "A02"
                    if P == 'S13' and R in ['R28']:
                        annotator_file = "A01"
                    if P == 'S13' and R in ['R29', 'R30']:
                        annotator_file = "A11"
                    if P == 'S09' and R in ['R28', 'R29']:
                        annotator_file = "A01"
                    if P == 'S09' and R in ['R21', 'R22', 'R23', 'R24', 'R25']:
                        annotator_file = "A11"
                    file_name_norm = "{}/{}_{}_{}_{}_{}_norm_data.csv".format(P, S, P, R, annotator_file, N)
                    file_name_label = "{}/{}_{}_{}_{}_{}_labels.csv".format(P, S, P, R, annotator_file, N)
                    #file_name_label = "{}/{}_{}_{}_labels.csv".format(P, S, P, R)
                    print("------------------------------\n{}".format(file_name_norm))
                    # getting data
                    path=dataset_path_imu + file_name_norm
                    pathlabels= dataset_path_imu + file_name_label
                    print(path)
                    print(pathlabels)
                    try:
                        #getting data
                        print("check")
                        data = np.loadtxt(path, delimiter=',', usecols=(range(2,134)), skiprows=1)
                        print("data size")
                        print(data)
                        print(data.shape)
                        print("Data Loaded")
                        #print("\nFiles loaded in modus\n{}".format(file_name_norm))
                        data = select_columns_opp(data)
                        #print("Columns selected")
                    except:
                        print("\n In generating data, No file {}".format(path))
                        continue        
                
                    try:
                        #Getting labels and attributes
                        labels = csv_reader.reader_labels(pathlabels)
                        class_labels = np.where(labels[:, 0] == 7)[0]

                        # Deleting rows containing the "none" class
                        data = np.delete(data, class_labels, 0)
                        labels = np.delete(labels, class_labels, 0)
                    except:
                        print("\n In generating data, Error getting the data {}".format(path))
                        continue
                
                    try:
                        print("data_x shape")
                        print(data.shape)
                        print(accumulator_measurements.shape)
                        accumulator_measurements = np.append(accumulator_measurements, data, axis=0)
                        print("\nFiles loaded")
                    except:          
                        print("\n1 In loading data,  in file {}".format(path))
                        continue
                            
    
    try:
        max_values = np.max(accumulator_measurements, axis=0)
        print("Max values")
        print(max_values)
        min_values = np.min(accumulator_measurements, axis=0)
        print("Min values")
        print(min_values)
        mean_values = np.mean(accumulator_measurements, axis=0)
        print("Mean values")
        print(mean_values)
        std_values = np.std(accumulator_measurements, axis=0)
        print("std values")
        print(std_values)
    except:
        max_values = 0
        min_values = 0
        mean_values = 0
        std_values = 0
        print("Error computing statistics")
    
    return max_values, min_values, mean_values, std_values

if __name__ == '__main__':
    
    #Computing Statistics of data
    max_values, min_values, mean_values, std_values = statistics_measurements()
    
    x = []
    x.append(list(max_values))
    x.append(list(min_values))
    x.append(list(mean_values))
    x.append(list(std_values))
    x=np.asarray(x)
    print(x)
  
    base_directory='/data/nnair/databias/'
    
    csv_dir=  base_directory+"norm_all.csv"
    print(csv_dir)
    np.savetxt(csv_dir, x, delimiter="\n", fmt='%s')