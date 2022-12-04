# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:59:33 2022

@author: nilah
"""
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
import time

def readData(accDir, annotFile):
    files = os.listdir(accDir)
    files_csv = [f for f in files if f[-3:] == 'csv']
    empatica_dict = dict()
    for f in files_csv:
        data = np.genfromtxt(accDir+f, delimiter=',') # creates numpy array for each Empatica acc csv file
        worker_id = int(float(f.strip("ACC.csv")))
        empatica_dict[worker_id] = data
    tmp = pd.read_excel(annotFile, sheet_name=None)
    annot_dict = dict(zip(tmp.keys(), [i.dropna() for i in tmp.values()])) # Remove the rows with NaN values (some with ladder 2 missing)
    return empatica_dict, annot_dict

def norm_lw(data):
        mean_values= np.array([-35.48195703, -27.12987543,  13.2462328])
        mean_values = np.reshape(mean_values, [1, 3])
        std_values = np.array([21.19355211, 36.44266775, 23.90226305])
        std_values = np.reshape(std_values, [1, 3])
        
        mean_array = np.repeat(mean_values, data.shape[0], axis=0)
        std_array = np.repeat(std_values, data.shape[0], axis=0)
        try:
           max_values = mean_array + 2 * std_array
           min_values = mean_array - 2 * std_array

           data_norm = (data - min_values) / (max_values - min_values)

           data_norm[data_norm > 1] = 1
           data_norm[data_norm < 0] = 0
        except:
            raise("Error in normalisation")

        return data_norm


#def getLabeledDict(empatica_dict, annot_dict, subject_ids, accumulator_measurements):
def getLabeledDict(empatica_dict, annot_dict, subject_ids, SR):
    labeled_dict = {}; taskInd_dict = {}
    for ids in subject_ids:
        start_time = int(empatica_dict[ids][0,0])
        print('START TIME :', start_time)
        acc = empatica_dict[ids][2:,:]
        #print("acc")
        #print(acc)
        label = list(map(lambda i: i.replace("_end", "").replace("_start", ""), annot_dict['P'+ str(ids)].taskName.tolist()))
        task_time= list(map(lambda i: time.mktime(datetime.datetime.strptime(i[:6] + '20' + i[6:], "%m/%d/%Y %H:%M:%S").timetuple()),
                            annot_dict['P'+ str(ids)].startTime_global.tolist()))
        print("task_time")
        print(task_time)
        print("sampling rate")
        print(SR)
        # changed "int(x - start_time)*SR" to "int(x - 14400)" in below line to reduce 4 hours (UTC to EDT conversion)
        task_ind = [int(x - 14400) for x in task_time]
        print('TASK IND:', task_ind)
        for x in task_time:
           print("start time")
           print(start_time)
           print("x")
           print(x)
           print(x-14400)
           print('complete')
           print(int(x-14400)*SR)
        #print("task_ind")
        #print(task_ind)
        taskInd_dict[id] = task_ind
        label_tmp = np.empty(acc.shape[0], dtype=object)
        setup=0
        new_limit=0
        
        for i, (j, k) in enumerate(zip(task_ind[0::2], task_ind[1::2])):
            if i==0:
                prevk=0
                new_limit = k-j
            else:
                prevk=j-prevk
                setup= setup+prevk
                new_limit=new_limit+ (k-j) +prevk
            tmpInd = 2*i
            label_tmp[setup:new_limit] = label[tmpInd]
            setup=new_limit
            prevk=k
        remove_rows =np.where(label_tmp == None)
        label_tmp=np.delete(label_tmp, np.where(label_tmp==None), axis=0)
        acc=np.delete(acc, remove_rows, axis=0)
        norm_acc= norm_lw(acc)
        #acctivate if attempting at finding norm min max std and mean
        #accumulator_measurements = np.append(accumulator_measurements, acc, axis=0)
        labeled_dict[id] = pd.DataFrame(np.hstack((norm_acc, label_tmp.reshape(label_tmp.shape[0],1))), columns=['X', 'Y', 'Z', 'label'])#add id label here itself? from dataframe to directly pickle?
    #return labeled_dict, taskInd_dict, accumulator_measurements
    return labeled_dict, taskInd_dict
    
if __name__ == '__main__':
    sepAccDict, sepAnnotDict = readData(accDir='/vol/actrec/Data_lineworker/Acc Data/', annotFile='/vol/actrec/Data_lineworker/Annotation Data/separate.xlsx')
    SR=int(sepAccDict[8][1,0])
    sepSubIDs = list(range(8,45))
    #print(len(sepSubIDs))
    #activate for finding the max min mean std of the accelerometer values for normalisationl
    '''
    accumulator_measurements = np.empty((0, 3))
    sepLabeledDict_, sepTaskIndDict, accu_meas = getLabeledDict(sepAccDict, sepAnnotDict, sepSubIDs, accumulator_measurements)
    try:
        max_values = np.max(accu_meas, axis=0)
        print("Max values")
        print(max_values)
        min_values = np.min(accu_meas, axis=0)
        print("Min values")
        print(min_values)
        mean_values = np.mean(accu_meas, axis=0)
        print("Mean values")
        print(mean_values)
        std_values = np.std(accu_meas, axis=0)
        print("std values")
        print(std_values)
    except:
        max_values = 0
        min_values = 0
        mean_values = 0
        std_values = 0
        print("Error computing statistics")
    '''
    sepLabeledDict_, sepTaskIndDict = getLabeledDict(sepAccDict, sepAnnotDict, sepSubIDs, SR)
    #print(sepLabeledDict_)
    #print(sepTaskIndDict)