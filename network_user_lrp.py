# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 15:24:03 2021

@author: nilah

Code by Fernando Moya
"""

from __future__ import print_function
import os
import sys
import logging
import numpy as np
import time
import math
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from numpy import savetxt

#from hdfs.config import catch

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.collections import PolyCollection

from network_act import Network
#from network_lstm import Network

import model_io
import modules
import csv

from HARWindows_act import HARWindows

from metrics import Metrics
#from metrics_act import Metrics



class Network_User(object):
    '''
    classdocs
    '''


    def __init__(self, config, exp):
        '''
        Constructor
        '''

        logging.info('        Network_User: Constructor')

        self.config = config
        self.device = torch.device("cuda:{}".format(self.config["GPU"]) if torch.cuda.is_available() else "cpu")

        #self.attrs = self.reader_att_rep("/home/nnair/Master_Thesis_2021/id_attr_one.txt")
        #self.attrs = self.reader_att_rep("/home/nnair/Master_Thesis_2021/id_attr_two.txt")
        
        #self.attr_representation = self.reader_att_rep("atts_per_class_lara.txt")

        self.normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([0.001]))
        self.exp = exp

        return
    
    ##################################################
    ###################  reader_att_rep  #############
    ##################################################

    def reader_att_rep(self, path: str) -> np.array:
        '''
        gets attribute representation from txt file.

        returns a numpy array

        @param path: path to file
        @param att_rep: Numpy matrix with the attribute representation
        '''

        att_rep = np.loadtxt(path, delimiter=',')
        #att_rep= torch.from_numpy(att_rep)
        return att_rep


    ##################################################
    ###################  plot  ######################
    ##################################################
    
    def plot(self, fig, axis_list, plot_list, metrics_list, activaciones, tgt, pred):
        '''
        Plots the input, and feature maps through the network.
        Deprecated for now.

        returns a numpy array

        @param fig: figure object
        @param axis_list: list with all of the axis. Each axis will represent a feature map
        @param plot_list: list of all of the plots of the feature maps
        @param metrics_list: Matrix with results
        @param tgt: Target class
        @param pred: Predicted class
        '''

        logging.info('        Network_User:    Plotting')
        if self.config['plotting']:
            #Plot

            for an, act in enumerate(activaciones):
                X = np.arange(0, act.shape[1])
                Y = np.arange(0, act.shape[0])
                X, Y = np.meshgrid(X, Y)

                axis_list[an * 2].plot_surface(X, Y, act, cmap=cm.coolwarm, linewidth=1, antialiased=False)

                axis_list[an * 2].set_title('Target {} and Pred {}'.format(tgt, pred))
                axis_list[an * 2].set_xlim3d(X.min(), X.max())
                axis_list[an * 2].set_xlabel('Sensor')
                axis_list[an * 2].set_ylim3d(Y.min(), Y.max())
                axis_list[an * 2].set_ylabel('Time')
                axis_list[an * 2].set_zlim3d(act.min(), act.max())
                axis_list[an * 2].set_zlabel('Measurement')


            for pl in range(len(metrics_list)):
                plot_list[pl].set_ydata(metrics_list[pl])
                plot_list[pl].set_xdata(range(len(metrics_list[pl])))


            '''      
                
            plot_list[0].set_ydata(metrics_list[0])
            plot_list[0].set_xdata(range(len(metrics_list[0])))
            
            plot_list[1].set_ydata(metrics_list[1])
            plot_list[1].set_xdata(range(len(metrics_list[1])))
            
            plot_list[2].set_ydata(metrics_list[2])
            plot_list[2].set_xdata(range(len(metrics_list[2])))
            
            plot_list[3].set_ydata(metrics_list[3])
            plot_list[3].set_xdata(range(len(metrics_list[3])))
            
            '''

            axis_list[1].relim()
            axis_list[1].autoscale_view()
            axis_list[1].legend(loc='best')

            axis_list[3].relim()
            axis_list[3].autoscale_view()
            axis_list[3].legend(loc='best')

            axis_list[5].relim()
            axis_list[5].autoscale_view()
            axis_list[5].legend(loc='best')

            axis_list[7].relim()
            axis_list[7].autoscale_view()
            axis_list[7].legend(loc='best')

            fig.canvas.draw()
            plt.savefig(self.config['folder_exp'] + 'training.png')
            #plt.show()
            plt.pause(0.2)
            axis_list[0].cla()
            axis_list[2].cla()
            axis_list[4].cla()
            axis_list[6].cla()
            axis_list[8].cla()

        return



    ##################################################
    ################  load_weights  ##################
    ##################################################
    def load_weights(self, network):
        '''
        Load weights from a trained network

        @param network: target network with orthonormal initialisation
        @return network: network with transferred CNN layers
        '''
        model_dict = network.state_dict()
        # 1. filter out unnecessary keys
        logging.info('        Network_User:        Loading Weights')

        #print(torch.load(self.config['folder_exp_base_fine_tuning'] + 'network.pt')['state_dict'])

        # Selects the source network according to configuration
        pretrained_dict = torch.load(self.config['folder_exp_base_fine_tuning'] + 'network.pt')['state_dict']
        logging.info('        Network_User:        Pretrained model loaded')

        #for k, v in pretrained_dict.items():
        #    print(k)

        if self.config["network"] == 'cnn':
            list_layers = ['conv1_1.weight', 'conv1_1.bias', 'conv1_2.weight', 'conv1_2.bias',
                           'conv2_1.weight', 'conv2_1.bias', 'conv2_2.weight', 'conv2_2.bias']
        elif self.config["network"] == 'cnn_imu':
            list_layers = ['conv_LA_1_1.weight', 'conv_LA_1_1.bias', 'conv_LA_1_2.weight', 'conv_LA_1_2.bias',
                           'conv_LA_2_1.weight', 'conv_LA_2_1.bias', 'conv_LA_2_2.weight', 'conv_LA_2_2.bias',
                           'conv_LL_1_1.weight', 'conv_LL_1_1.bias', 'conv_LL_1_2.weight', 'conv_LL_1_2.bias',
                           'conv_LL_2_1.weight', 'conv_LL_2_1.bias', 'conv_LL_2_2.weight', 'conv_LL_2_2.bias',
                           'conv_N_1_1.weight', 'conv_N_1_1.bias', 'conv_N_1_2.weight', 'conv_N_1_2.bias',
                           'conv_N_2_1.weight', 'conv_N_2_1.bias', 'conv_N_2_2.weight', 'conv_N_2_2.bias',
                           'conv_RA_1_1.weight', 'conv_RA_1_1.bias', 'conv_RA_1_2.weight', 'conv_RA_1_2.bias',
                           'conv_RA_2_1.weight', 'conv_RA_2_1.bias', 'conv_RA_2_2.weight', 'conv_RA_2_2.bias',
                           'conv_RL_1_1.weight', 'conv_RL_1_1.bias', 'conv_RL_1_2.weight',  'conv_RL_1_2.bias',
                           'conv_RL_2_1.weight', 'conv_RL_2_1.bias', 'conv_RL_2_2.weight', 'conv_RL_2_2.bias']

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in list_layers}
        #print(pretrained_dict)

        logging.info('        Network_User:        Pretrained layers selected')
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        logging.info('        Network_User:        Pretrained layers selected')
        # 3. load the new state dict
        network.load_state_dict(model_dict)
        logging.info('        Network_User:        Weights loaded')

        return network


    ##################################################
    ############  set_required_grad  #################
    ##################################################

    def set_required_grad(self, network):
        '''
        Setting the computing of the gradients for some layers as False
        This will act as the freezing of layers

        @param network: target network
        @return network: network with frozen layers
        '''

        model_dict = network.state_dict()
        # 1. filter out unnecessary keys
        logging.info('        Network_User:        Setting Required_grad to Weights')

        if self.config["network"] == 'cnn':
            list_layers = ['conv1_1.weight', 'conv1_1.bias', 'conv1_2.weight', 'conv1_2.bias',
                           'conv2_1.weight', 'conv2_1.bias', 'conv2_2.weight', 'conv2_2.bias']
        elif self.config["network"] == 'cnn_imu':
            list_layers = ['conv_LA_1_1.weight', 'conv_LA_1_1.bias', 'conv_LA_1_2.weight', 'conv_LA_1_2.bias',
                           'conv_LA_2_1.weight', 'conv_LA_2_1.bias', 'conv_LA_2_2.weight', 'conv_LA_2_2.bias',
                           'conv_LL_1_1.weight', 'conv_LL_1_1.bias', 'conv_LL_1_2.weight', 'conv_LL_1_2.bias',
                           'conv_LL_2_1.weight', 'conv_LL_2_1.bias', 'conv_LL_2_2.weight', 'conv_LL_2_2.bias',
                           'conv_N_1_1.weight', 'conv_N_1_1.bias', 'conv_N_1_2.weight', 'conv_N_1_2.bias',
                           'conv_N_2_1.weight', 'conv_N_2_1.bias', 'conv_N_2_2.weight', 'conv_N_2_2.bias',
                           'conv_RA_1_1.weight', 'conv_RA_1_1.bias', 'conv_RA_1_2.weight', 'conv_RA_1_2.bias',
                           'conv_RA_2_1.weight', 'conv_RA_2_1.bias', 'conv_RA_2_2.weight', 'conv_RA_2_2.bias',
                           'conv_RL_1_1.weight', 'conv_RL_1_1.bias', 'conv_RL_1_2.weight', 'conv_RL_1_2.bias',
                           'conv_RL_2_1.weight', 'conv_RL_2_1.bias', 'conv_RL_2_2.weight', 'conv_RL_2_2.bias']

        for pn, pv in network.named_parameters():
            if pn in list_layers:
                pv.requires_grad = False

        return network



    ##################################################
    ###################  train  ######################
    ##################################################


    def train(self, ea_itera):
        '''
        Training and validating a network

        @return results_val: dict with validation results
        @return best_itera: best iteration when validating
        '''

        logging.info('        Network_User: Train---->')

        logging.info('        Network_User:     Creating Dataloader---->')
        #if self.config["dataset"] == "mbientlab":
        #    harwindows_train = HARWindows(csv_file=self.config['dataset_root'] + "train_{}.csv".format(self.config["percentages_names"]),
        #                                  root_dir=self.config['dataset_root'])
        #else:

        # Selecting the training sets, either train or train final (train  + Validation)
        ###########change train_final to tain and create a new csv file
        if self.config['usage_modus'] == 'train':
            harwindows_train = HARWindows(csv_file=self.config['dataset_root'] + "train.csv",
                                          root_dir=self.config['dataset_root'])
        elif self.config['usage_modus'] == 'train_final':
            harwindows_train = HARWindows(csv_file=self.config['dataset_root'] + "train.csv",
                                         root_dir=self.config['dataset_root'])
        elif self.config['usage_modus'] == 'fine_tuning':
            harwindows_train = HARWindows(csv_file=self.config['dataset_root'] + "train.csv",
                                         root_dir=self.config['dataset_root'])

        # Creating the dataloader
        dataLoader_train = DataLoader(harwindows_train, batch_size=self.config['batch_size_train'], shuffle=True)
        print(dataLoader_train)
        # Setting the network
        logging.info('        Network_User:    Train:    creating network')
        if self.config['network'] == 'cnn' or self.config['network'] == 'cnn_imu':
            network_obj = Network(self.config)
            network_obj.init_weights()

            # IF finetuning, load the weights from a source dataset
            if self.config["usage_modus"] == "fine_tuning":
                network_obj = self.load_weights(network_obj)

            # Displaying size of tensors
            logging.info('        Network_User:    Train:    network layers')
            for l in list(network_obj.named_parameters()):
                logging.info('        Network_User:    Train:    {} : {}'.format(l[0], l[1].detach().numpy().shape))

            logging.info('        Network_User:    Train:    setting device')
            network_obj.to(self.device)

        # Setting loss
        if self.config['output'] == 'softmax':
            logging.info('        Network_User:    Train:    setting criterion optimizer Softmax')
            if self.config["fully_convolutional"] == "FCN":
                criterion = nn.CrossEntropyLoss()
            elif self.config["fully_convolutional"] == "FC":
                criterion = nn.CrossEntropyLoss()
        elif self.config['output'] == 'attribute':
            logging.info('        Network_User:    Train:    setting criterion optimizer Attribute')
            if self.config["fully_convolutional"] == "FCN":
                criterion = nn.BCELoss()
            elif self.config["fully_convolutional"] == "FC":
               criterion = nn.BCELoss()
               #criterion = nn.BCEWithLogitsLoss()

        # Setting the freezing or not freezing from conv layers
        if self.config['freeze_options']:
            network_obj = self.set_required_grad(network_obj)

        # Setting optimizer
        optimizer = optim.RMSprop(network_obj.parameters(), lr=self.config['lr'], alpha=0.95)
        
        # Setting scheduler
        step_lr = self.config['epochs'] / 3
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=math.ceil(step_lr), gamma=0.1)

        if self.config['plotting']:
            # Plotting the input or feature maps through the network.
            # For now deprecated, as features for all the layers arent stored.

            logging.info('        Network_User:    Train:    setting plotting objects')

            fig = plt.figure(figsize=(16, 12), dpi=160)
            axis_list = []
            axis_list.append(fig.add_subplot(521, projection='3d'))
            axis_list.append(fig.add_subplot(522))
            axis_list.append(fig.add_subplot(523, projection='3d'))
            axis_list.append(fig.add_subplot(524))
            axis_list.append(fig.add_subplot(525, projection='3d'))
            axis_list.append(fig.add_subplot(526))
            axis_list.append(fig.add_subplot(527, projection='3d'))
            axis_list.append(fig.add_subplot(528))
            axis_list.append(fig.add_subplot(529, projection='3d'))

            plot_list = []
            # loss_plot, = axis_list[1].plot([], [],'-r',label='red')
            # plots acc, f1w, f1m for training
            plot_list.append(axis_list[1].plot([], [],'-r',label='acc')[0])
            plot_list.append(axis_list[1].plot([], [],'-b',label='f1w')[0])
            plot_list.append(axis_list[1].plot([], [],'-g',label='f1m')[0])

            # plot loss training
            plot_list.append(axis_list[3].plot([], [],'-r',label='loss tr')[0])

            # plots acc, f1w, f1m for training and validation
            plot_list.append(axis_list[5].plot([], [],'-r',label='acc tr')[0])
            plot_list.append(axis_list[5].plot([], [],'-b',label='f1w tr')[0])
            plot_list.append(axis_list[5].plot([], [],'-g',label='f1m tr')[0])

            plot_list.append(axis_list[5].plot([], [],'-c',label='acc vl')[0])
            plot_list.append(axis_list[5].plot([], [],'-m',label='f1w vl')[0])
            plot_list.append(axis_list[5].plot([], [],'-y',label='f1m vl')[0])

            # plot loss for training and validation
            plot_list.append(axis_list[7].plot([], [],'-r',label='loss tr')[0])
            plot_list.append(axis_list[7].plot([], [],'-b',label='loss vl')[0])

            # Customize the z axis.
            for al in range(len(axis_list)):
                if al%2 ==0:
                    axis_list[al].set_zlim(0.0, 1.0)
                    axis_list[al].zaxis.set_major_locator(LinearLocator(10))
                    axis_list[al].zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Initializing lists for plots and results
        losses_train = []
        accs_train = []
        f1w_train = []
        f1m_train = []

        losses_val = []
        accs_val = []
        f1w_val = []
        f1m_val = []

        loss_train_val = []
        accs_train_val = []
        f1w_train_val = []
        f1m_train_val = []
        
        count_pos_val = [0, 0, 0, 0, 0, 0, 0, 0]
        count_neg_val = [0, 0, 0, 0, 0, 0, 0, 0]

        best_acc_val = 0
        
        
        # initialising object for computing metrics
        if self.config['output'] == 'softmax':
            metrics_obj = Metrics(self.config, self.device)
        elif self.config['output'] == 'attribute': 
            metrics_obj = Metrics(self.config, self.device, self.attrs)
           

        itera = 0
        start_time_train = time.time()

        # zero the parameter gradients
        optimizer.zero_grad()

        best_itera = 0
        
        # loop for training
        # Validation is not carried out per epoch, but after certain number of iterations, specified
        # in configuration in main
        for e in range(self.config['epochs']):
            start_time_train = time.time()
            logging.info('\n        Network_User:    Train:    Training epoch {}'.format(e))
            start_time_batch = time.time()
            
            #loop per batch:
            for b, harwindow_batched in enumerate(dataLoader_train):
                start_time_batch = time.time()
                sys.stdout.write("\rTraining: Epoch {}/{} Batch {}/{} and itera {}".format(e,
                                                                                           self.config['epochs'],
                                                                                           b,
                                                                                           len(dataLoader_train),
                                                                                           itera))
                sys.stdout.flush()

                #Setting the network to train mode
                network_obj.train(mode=True)
                
                #Counting iterations
                #itera = (e * harwindow_batched["data"].shape[0]) + b
                
                #Selecting batch
                train_batch_v = harwindow_batched["data"]
                
                if self.config['output'] == 'softmax':
                    if self.config["fully_convolutional"] == "FCN":
                        train_batch_l = harwindow_batched["labels"][:, :, 0]
                        train_batch_l = train_batch_l.reshape(-1)
                    elif self.config["fully_convolutional"] == "FC":
                        train_batch_l = harwindow_batched["label"]
                        train_batch_l = train_batch_l.reshape(-1)
                elif self.config['output'] == 'attribute':
                    if self.config["fully_convolutional"] == "FCN":
                        train_batch_l = harwindow_batched["labels"][:, :, 1:]
                    elif self.config["fully_convolutional"] == "FC":
                        sample = harwindow_batched["label"]
                        sample = sample.reshape(-1)
                        train_batch_l=np.zeros([sample.shape[0],self.config['num_attributes']+1])
                        
                        for i in range(0,sample.shape[0]):
                            if sample[i]==self.attrs[sample[i],0]:
                                n=sample[i].item()
                                train_batch_l[i]= self.attrs[n]
                       
                '''
                if self.config['output'] == 'softmax':
                    if self.config["fully_convolutional"] == "FCN":
                        train_batch_l = harwindow_batched["labels"][:, :, 0]
                        train_batch_l = train_batch_l.reshape(-1)
                    elif self.config["fully_convolutional"] == "FC":
                        train_batch_l = harwindow_batched["label"][:, 0]
                        train_batch_l = train_batch_l.reshape(-1)
                elif self.config['output'] == 'attribute':
                    if self.config["fully_convolutional"] == "FCN":
                        train_batch_l = harwindow_batched["labels"][:, :, 1:]
                    elif self.config["fully_convolutional"] == "FC":
                        train_batch_l = harwindow_batched["label"]
                '''        
                # Adding gaussian noise
                noise = self.normal.sample((train_batch_v.size()))
                noise = noise.reshape(train_batch_v.size())
                noise = noise.to(self.device, dtype=torch.float)

                # Sending to GPU
                train_batch_v = train_batch_v.to(self.device, dtype=torch.float)
                train_batch_v += noise
                if self.config['output'] == 'softmax':
                    train_batch_l = train_batch_l.to(self.device, dtype=torch.long) #labels for crossentropy needs long type
                elif self.config['output'] == 'attribute':
                    train_batch_l=torch.from_numpy(train_batch_l)
                    train_batch_l=train_batch_l.to(self.device, dtype=torch.float) #labels for binerycrossentropy needs float type
                
                # forward + backward + optimize
                
                feature_maps= network_obj(train_batch_v)
                
                if self.config["fully_convolutional"] == "FCN":
                    feature_maps = feature_maps.reshape(-1, feature_maps.size()[2])
                if self.config['output'] == 'softmax':
                    #print()
                    loss = criterion(feature_maps, train_batch_l) * (1 / self.config['accumulation_steps'])
                    '''
                    loss_id = criterion(feature_maps, train_batch_l) * (1 / self.config['accumulation_steps'])
                    loss_act = criterion(feature_maps_act, train_batch_l_activity) * (1 / self.config['accumulation_steps'])
                    loss=loss_id+loss_act
                    '''
                elif self.config['output'] == 'attribute':
                    loss = criterion(feature_maps, train_batch_l[:, 1:]) * (1 / self.config['accumulation_steps'])
                   
                loss.backward()

                if (itera + 1) % self.config['accumulation_steps'] == 0:
                    optimizer.step()
                    # zero the parameter gradients
                    optimizer.zero_grad()

                loss_train = loss.item()

                elapsed_time_batch = time.time() - start_time_batch

                ################################## Validating ##################################################
                
                if (itera + 1) % self.config['valid_show'] == 0 or (itera + 1) == (self.config['epochs'] * harwindow_batched["data"].shape[0]):
                    logging.info('\n')
                    logging.info('        Network_User:        Validating')
                    start_time_val = time.time()

                    #Setting the network to eval mode
                    network_obj.eval()

                    # Metrics for training for keeping the same number of metrics for val and training
                    # Metrics return a dict with the metrics.
                    results_train = metrics_obj.metric(targets=train_batch_l, predictions=feature_maps)
                    loss_train_val.append(loss_train)
                    accs_train_val.append(results_train['acc'])
                    f1w_train_val.append(results_train['f1_weighted'])
                    f1m_train_val.append(results_train['f1_mean'])
                    
                    # Validation
                    # Calling the val() function with the current network and criterion
                    del train_batch_v, noise
                    results_val, loss_val, c_pos_val, c_neg_val = self.validate(network_obj, criterion)
                 
                    self.exp.log_scalar("loss_val_int_{}".format(ea_itera), loss_val, itera)

                    elapsed_time_val = time.time() - start_time_val

                    # Appending the val metrics
                    losses_val.append(loss_val)
                    accs_val.append(results_val['acc'])
                    f1w_val.append(results_val['f1_weighted'])
                    f1m_val.append(results_val['f1_mean'])
                    
                    self.exp.log_scalar("accuracy_val_int_{}".format(ea_itera),results_val['acc'], itera)
                    self.exp.log_scalar("f1_w_val_int_{}".format(ea_itera),results_val['f1_weighted'], itera)
                    self.exp.log_scalar("f1_m_val_int_{}".format(ea_itera), results_val['f1_mean'], itera)
                    if self.config['output']== 'attribute':
                        p=results_val['acc_attrs']
                        for i in range(0,p.shape[0]):
                            self.exp.log_scalar("acc_attr_{}_val_int_{}".format(i, ea_itera),p[i], itera)
                    '''
                    if c_pos_val[0] == 0:
                        self.exp.log_scalar("standing_pos_val_{}".format(ea_itera), c_pos_val[0], itera)
                    else:
                        self.exp.log_scalar("standing_pos_val_{}".format(ea_itera), c_pos_val[0]/(c_pos_val[0]+c_neg_val[0]), itera)
                    if c_pos_val[1] == 0:
                        self.exp.log_scalar("walking_pos_val_{}".format(ea_itera), c_pos_val[1], itera)
                    else:
                        self.exp.log_scalar("walking_pos_val_{}".format(ea_itera), c_pos_val[1]/(c_pos_val[1]+c_neg_val[1]), itera)
                    if c_pos_val[2] == 0:
                        self.exp.log_scalar("cart_pos_val_{}".format(ea_itera), c_pos_val[2], itera)
                    else:
                        self.exp.log_scalar("cart_pos_val_{}".format(ea_itera), c_pos_val[2]/(c_pos_val[2]+c_neg_val[2]), itera)
                    if c_pos_val[3] == 0:
                        self.exp.log_scalar("handling_up_pos_val_{}".format(ea_itera), c_pos_val[3], itera)
                    else:
                        self.exp.log_scalar("handling_up_pos_val_{}".format(ea_itera), c_pos_val[3]/(c_pos_val[3]+c_neg_val[3]), itera)
                    if c_pos_val[4] == 0:
                        self.exp.log_scalar("handling_cen_pos_val_{}".format(ea_itera), c_pos_val[4], itera)
                    else:
                        self.exp.log_scalar("handling_cen_pos_val_{}".format(ea_itera), c_pos_val[4]/(c_pos_val[4]+c_neg_val[4]), itera)
                    if c_pos_val[5] == 0:
                        self.exp.log_scalar("handling_down_pos_val_{}".format(ea_itera), c_pos_val[5], itera)
                    else:
                        self.exp.log_scalar("handling_down_pos_val_{}".format(ea_itera), c_pos_val[5]/(c_pos_val[5]+c_neg_val[5]), itera)
                    if c_pos_val[6] == 0:
                        self.exp.log_scalar("synch_pos_val_{}".format(ea_itera), c_pos_val[6], itera)
                    else:
                        self.exp.log_scalar("synch_pos_val_{}".format(ea_itera), c_pos_val[6]/(c_pos_val[6]+c_neg_val[6]), itera)
                    if c_pos_val[7] == 0:
                        self.exp.log_scalar("none_pos_val_{}".format(ea_itera), c_pos_val[7], itera)
                    else:
                        self.exp.log_scalar("none_pos_val_{}".format(ea_itera), c_pos_val[7]/(c_pos_val[7]+c_neg_val[7]), itera)
                    
                    if c_neg_val[0] == 0:
                        self.exp.log_scalar("standing_neg_val_{}".format(ea_itera), c_neg_val[0], itera)
                    else:
                        self.exp.log_scalar("standing_neg_val_{}".format(ea_itera), c_neg_val[0]/(c_pos_val[0]+c_neg_val[0]), itera)
                    if c_neg_val[1] == 0:
                        self.exp.log_scalar("walking_neg_val_{}".format(ea_itera), c_neg_val[1], itera)
                    else:
                        self.exp.log_scalar("walking_neg_val_{}".format(ea_itera), c_neg_val[1]/(c_pos_val[1]+c_neg_val[1]), itera)
                    if c_neg_val[2] == 0:
                        self.exp.log_scalar("cart_neg_val_{}".format(ea_itera), c_neg_val[2], itera)
                    else:
                        self.exp.log_scalar("cart_neg_val_{}".format(ea_itera), c_neg_val[2]/(c_pos_val[2]+c_neg_val[2]), itera)
                    if c_neg_val[3] == 0:
                        self.exp.log_scalar("handling_up_neg_val_{}".format(ea_itera), c_neg_val[3], itera)
                    else:
                        self.exp.log_scalar("handling_up_neg_val_{}".format(ea_itera), c_neg_val[3]/(c_pos_val[3]+c_neg_val[3]), itera)
                    if c_neg_val[4] == 0:
                        self.exp.log_scalar("handling_cen_neg_val_{}".format(ea_itera), c_neg_val[4], itera)
                    else:
                        self.exp.log_scalar("handling_cen_neg_val_{}".format(ea_itera), c_neg_val[4]/(c_pos_val[4]+c_neg_val[4]), itera)
                    if c_neg_val[5] == 0:
                        self.exp.log_scalar("handling_down_neg_val_{}".format(ea_itera), c_neg_val[5], itera)
                    else:
                        self.exp.log_scalar("handling_down_neg_val_{}".format(ea_itera), c_neg_val[5]/(c_pos_val[5]+c_neg_val[5]), itera)
                    if c_neg_val[6] == 0:
                        self.exp.log_scalar("synch_neg_val_{}".format(ea_itera), c_neg_val[6], itera)
                    else:
                        self.exp.log_scalar("synch_neg_val_{}".format(ea_itera), c_neg_val[6]/(c_pos_val[6]+c_neg_val[6]), itera)
                    if c_neg_val[7] == 0:
                        self.exp.log_scalar("none_neg_val_{}".format(ea_itera), c_neg_val[7], itera)
                    else:
                        self.exp.log_scalar("none_neg_val_{}".format(ea_itera), c_neg_val[7]/(c_pos_val[7]+c_neg_val[7]), itera)
                    '''
                    
                    count_pos_val=np.array(count_pos_val)
                    count_neg_val=np.array(count_neg_val)
                    c_pos_val= np.array(c_pos_val)
                    c_neg_val= np.array(c_neg_val)
                    count_pos_val= count_pos_val+ c_pos_val
                    count_neg_val= count_neg_val+ c_neg_val
                    count_pos_val=count_pos_val.tolist()
                    count_neg_val=count_neg_val.tolist()
                    
                    # print statistics
                    logging.info('\n')
                    logging.info(
                        '        Network_User:        Validating:    '
                        'epoch {} batch {} itera {} elapsed time {}, best itera {}'.format(e, b, itera,
                                                                                           elapsed_time_val,
                                                                                           best_itera))
                    logging.info(
                        '        Network_User:        Validating:    '
                        'acc {}, f1_weighted {}, f1_mean {}'.format(results_val['acc'], results_val['f1_weighted'],
                                                                    results_val['f1_mean']))
                    # Saving the network for the best iteration accuracy
                    if results_val['acc'] > best_acc_val:
                        network_config = {
                            'NB_sensor_channels': self.config['NB_sensor_channels'],
                            'sliding_window_length': self.config['sliding_window_length'],
                            'filter_size': self.config['filter_size'],
                            'num_filters': self.config['num_filters'],
                            'reshape_input': self.config['reshape_input'],
                            'network': self.config['network'],
                            'output': self.config['output'],
                            'num_classes': self.config['num_classes'],
                            #'num_attributes': self.config['num_attributes'],
                            'fully_convolutional': self.config['fully_convolutional'],
                            'labeltype': self.config['labeltype']
                        }

                        logging.info('        Network_User:            Saving the network')

                        torch.save({'state_dict': network_obj.state_dict(),
                                    'network_config': network_config,
                                    #'att_rep': self.attr_representation
                                    },
                                   self.config['folder_exp'] + 'network.pt')
                        best_acc_val = results_val['acc']
                        best_itera = itera
                
                # Computing metrics for current training batch
                
                if (itera) % self.config['train_show'] == 0:
                    # Metrics for training
                    
                    results_train = metrics_obj.metric(targets=train_batch_l, predictions=feature_maps)

                    activaciones = []
                    metrics_list = []
                    accs_train.append(results_train['acc'])
                    f1w_train.append(results_train['f1_weighted'])
                    f1m_train.append(results_train['f1_mean'])
                    losses_train.append(loss_train)

                    # Plotting for now deprecated
                    if self.config['plotting']:
                        #For plotting
                        metrics_list.append(accs_train)
                        metrics_list.append(f1w_train)
                        metrics_list.append(f1m_train)
                        metrics_list.append(losses_train)
                        metrics_list.append(accs_train_val)
                        metrics_list.append(f1w_train_val)
                        metrics_list.append(f1m_train_val)
                        metrics_list.append(accs_val)
                        metrics_list.append(f1w_val)
                        metrics_list.append(f1m_val)
                        metrics_list.append(loss_train_val)
                        metrics_list.append(losses_val)
                        #activaciones.append(train_batch_v.to("cpu", torch.double).detach().numpy()[0,0,:,:])
                        #activaciones.append(feature_maps[0].to("cpu", torch.double).detach().numpy()[0,0,:,:])
                        #activaciones.append(feature_maps[1].to("cpu", torch.double).detach().numpy()[0,0,:,:])
                        #activaciones.append(feature_maps[2].to("cpu", torch.double).detach().numpy()[0,0,:,:])
                        #activaciones.append(feature_maps[3].to("cpu", torch.double).detach().numpy()[0,0,:,:])
                        self.plot(fig, axis_list, plot_list, metrics_list, activaciones,
                                  harwindow_batched["label"][0].item(),
                                  torch.argmax(feature_maps[0], dim=0).item())
                        
                    # print statistics
                    logging.info('        Network_User:            Dataset {} network {} lr {} '
                                 'lr_optimizer {} Reshape {} '.format(self.config["dataset"], self.config["network"],
                                                                      self.config["lr"],
                                                                      optimizer.param_groups[0]['lr'],
                                                                      self.config["reshape_input"]))
                    logging.info(
                        '        Network_User:    Train:    epoch {}/{} batch {}/{} itera {} '
                        'elapsed time {} best itera {}'.format(e, self.config['epochs'], b, len(dataLoader_train),
                                                               itera, elapsed_time_batch, best_itera))
                    logging.info('        Network_User:    Train:    loss {}'.format(loss))
                    logging.info(
                        '        Network_User:    Train:    acc {}, '
                        'f1_weighted {}, f1_mean {}'.format(results_train['acc'], results_train['f1_weighted'],
                                                            results_train['f1_mean']))
                    logging.info(
                        '        Network_User:    Train:    '
                        'Allocated {} GB Cached {} GB'.format(round(torch.cuda.memory_allocated(0)/1024**3, 1),
                                                              round(torch.cuda.memory_cached(0)/1024**3, 1)))
                    logging.info('\n\n--------------------------')
                    
                    if self.config["sacred"]==True:
                        self.exp.log_scalar("accuracy_train_int_{}".format(ea_itera),results_train['acc'], itera)
                        self.exp.log_scalar("f1_w_train_int_{}".format(ea_itera),results_train['f1_weighted'], itera)
                        self.exp.log_scalar("f1_m_train_int_{}".format(ea_itera), results_train['f1_mean'], itera)
                        self.exp.log_scalar("loss_train_int_{}".format(ea_itera), loss_train, itera)
                        if self.config['output']== 'attribute':
                            p=results_train['acc_attrs']
                            for i in range(0,p.shape[0]):
                                self.exp.log_scalar("acc_attr_{}_train_int_{}".format(i, ea_itera),p[i], itera)
                    
                                           
                itera+=1
                        
            #Step of the scheduler
            scheduler.step()

        elapsed_time_train = time.time() - start_time_train

        logging.info('\n')
        logging.info(
            '        Network_User:    Train:    epoch {} batch {} itera {} '
            'Total training time {}'.format(e, b, itera, elapsed_time_train))

        # Storing the acc, f1s and losses of training and validation for the current training run
        np.savetxt(self.config['folder_exp'] + 'plots/acc_train.txt', accs_train_val, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/f1m_train.txt', f1m_train_val, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/f1w_train.txt', f1w_train_val, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/loss_train.txt', loss_train_val, delimiter=",", fmt='%s')

        np.savetxt(self.config['folder_exp'] + 'plots/acc_val.txt', accs_val, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/f1m_val.txt', f1m_val, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/f1w_val.txt', f1w_val, delimiter=",", fmt='%s')
        np.savetxt(self.config['folder_exp'] + 'plots/loss_val.txt', losses_val, delimiter=",", fmt='%s')
        
        '''save the model into the desired location'''
        torch.save({'state_dict': network_obj.state_dict(),'network_config': network_config}, '/data/nnair/model/cnnimu_imu.pt')
        #model_io.write(network_obj, '../Master_Thesis_2021/model/model_save_mocap2.pkl')
       
        del losses_train, accs_train, f1w_train, f1m_train
        del losses_val, accs_val, f1w_val, f1m_val
        del loss_train_val, accs_train_val, f1w_train_val, f1m_train_val
        del metrics_list, feature_maps
        del network_obj
       
        torch.cuda.empty_cache()

        if self.config['plotting']:
            plt.savefig(self.config['folder_exp'] + 'training_final.png')
            plt.close()
        
               
        return results_val, best_itera, count_pos_val, count_neg_val


    ##################################################
    ################  Validate  ######################
    ##################################################

    def validate(self, network_obj, criterion):
        '''
        Validating a network

        @param network_obj: network object
        @param criterion: torch criterion object
        @return results_val: dict with validation results
        @return loss: loss of the validation
        '''
            
        # Setting validation set and dataloader
        harwindows_val = HARWindows(csv_file=self.config['dataset_root'] + "val.csv",
                                    root_dir=self.config['dataset_root'])

        dataLoader_val = DataLoader(harwindows_val, batch_size=self.config['batch_size_val'])

        # Setting the network to eval mode
        network_obj.eval()

        # Creating metric object
        if self.config['output'] == 'softmax':
            metrics_obj = Metrics(self.config, self.device)
        elif self.config['output'] == 'attribute': 
            metrics_obj = Metrics(self.config, self.device, self.attrs)
       
        loss_val = 0
        count_pos_val = [0, 0, 0, 0, 0, 0, 0, 0]
        count_neg_val = [0, 0, 0, 0, 0, 0, 0, 0]

        # One doesnt need the gradients
        with torch.no_grad():
            for v, harwindow_batched_val in enumerate(dataLoader_val):
                # Selecting batch
                test_batch_v = harwindow_batched_val["data"]
                if self.config['output'] == 'softmax':
                    if self.config["fully_convolutional"] == "FCN":
                        test_batch_l = harwindow_batched_val["labels"][:, 0]
                        test_batch_l = test_batch_l.reshape(-1)
                    elif self.config["fully_convolutional"] == "FC":
                        test_batch_l = harwindow_batched_val["label"]
                        test_batch_l = test_batch_l.reshape(-1)
                elif self.config['output'] == 'attribute':
                    if self.config["fully_convolutional"] == "FCN":
                        test_batch_l = harwindow_batched_val["labels"][:, 0]
                        test_batch_l = test_batch_l.reshape(-1)
                    elif self.config["fully_convolutional"] == "FC":
                        sample = harwindow_batched_val["label"]
                        sample = sample.reshape(-1)
                        test_batch_l=np.zeros([sample.shape[0],self.config['num_attributes']+1])
                        
                        for i in range(0,sample.shape[0]):
                            if sample[i]==self.attrs[sample[i],0]:
                                n=sample[i].item()
                                test_batch_l[i]= self.attrs[n]
                        
                            
                # Creating torch tensors
                # test_batch_v = torch.from_numpy(test_batch_v)
                test_batch_v = test_batch_v.to(self.device, dtype=torch.float)
                if self.config['output'] == 'softmax':
                    # labels for crossentropy needs long type
                    test_batch_l = test_batch_l.to(self.device, dtype=torch.long)
                elif self.config['output'] == 'attribute':
                    # labels for binerycrossentropy needs float type
                    test_batch_l=torch.from_numpy(test_batch_l)
                    test_batch_l = test_batch_l.to(self.device, dtype=torch.float)
                    # labels for crossentropy needs long type

                # forward
                predictions = network_obj(test_batch_v)
                
                if self.config['output'] == 'softmax':
                    loss = criterion(predictions, test_batch_l)
                elif self.config['output'] == 'attribute':
                    loss = criterion(predictions, test_batch_l[:, 1:])
                    
                loss_val = loss_val + loss.item()
               
                act_class=harwindow_batched_val["act_label"] 
                act_class = act_class.reshape(-1)
                
                if self.config['output'] == 'softmax':
                    pred_index= predictions.argmax(1)
                    
                    label=test_batch_l
                    for i,x in enumerate(pred_index):
                        if pred_index[i]==label[i]:
                           for c,z in enumerate(count_pos_val):
                                if c==act_class[i]:
                                    count_pos_val[c]+=1
                        else:
                            for c,z in enumerate(count_neg_val):
                                if c==act_class[i]:
                                    count_neg_val[c]+=1
                elif self.config['output'] == 'attribute': 
                    pred=np.zeros([predictions.shape[0],predictions.shape[1]])
                    pred=torch.from_numpy(pred)
                    pred= pred.to(self.device, dtype=torch.float)
                    for i in range(predictions.shape[0]):
                      pred[i]= (predictions[i]>0.5).float()
                    
                    label=test_batch_l[:,1:]
                    for i,k in enumerate([pred.shape[0]]):
                        if torch.all(pred[i].eq(label[i])):
                           for c,z in enumerate(count_pos_val):
                                if c==act_class[i]:
                                    count_pos_val[c]+=1
                        else:
                            for c,z in enumerate(count_neg_val):
                                if c==act_class[i]:
                                    count_neg_val[c]+=1
               
                # Concatenating all of the batches for computing the metrics
                # As creating an empty tensor and sending to device and then concatenating isnt working
                if v == 0:
                    predictions_val = predictions
                    if self.config['output'] == 'softmax':
                        test_labels = harwindow_batched_val["label"]
                        test_labels = test_labels.reshape(-1)
                    elif self.config['output'] == 'attribute':
                        sample = harwindow_batched_val["label"]
                        sample = sample.reshape(-1)
                        test_labels=np.zeros([sample.shape[0],self.config['num_attributes']+1])
                        
                        for i in range(0,sample.shape[0]):
                            if sample[i]==self.attrs[sample[i],0]:
                                n=sample[i].item()
                                test_labels[i]= self.attrs[n]
                        
                        test_labels=torch.from_numpy(test_labels)
                        test_labels= test_labels.to(self.device, dtype=torch.float)
                else:
                    predictions_val = torch.cat((predictions_val, predictions), dim=0)
                    if self.config['output'] == 'softmax':
                        test_labels_batch = harwindow_batched_val["label"]
                        test_labels_batch = test_labels_batch.reshape(-1)
                    elif self.config['output'] == 'attribute':
                        sample = harwindow_batched_val["label"]
                        sample = sample.reshape(-1)
                        test_labels_batch=np.zeros([sample.shape[0],self.config['num_attributes']+1])
                        
                        for i in range(0,sample.shape[0]):
                            if sample[i]==self.attrs[sample[i],0]:
                                n=sample[i].item()
                                test_labels_batch[i]= self.attrs[n]
                        
                        test_labels_batch=torch.from_numpy(test_labels_batch)
                        test_labels_batch = test_labels_batch.to(self.device, dtype=torch.float)
                    test_labels = torch.cat((test_labels, test_labels_batch), dim=0)
                    
                
                '''
                if v == 0:
                    predictions_val = predictions
                    if self.config['output'] == 'softmax':
                        test_labels = harwindow_batched_val["label"][:, 0]
                        test_labels = test_labels.reshape(-1)
                    elif self.config['output'] == 'attribute':
                        test_labels = harwindow_batched_val["label"]
                else:
                    predictions_val = torch.cat((predictions_val, predictions), dim=0)
                    if self.config['output'] == 'softmax':
                        test_labels_batch = harwindow_batched_val["label"][:, 0]
                        test_labels_batch = test_labels_batch.reshape(-1)
                    elif self.config['output'] == 'attribute':
                        test_labels_batch = harwindow_batched_val["label"]
                    test_labels = torch.cat((test_labels, test_labels_batch), dim=0)
                '''
                sys.stdout.write("\rValidating: Batch  {}/{}".format(v, len(dataLoader_val)))
                sys.stdout.flush()

        print("\n")
        # Computing metrics of validation
        test_labels = test_labels.to(self.device, dtype=torch.float)
        results_val = metrics_obj.metric(test_labels, predictions_val)

        del test_batch_v, test_batch_l
        del predictions, predictions_val
        del test_labels_batch, test_labels

        torch.cuda.empty_cache()

        return results_val, loss_val / v, count_pos_val, count_neg_val




    ##################################################
    ###################  test  ######################
    ##################################################

    def test(self, ea_itera):
        '''
        Testing a network

        @param ea_itera: evolution iteration
        @return results_test: dict with testing results
        @return confusion matrix: confusion matrix of testing results
        '''
        logging.info('        Network_User:    Test ---->')
        
        
        # Setting the testing set
        logging.info('        Network_User:     Creating Dataloader---->')
        harwindows_test = HARWindows(csv_file=self.config['dataset_root'] + "test.csv",
                                     root_dir=self.config['dataset_root'])

        dataLoader_test = DataLoader(harwindows_test, batch_size=self.config['batch_size_train'], shuffle=False)

        # Creating a network and loading the weights for testing
        # network is loaded from saved file in the folder of experiment
        logging.info('        Network_User:    Test:    creating network')
        
        '''
        if self.config['network'] == 'cnn' or self.config['network'] == 'cnn_imu':
            network_obj = Network(self.config)

            #Loading the model
            network_obj.load_state_dict(torch.load(self.config['folder_exp'] + 'network.pt')['state_dict'])
            network_obj.eval()

            logging.info('        Network_User:    Test:    setting device')
            network_obj.to(self.device)
        '''
        
        network_obj = Network(self.config)
        print("network weights before initialisation")
        #print(network_obj.conv1_1.weight)
        network_obj.init_weights()
        print("initalised network with weight")
        #print(network_obj)
        #print(network_obj.conv1_1.weight)
        model_dict = network_obj.state_dict()
        print("model dict with state dict loaded")
        #print(model_dict)
        #print(model_dict.conv1_1.weight)
        #pretrained_dict= torch.load('/data/nnair/model/cnn_imu_new.pt')['state_dict']
        pretrained_dict= torch.load('/data/nnair/model/cnn_mocap.pt')['state_dict']
        #print("network loaded from cnn_imu.pt")
        print("network loaded from cnn_mocap.pt")
        
        #print(network_obj)
        #print(network_obj.conv_LA_1_1.weight)
        '''
        if self.config["dataset"]=='mocap':
            #network_obj.load_state_dict(torch.load('/data/nnair/model/model_save_mocap.pt'))
            #network_obj.load_state_dict(torch.load(self.config['folder_exp'] + 'network.pt')['state_dict'])
            pretrained_dict= torch.load('/data/nnair/model/cnn_mocap.pt')['state_dict']
            print("network loaded from cnn_mocap.pt")
        elif self.config["dataset"]=='mbientlab':
            #network_obj.load_state_dict(torch.load('/data/nnair/model/model_save_imu.pt'))
            pretrained_dict= torch.load('/data/nnair/model/cnn_imu_new.pt')['state_dict']
            print("network loaded from cnn_imu.pt")
        '''
        '''
        list_layers = ['conv_LA_1_1.weight', 'conv_LA_1_1.bias', 'conv_LA_1_2.weight', 'conv_LA_1_2.bias',
                           'conv_LA_2_1.weight', 'conv_LA_2_1.bias', 'conv_LA_2_2.weight', 'conv_LA_2_2.bias',
                           'conv_LL_1_1.weight', 'conv_LL_1_1.bias', 'conv_LL_1_2.weight', 'conv_LL_1_2.bias',
                           'conv_LL_2_1.weight', 'conv_LL_2_1.bias', 'conv_LL_2_2.weight', 'conv_LL_2_2.bias',
                           'conv_N_1_1.weight', 'conv_N_1_1.bias', 'conv_N_1_2.weight', 'conv_N_1_2.bias',
                           'conv_N_2_1.weight', 'conv_N_2_1.bias', 'conv_N_2_2.weight', 'conv_N_2_2.bias',
                           'conv_RA_1_1.weight', 'conv_RA_1_1.bias', 'conv_RA_1_2.weight', 'conv_RA_1_2.bias',
                           'conv_RA_2_1.weight', 'conv_RA_2_1.bias', 'conv_RA_2_2.weight', 'conv_RA_2_2.bias',
                           'conv_RL_1_1.weight', 'conv_RL_1_1.bias', 'conv_RL_1_2.weight',  'conv_RL_1_2.bias',
                           'conv_RL_2_1.weight', 'conv_RL_2_1.bias', 'conv_RL_2_2.weight', 'conv_RL_2_2.bias',
                           'fc3_LA.weight', 'fc3_LA.bias', 'fc3_LL.weight', 'fc3_LL.bias', 'fc3_N.weight', 'fc3_N.bias',
                           'fc3_RA.weight', 'fc3_RA.bias', 'fc3_RL.weight', 'fc3_RL.bias', 'fc4.weight', 'fc4.bias',
                           'fc5.weight', 'fc5.bias']
        '''
        list_layers = ['conv1_1.weight', 'conv1_1.bias', 'conv1_2.weight', 'conv1_2.bias',
                           'conv2_1.weight', 'conv2_1.bias', 'conv2_2.weight', 'conv2_2.bias',
                           'fc3.weight', 'fc3.bias', 'fc4.weight', 'fc4.bias',
                           'fc5.weight', 'fc5.bias']
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in list_layers}
        print(pretrained_dict)

        logging.info('        Network_User:        Pretrained layers selected')
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        logging.info('        Network_User:        Pretrained layers selected')
        # 3. load the new state dict
        network_obj.load_state_dict(model_dict)
        logging.info('        Network_User:        Weights loaded')
        
        #network_obj.eval()
        print(network_obj)
        #network_obj= nn.Sequential(*list(network_obj.children())[:-1])
        #print(network_obj)
        #print(network_obj.conv_LA_1_1.weight)
        logging.info('        Network_User:    Train:    network layers')
        '''
        for l in list(network_obj.named_parameters()):
            logging.info('        Network_User:    Train:    {} : {}'.format(l[0], l[1].detach().numpy().shape))
        '''        
        network_obj.eval()
        network_obj.to(self.device)
        #print(network_obj)
        '''
        
        # Creating a network and loading the weights for testing
        # network is loaded from saved file in the folder of experiment
        logging.info('        Network_User:    Test:    creating network')
        if self.config['network'] == 'cnn' or self.config['network'] == 'cnn_imu':
            network_obj = Network(self.config)

            #Loading the model
            network_obj.load_state_dict(torch.load(self.config['folder_exp'] + 'network.pt')['state_dict'])
            network_obj.eval()
        logging.info('        Network_User:    Test:    setting device')
        network_obj.to(self.device)
        '''
        
        # Setting loss, only for being measured. Network wont be trained
        if self.config['output'] == 'softmax':
            logging.info('        Network_User:    Test:    setting criterion optimizer Softmax')
            criterion = nn.CrossEntropyLoss()
        elif self.config['output'] == 'attribute':
            logging.info('        Network_User:    Test:    setting criterion optimizer Attribute')
            criterion = nn.BCELoss()

        loss_test = 0
        
        count_pos_test = [0, 0, 0, 0, 0, 0, 0, 0]
        count_neg_test = [0, 0, 0, 0, 0, 0, 0, 0]
        
        # Creating metric object
        if self.config['output'] == 'softmax':
            metrics_obj = Metrics(self.config, self.device)
        elif self.config['output'] == 'attribute': 
            metrics_obj = Metrics(self.config, self.device, self.attrs)
            
        #dict_all=[]
        
        logging.info('        Network_User:    Testing')
        start_time_test = time.time()
        # loop for testing
        with torch.no_grad():
            for v, harwindow_batched_test in enumerate(dataLoader_test):
                #Selecting batch
                test_batch_v = harwindow_batched_test["data"]
                if self.config['output'] == 'softmax':
                    if self.config["fully_convolutional"] == "FCN":
                        test_batch_l = harwindow_batched_test["labels"][:, 0]
                        test_batch_l = test_batch_l.reshape(-1)
                    elif self.config["fully_convolutional"] == "FC":
                        test_batch_l = harwindow_batched_test["label"]
                        test_batch_l = test_batch_l.reshape(-1)
                elif self.config['output'] == 'attribute':
                    if self.config["fully_convolutional"] == "FCN":
                        test_batch_l = harwindow_batched_test["labels"]
                    elif self.config["fully_convolutional"] == "FC":
                        sample = harwindow_batched_test["label"]
                        sample = sample.reshape(-1)
                        test_batch_l =np.zeros([sample.shape[0],self.config['num_attributes']+1])
                        
                        for i in range(0,sample.shape[0]):
                            if sample[i]==self.attrs[sample[i],0]:
                                n=sample[i].item()
                                test_batch_l [i]= self.attrs[n]
                        
                        
                # Sending to GPU
                test_batch_v = test_batch_v.to(self.device, dtype=torch.float)
                if self.config['output'] == 'softmax':
                    test_batch_l = test_batch_l.to(self.device, dtype=torch.long)
                    # labels for crossentropy needs long type
                elif self.config['output'] == 'attribute':
                    # labels for binerycrossentropy needs float type
                    test_batch_l=torch.from_numpy(test_batch_l)
                    test_batch_l = test_batch_l.to(self.device, dtype=torch.float)

                #forward
                
                predictions = network_obj(test_batch_v)
                
                if self.config['output'] == 'softmax':
                    loss = criterion(predictions, test_batch_l)
                    
                elif self.config['output'] == 'attribute':
                    loss = criterion(predictions, test_batch_l[:, 1:])
                loss_test = loss_test + loss.item()

                # Summing the loss
                loss_test = loss_test + loss.item()
          
                act_class=harwindow_batched_test["act_label"] 
                act_class = act_class.reshape(-1)
                
                if self.config['output'] == 'softmax':
                    pred_index= predictions.argmax(1)
                    
                    label=test_batch_l
                    for i,x in enumerate(pred_index):
                        if pred_index[i]==label[i]:
                           for c,z in enumerate(count_pos_test):
                                if c==act_class[i]:
                                    count_pos_test[c]+=1
                        else:
                            for c,z in enumerate(count_neg_test):
                                if c==act_class[i]:
                                    count_neg_test[c]+=1
                elif self.config['output'] == 'attribute':
                    pred=np.zeros([predictions.shape[0],predictions.shape[1]])
                    pred=torch.from_numpy(pred)
                    pred=pred.to(self.device, dtype=torch.float)
                    for i in range(predictions.shape[0]):
                      pred[i]= (predictions[i]>0.5).float()
                    label=test_batch_l[:,1:]
                    for i,k in enumerate([pred.shape[0]]):
                        if torch.all(pred[i].eq(label[i])):
                           for c,z in enumerate(count_pos_test):
                                if c==act_class[i]:
                                    count_pos_test[c]+=1
                        else:
                            for c,z in enumerate(count_neg_test):
                                if c==act_class[i]:
                                    count_neg_test[c]+=1

                # Concatenating all of the batches for computing the metrics for the entire testing set
                # and not only for a batch
                # As creating an empty tensor and sending to device and then concatenating isnt working
                #dict_all=[]
                if v == 0:
                    
                    predictions_test = predictions
                    if self.config['output'] == 'softmax':
                        test_labels = harwindow_batched_test["label"]
                        test_labels = test_labels.reshape(-1)
                        d= harwindow_batched_test["data"].numpy()
                        l= harwindow_batched_test["label"].numpy()
                        al= harwindow_batched_test["act_label"].numpy()
                        p= predictions.detach().cpu().numpy()
        
                        '''
                        print("first time")
                        print(d.shape)
                        print(l.shape)
                        print(al.shape)
                        print(p.shape)
                        '''
                        '''
                        for i in range(len(l)):
                            dict={"data": d[i], "label": l[i], "act_label": al[i], "pred": p[i]}
                            dict_all.append(dict)
                        '''
        
                    elif self.config['output'] == 'attribute':
                        sample = harwindow_batched_test["label"]
                        sample = sample.reshape(-1)
                        test_labels =np.zeros([sample.shape[0],self.config['num_attributes']+1])
                        
                        for i in range(0,sample.shape[0]):
                            if sample[i]==self.attrs[sample[i],0]:
                                n=sample[i].item()
                                test_labels[i]= self.attrs[n]
                        test_labels=torch.from_numpy(test_labels)   
                else:
                    predictions_test = torch.cat((predictions_test, predictions), dim=0)
                    if self.config['output'] == 'softmax':
                        test_labels_batch = harwindow_batched_test["label"]
                        test_labels_batch = test_labels_batch.reshape(-1)
                        a= harwindow_batched_test["data"].numpy()
                        b= harwindow_batched_test["label"].numpy()
                        c= harwindow_batched_test["act_label"].numpy()
                        pre= predictions.detach().cpu().numpy()
                        '''
                        for i in range(len(b)):
                            dict={"data": a[i], "label": b[i], "act_label": c[i], "pred": d[i]}
                            dict_all.append(dict)
                        '''
                    elif self.config['output'] == 'attribute':
                        sample = harwindow_batched_test["label"]
                        sample = sample.reshape(-1)
                        test_labels_batch =np.zeros([sample.shape[0],self.config['num_attributes']+1])
                        
                        for i in range(0,sample.shape[0]):
                            if sample[i]==self.attrs[sample[i],0]:
                                n=sample[i].item()
                                test_labels_batch[i]= self.attrs[n]
                        test_labels_batch=torch.from_numpy(test_labels_batch) 
                      
                    test_labels = torch.cat((test_labels, test_labels_batch), dim=0)
                    d=np.concatenate((d,a), axis=0)
                    l=np.concatenate((l,b),)
                    al=np.concatenate((al,c), axis=0)
                    p=np.concatenate((p,pre), axis=0)
                    
                    '''
                    print("hence forth")
                    print(d.shape)
                    print(l.shape)
                    print(al.shape)
                    print(p.shape)
                    '''
                              
                sys.stdout.write("\rTesting: Batch  {}/{}".format(v, len(dataLoader_test)))
                sys.stdout.flush()
        
        
        '''
        print("final")
        print(d.shape)
        print(l.shape)
        print(al.shape)
        print(p.shape)
        '''
        
        elapsed_time_test = time.time() - start_time_test

        #Computing metrics for the entire testing set
        test_labels = test_labels.to(self.device, dtype=torch.float)
        logging.info('            Train:    type targets vector: {}'.format(test_labels.type()))
        results_test = metrics_obj.metric(test_labels, predictions_test)

        # print statistics
        if self.config['output'] == 'softmax':
            logging.info(
            '        Network_User:        Testing:    elapsed time {} acc {}, f1_weighted {}, f1_mean {}'.format(
                elapsed_time_test, results_test['acc'], results_test['f1_weighted'], results_test['f1_mean']))
        elif self.config['output'] == 'attribute':
            logging.info(
            '        Network_User:        Testing:    elapsed time {} acc {}, f1_weighted {}, f1_mean {}, acc_attr {}'.format(
                elapsed_time_test, results_test['acc'], results_test['f1_weighted'], results_test['f1_mean'], results_test['acc_attrs'] ))
        

        #predictions_labels = torch.argmax(predictions_test, dim=1)
        predictions_labels = results_test['predicted_classes'].to("cpu", torch.double).numpy()
        test_labels = test_labels.to("cpu", torch.double).numpy()
        if self.config['output'] == 'softmax':
            test_labels = test_labels
        elif self.config['output'] == 'attribute':
            test_labels = test_labels[:, 0]
        
        #print("testlabels shape")
        #print(test_labels.shape)
        
        if self.config["dataset"]=='mocap':
            npz_file = "/data/nnair/lrp/exp1/cnn_mocap.npz"
        elif self.config["dataset"]=='mbientlab':
            npz_file = "/data/nnair/lrp/exp1/cnn_imu.npz"
        
        np.savez(npz_file, d=d, l=l, al=al, p=p)
        
        if self.config["dataset"]=='mocap':
            with np.load("/data/nnair/lrp/exp1/cnn_mocap.npz") as data:
                 d=data['d']
                 l=data['l']
                 al=data['al']
                 p=data['p']
        elif self.config["dataset"]=='mbientlab':
            with np.load("/data/nnair/lrp/exp1/cnn_imu.npz") as data:
                d=data['d']
                l=data['l']
                al=data['al']
                p=data['p']
        
        
        counterp0=[]
        counterp1=[]
        counterp2=[]
        counterp3=[]
        counterp4=[]
        counterp5=[]
        counterp6=[]
        counterp7=[]
        countern0=[]
        countern1=[]
        countern2=[]
        countern3=[]
        countern4=[]
        countern5=[]
        countern6=[]
        countern7=[]

        indxp0=[]
        indxp1=[]
        indxp2=[]
        indxp3=[]
        indxp4=[]
        indxp5=[]
        indxp6=[]
        indxp7=[]
        indxn0=[]
        indxn1=[]
        indxn2=[]
        indxn3=[]
        indxn4=[]
        indxn5=[]
        indxn6=[]
        indxn7=[]
        
        for i in range(len(l)):
            k=p[i]
            #print(k)
            for j in range(len(k)):
                if j==0:
                    if (l[i] == 0) and (np.argmax(k)==0):
                        counterp0.append(k[j])
                        indxp0.append(i)
                    elif (l[i] == 0) and (np.argmax(k) !=0):
                        countern0.append(k[j])
                        indxn0.append(i)
                elif j==1:
                    if (l[i] == 1) and (np.argmax(k)==1):
                        counterp1.append(k[j])
                        indxp1.append(i)
                    elif (l[i] == 1) and (np.argmax(k)!=1):
                        countern1.append(k[j])
                        indxn1.append(i)
                elif j==2:
                    if (l[i] == 2) and (np.argmax(k)==2):
                        counterp2.append(k[j])
                        indxp2.append(i)
                    elif (l[i] == 2) and (np.argmax(k)!=2):
                        countern2.append(k[j])
                        indxn2.append(i)
                elif j==3:
                    if (l[i] == 3) and (np.argmax(k)==3):
                        counterp3.append(k[j])
                        indxp3.append(i)
                    elif (l[i] == 3) and (np.argmax(k)!=3):
                        countern3.append(k[j])
                        indxn3.append(i)
                elif j==4:
                    if (l[i] == 4) and (np.argmax(k)==4):
                        counterp4.append(k[j])
                        indxp4.append(i)
                    elif (l[i] == 4) and (np.argmax(k)==4):
                        countern4.append(k[j])
                        indxn4.append(i)
                elif j==5:
                    if (l[i] == 5) and (np.argmax(k)==5):
                        counterp5.append(k[j])
                        indxp5.append(i)
                    elif (l[i] == 5) and (np.argmax(k)==5):
                        countern5.append(k[j])
                        indxn5.append(i)
                elif j==6:    
                    if (l[i] == 6) and (np.argmax(k)==6):
                        counterp6.append(k[j])
                        indxp6.append(i)
                    elif (l[i] == 6) and (np.argmax(k)==6):
                        countern6.append(k[j])
                        indxn6.append(i)
                elif j==7:
                    if (l[i] == 7) and (np.argmax(k)==7):
                        counterp7.append(k[j])
                        indxp7.append(i)
                    elif (l[i] == 7) and (np.argmax(k)==7):
                        countern7.append(k[j])
                        indxn7.append(i)
                        
                  
        b_div=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        
        fig1,axs1 = plt.subplots(1,1, figsize=(10,7), tight_layout= True)
        axs1.hist(counterp0, bins= b_div)
        plt.xlabel("Softmax")
        plt.ylabel("No: of values")
        plt.title('Sub 0 - +ve')
        plt.savefig("i_sub0_p.png")
        
        fig2,axs2 = plt.subplots(1,1, figsize=(10,7), tight_layout= True)
        axs2.hist(countern0, bins= b_div)
        plt.xlabel("Softmax")
        plt.ylabel("No: of values")
        plt.title('Sub 0 - -ve')
        plt.savefig("i_sub0_n.png")
        
        fig3,axs3 = plt.subplots(1,1, figsize=(10,7), tight_layout= True)
        axs3.hist(counterp1, bins= b_div)
        plt.xlabel("Softmax")
        plt.ylabel("No: of values")
        plt.title('Sub 1 - +ve')
        plt.savefig("i_sub1_p.png")
        
        fig4,axs4 = plt.subplots(1,1, figsize=(10,7), tight_layout= True)
        axs4.hist(countern1, bins= b_div)
        plt.xlabel("Softmax")
        plt.ylabel("No: of values")
        plt.title('Sub 1 - -ve')
        plt.savefig("i_sub1_n.png")
        
        fig5,axs5 = plt.subplots(1,1, figsize=(10,7), tight_layout= True)
        axs5.hist(counterp2, bins= b_div)
        plt.xlabel("Softmax")
        plt.ylabel("No: of values")
        plt.title('Sub 2 - +ve')
        plt.savefig("i_sub2_p.png")
        
        fig6,axs6 = plt.subplots(1,1, figsize=(10,7), tight_layout= True)
        axs6.hist(countern2, bins= b_div)
        plt.xlabel("Softmax")
        plt.ylabel("No: of values")
        plt.title('Sub 2 - -ve')
        plt.savefig("i_sub2_n.png")
        
        fig7,axs7 = plt.subplots(1,1, figsize=(10,7), tight_layout= True)
        axs7.hist(counterp3, bins= b_div)
        plt.xlabel("Softmax")
        plt.ylabel("No: of values")
        plt.title('Sub 3 - +ve')
        plt.savefig("i_sub3_p.png")
        
        fig8,axs8 = plt.subplots(1,1, figsize=(10,7), tight_layout= True)
        axs8.hist(countern3, bins= b_div)
        plt.xlabel("Softmax")
        plt.ylabel("No: of values")
        plt.title('Sub 3 - -ve')
        plt.savefig("i_sub3_n.png")
        
        fig9,axs9 = plt.subplots(1,1, figsize=(10,7), tight_layout= True)
        axs9.hist(counterp4, bins= b_div)
        plt.xlabel("Softmax")
        plt.ylabel("No: of values")
        plt.title('Sub 4 - +ve')
        plt.savefig("i_sub4_p.png")
        
        fig10,axs10 = plt.subplots(1,1, figsize=(10,7), tight_layout= True)
        axs10.hist(countern4, bins= b_div)
        plt.xlabel("Softmax")
        plt.ylabel("No: of values")
        plt.title('Sub 4 - -ve')
        plt.savefig("i_sub4_n.png")
        
        fig11,axs11 = plt.subplots(1,1, figsize=(10,7), tight_layout= True)
        axs11.hist(counterp5, bins= b_div)
        plt.xlabel("Softmax")
        plt.ylabel("No: of values")
        plt.title('Sub 5 - +ve')
        plt.savefig("i_sub5_p.png")
        
        fig12,axs12 = plt.subplots(1,1, figsize=(10,7), tight_layout= True)
        axs12.hist(countern5, bins= b_div)
        plt.xlabel("Softmax")
        plt.ylabel("No: of values")
        plt.title('Sub 5 - -ve')
        plt.savefig("i_sub5_n.png")
        
        fig13,axs13 = plt.subplots(1,1, figsize=(10,7), tight_layout= True)
        axs13.hist(counterp6, bins= b_div)
        plt.xlabel("Softmax")
        plt.ylabel("No: of values")
        plt.title('Sub 6 - +ve')
        plt.savefig("i_sub6_p.png")
        
        fig14,axs14 = plt.subplots(1,1, figsize=(10,7), tight_layout= True)
        axs14.hist(countern6, bins= b_div)
        plt.xlabel("Softmax")
        plt.ylabel("No: of values")
        plt.title('Sub 6 - -ve')
        plt.savefig("i_sub6_n.png")
        
        fig15,axs15 = plt.subplots(1,1, figsize=(10,7), tight_layout= True)
        axs15.hist(counterp7, bins= b_div)
        plt.xlabel("Softmax")
        plt.ylabel("No: of values")
        plt.title('Sub 7 - +ve')
        plt.savefig("i_sub7_p.png")
        
        fig16,axs16 = plt.subplots(1,1, figsize=(10,7), tight_layout= True)
        axs16.hist(countern7, bins= b_div)
        plt.xlabel("Softmax")
        plt.ylabel("No: of values")
        plt.title('Sub 7 - -ve')
        plt.savefig("i_sub7_n.png")
        
    
        
        
                    
        '''    
        if self.config['output'] == 'attribute':
            if self.config['num_attributes'] == 4:
                for i in range(0, test_labels[0]):
                    if test_labels[i,0]==6:
                        test_labels[i,0]=5
                    elif test_labels[i,0]==7:
                        test_labels[i,0]=3
            elif self.config['num_attributes'] == 10:
                for i in range(0, test_labels.shape[0]):
                    if test_labels[i,0]==6:
                        test_labels[i,0]=5
                    elif test_labels[i,0]==7:
                       test_labels[i,0]=6
        '''
        
        # Computing confusion matrix
        confusion_matrix = np.zeros((self.config['num_classes'], self.config['num_classes']))
        for cl in range(self.config['num_classes']):
            pos_tg = test_labels == cl
            pos_pred = predictions_labels[pos_tg]
            bincount = np.bincount(pos_pred.astype(int), minlength=self.config['num_classes'])
            confusion_matrix[cl, :] = bincount

        logging.info("        Network_User:        Testing:    Confusion matrix \n{}\n".format(confusion_matrix.astype(int)))

        percentage_pred = []
        for cl in range(self.config['num_classes']):
            pos_trg = np.reshape(test_labels, newshape=test_labels.shape[0]) == cl
            percentage_pred.append(confusion_matrix[cl, cl] / float(np.sum(pos_trg)))
        percentage_pred = np.array(percentage_pred)

        logging.info("        Network_User:        Validating:    percentage Pred \n{}\n".format(percentage_pred))

        #plot predictions
        if self.config["plotting"]:
            fig = plt.figure()
            axis_test = fig.add_subplot(111)
            plot_trg = axis_test.plot([], [],'-r',label='trg')[0]
            #plot_pred = axis_test.plot([], [],'-b',label='pred')[0]

            plot_trg.set_ydata(test_labels)
            plot_trg.set_xdata(range(test_labels.shape[0]))

            #plot_pred.set_ydata(predictions_labels)
            #plot_pred.set_xdata(range(predictions_labels.shape[0]))

            axis_test.relim()
            axis_test.autoscale_view()
            axis_test.legend(loc='best')

            fig.canvas.draw()
            plt.pause(2.0)
            axis_test.cla()

        del test_batch_v, test_batch_l
        del predictions, predictions_test
        del test_labels, predictions_labels
        
        
        del network_obj

        torch.cuda.empty_cache()

        return results_test, confusion_matrix.astype(int), count_pos_test, count_neg_test
        #return
    
    def lrp(self):
        logging.info('        Network_User:    LRP ---->')
        logging.info('        Network_User:    LRP:    creating network')
        
        network_obj = Network(self.config)
        network_obj.init_weights()
        model_dict = network_obj.state_dict()
        print("model dict with state dict loaded")
        #print(network_obj)
        #print(network_obj.conv_LA_1_1.weight)
        if self.config["dataset"]=='mocap':
            #network_obj.load_state_dict(torch.load('/data/nnair/model/model_save_mocap.pt'))
            pretrained_dict= torch.load('/data/nnair/model/cnn_mocap.pt')['state_dict']
            print("network loaded from cnn_mocap.pt")
        elif self.config["dataset"]=='mbientlab':
            #network_obj.load_state_dict(torch.load('/data/nnair/model/model_save_imu.pt'))
            pretrained_dict= torch.load('/data/nnair/model/cnn_imu_new.pt')['state_dict']
            print("network loaded from cnn_imu_new.pt")
        '''   
        list_layers = ['conv_LA_1_1.weight', 'conv_LA_1_1.bias', 'conv_LA_1_2.weight', 'conv_LA_1_2.bias',
                           'conv_LA_2_1.weight', 'conv_LA_2_1.bias', 'conv_LA_2_2.weight', 'conv_LA_2_2.bias',
                           'conv_LL_1_1.weight', 'conv_LL_1_1.bias', 'conv_LL_1_2.weight', 'conv_LL_1_2.bias',
                           'conv_LL_2_1.weight', 'conv_LL_2_1.bias', 'conv_LL_2_2.weight', 'conv_LL_2_2.bias',
                           'conv_N_1_1.weight', 'conv_N_1_1.bias', 'conv_N_1_2.weight', 'conv_N_1_2.bias',
                           'conv_N_2_1.weight', 'conv_N_2_1.bias', 'conv_N_2_2.weight', 'conv_N_2_2.bias',
                           'conv_RA_1_1.weight', 'conv_RA_1_1.bias', 'conv_RA_1_2.weight', 'conv_RA_1_2.bias',
                           'conv_RA_2_1.weight', 'conv_RA_2_1.bias', 'conv_RA_2_2.weight', 'conv_RA_2_2.bias',
                           'conv_RL_1_1.weight', 'conv_RL_1_1.bias', 'conv_RL_1_2.weight',  'conv_RL_1_2.bias',
                           'conv_RL_2_1.weight', 'conv_RL_2_1.bias', 'conv_RL_2_2.weight', 'conv_RL_2_2.bias',
                           'fc3_LA.weight', 'fc3_LA.bias', 'fc3_LL.weight', 'fc3_LL.bias', 'fc3_N.weight', 'fc3_N.bias',
                           'fc3_RA.weight', 'fc3_RA.bias', 'fc3_RL.weight', 'fc3_RL.bias', 'fc4.weight', 'fc4.bias',
                           'fc5.weight', 'fc5.bias']
        '''
        list_layers = ['conv1_1.weight', 'conv1_1.bias', 'conv1_2.weight', 'conv1_2.bias',
                           'conv2_1.weight', 'conv2_1.bias', 'conv2_2.weight', 'conv2_2.bias',
                           'fc3.weight', 'fc3.bias', 'fc4.weight', 'fc4.bias',
                           'fc5.weight', 'fc5.bias']
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in list_layers}
        
        logging.info('        Network_User:        Pretrained layers selected')
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        logging.info('        Network_User:        Pretrained layers selected')
        # 3. load the new state dict
        network_obj.load_state_dict(model_dict)
        logging.info('        Network_User:        Weights loaded')
        
        
        #network_obj= nn.Sequential(*list(network_obj.children())[:-1])
        print(network_obj)
        #print(network_obj.conv_LA_1_1.weight)
        logging.info('        Network_User:    Train:    network layers')
        '''
        for l in list(network_obj.named_parameters()):
            logging.info('        Network_User:    Trained:    {} : {}'.format(l[0], l[1].detach().numpy().shape))
        '''        
        network_obj.eval()
        
        logging.info('        Network_User:    Test:    setting device')
        network_obj.to(self.device)
        
        # Setting loss, only for being measured. Network wont be trained
        if self.config['output'] == 'softmax':
            logging.info('        Network_User:    Test:    setting criterion optimizer Softmax')
            criterion = nn.CrossEntropyLoss()
        elif self.config['output'] == 'attribute':
            logging.info('        Network_User:    Test:    setting criterion optimizer Attribute')
            criterion = nn.BCELoss()
            
        # Creating metric object
        if self.config['output'] == 'softmax':
            metrics_obj = Metrics(self.config, self.device)
        elif self.config['output'] == 'attribute': 
            metrics_obj = Metrics(self.config, self.device, self.attrs)
            
        if self.config["dataset"]=='mocap':
            npz_file = "/data/nnair/lrp/exp1/cnn_mocap.npz"
            print("mocap output loaded")
        elif self.config["dataset"]=='mbientlab':
            npz_file = "/data/nnair/lrp/exp1/cnn_imu.npz"
            print("imu output loaded")
        
        with np.load(npz_file) as data:
            d=data['d']
            l=data['l']
            al=data['al']
            p=data['p']
        
        
        counterp0=[]
        counterp1=[]
        counterp2=[]
        counterp3=[]
        counterp4=[]
        counterp5=[]
        counterp6=[]
        counterp7=[]
        countern0=[]
        countern1=[]
        countern2=[]
        countern3=[]
        countern4=[]
        countern5=[]
        countern6=[]
        countern7=[]

        indxp0=[]
        indxp1=[]
        indxp2=[]
        indxp3=[]
        indxp4=[]
        indxp5=[]
        indxp6=[]
        indxp7=[]
        indxn0=[]
        indxn1=[]
        indxn2=[]
        indxn3=[]
        indxn4=[]
        indxn5=[]
        indxn6=[]
        indxn7=[]
        
        for i in range(len(l)):
            k=p[i]
            #print(k)
            for j in range(len(k)):
                if j==0:
                    if (l[i] == 0) and (np.argmax(k)==0):
                        counterp0.append(k[j])
                        indxp0.append(i)
                    elif (l[i] == 0) and (np.argmax(k) !=0):
                        countern0.append(k[j])
                        indxn0.append(i)
                elif j==1:
                    if (l[i] == 1) and (np.argmax(k)==1):
                        counterp1.append(k[j])
                        indxp1.append(i)
                    elif (l[i] == 1) and (np.argmax(k)!=1):
                        countern1.append(k[j])
                        indxn1.append(i)
                elif j==2:
                    if (l[i] == 2) and (np.argmax(k)==2):
                        counterp2.append(k[j])
                        indxp2.append(i)
                    elif (l[i] == 2) and (np.argmax(k)!=2):
                        countern2.append(k[j])
                        indxn2.append(i)
                elif j==3:
                    if (l[i] == 3) and (np.argmax(k)==3):
                        counterp3.append(k[j])
                        indxp3.append(i)
                    elif (l[i] == 3) and (np.argmax(k)!=3):
                        countern3.append(k[j])
                        indxn3.append(i)
                elif j==4:
                    if (l[i] == 4) and (np.argmax(k)==4):
                        counterp4.append(k[j])
                        indxp4.append(i)
                    elif (l[i] == 4) and (np.argmax(k)==4):
                        countern4.append(k[j])
                        indxn4.append(i)
                elif j==5:
                    if (l[i] == 5) and (np.argmax(k)==5):
                        counterp5.append(k[j])
                        indxp5.append(i)
                    elif (l[i] == 5) and (np.argmax(k)==5):
                        countern5.append(k[j])
                        indxn5.append(i)
                elif j==6:    
                    if (l[i] == 6) and (np.argmax(k)==6):
                        counterp6.append(k[j])
                        indxp6.append(i)
                    elif (l[i] == 6) and (np.argmax(k)==6):
                        countern6.append(k[j])
                        indxn6.append(i)
                elif j==7:
                    if (l[i] == 7) and (np.argmax(k)==7):
                        counterp7.append(k[j])
                        indxp7.append(i)
                    elif (l[i] == 7) and (np.argmax(k)==7):
                        countern7.append(k[j])
                        indxn7.append(i)
        
        '''
        for i in range(len(indxp0)):
            if counterp0[i]>=0.9:
                print("positive value and index greater than 0.9")
                print(counterp0[i])
                print(indxp0[i])
                print(al[indxp0[i]])
            elif counterp0[i]>=0.4 and counterp0[i]<=0.5:
                print("positive value and index greater than 0.4 and less than 0.5")
                print(counterp0[i])
                print(indxp0[i])
                print(al[indxp0[i]])
        
        for i in range(len(indxn0)):
            if countern0[i]>=0.2 and countern0[i]<=0.3:
                print("neg value and index greater than 0.2 and less than 0.3")
                print(countern0[i])
                print(indxn0[i])
                print(al[indxn0[i]])
            elif countern0[i]>=0.4 and countern0[i]<=0.5:
                print("neg value and index greater than 0.4 and less than 0.5")
                print(countern0[i])
                print(indxn0[i])
                print(al[indxn0[i]])
        '''
        
        #print("check activity")
        '''
        print("window 341")
        print(indxn0[0])
        print(al[indxn0[0]])
        print("\n window 713")
        print(indxn0[4])
        print(al[indxn0[4]])
        
        print("\n window 910")
        print(indxn0[5])
        print(al[indxn0[5]])
        
        print("\n window 916")
        print(indxn0[11])
        print(al[indxn0[11]])
        
        print("\n window 925")
        print(indxn0[16])
        print(al[indxn0[16]])
        '''
        '''
        print("subject 0 windows with activity cart")
        for i in range(len(indxp5)):
            if al[indxp5[i]] ==2:
                print(indxp5[i])
        '''
        '''
        
        lrp_test_indx=[]
        
        
        
        
        for i in range(len(indxp0)):
            #947
            if indxp0[i] == 225:
                lrp_test_indx.append(indxp0[i])
            elif indxp0[i] == 39:
                lrp_test_indx.append(indxp0[i])
                
                
        for i in range (len(indxn0)):
            if indxn0[i] == 341:
                lrp_test_indx.append(indxn0[i])
            elif indxn0[i] == 343:
                lrp_test_indx.append(indxn0[i])
        
        
        print("selected indexes")
        print(lrp_test_indx)
        
        
        for i in range(len(lrp_test_indx)):
        
            
            test_v=d[lrp_test_indx[i]]
            test_l=l[lrp_test_indx[i]]
            test_act = al[lrp_test_indx[i]]
            test_pred = p[lrp_test_indx[i]]
            print("test subject")
            print(lrp_test_indx[i])
            print(test_l)
            print(test_act)
            print(test_pred)
         
        '''      
        
        v=4910
        print(v)
        test_v=d[v]
        test_l=l[v]
        test_act = al[v]
        test_pre =p[v]
        print("test subject")
        print(test_l)
        print(test_act)
        print(test_pre)
        #test_act=al[lrp_test_indx[0]]
        '''
        '''
        print(6000)
        test_v=d[6000]
        test_l=l[6000]
        print("test subject")
        print(test_l)
        test_act=al[6000]
        '''
        '''
        test_v= torch.from_numpy(test_v)
        test_v= test_v.to(self.device, dtype=torch.float)
        test_l= np.array(test_l, dtype=np.float64)
        test_l= torch.from_numpy(test_l)
        test_l= test_l.to(self.device, dtype=torch.long) 
       
        
        layers= [module for module in network_obj.modules()]
        #first layer is just the network layout
        L=len(layers)
        #print(layers)
        #31 including networklayout, avgpool and sigmoid
        
        convlayers=layers[1:5]
        cl=len(convlayers)
        fc=layers[5:8]
        fcl=len(fc)
        #print("check")
        #print(convlayers)
        #print(fc)
        sm=layers[9]
        #print(sm)
        
        
        ##############################################setting input
        
        test_v = test_v.unsqueeze(0)
        
        A=[test_v] + [None]*(cl*2)
        fA=[None]*(5)
        j=1
        for i in range(cl):
            A[j]= convlayers[i].forward(A[j-1])
            A[j+1]=F.relu(A[j])
            j+=2
        #print(A)
        #print(len(A))    
        A[8] = A[8].reshape(-1, A[8].size()[1] * A[8].size()[2] * A[8].size()[3])
        fA[0]=fc[0].forward(A[8])
        fA[1]=F.relu(fA[0])
        fA[2]=fc[1].forward(fA[1])
        fA[3]=F.relu(fA[2])
        fA[4]=fc[2].forward(fA[3])
        
        #print(fA)
        #print(len(fA))
        sml=sm.forward(fA[4])
        print("predicted")
        print(test_pre)
        print("sml")
        print(sml)
        
        print("Relevance part")
        
        T = sml.cpu().detach().numpy().tolist()[0]
        index = T.index(max(T))
        #print("index")
        #print(index)
        T = np.abs(np.array(T)) * 0
        T[index] = 1
        T = torch.FloatTensor(T)
        # Create the list of relevances with (L + 1) elements and assign the value of the last one 
        R_fc = [None] * (3) + [(sml.cpu() * T).data + 1e-6]
        #print("R_fc")
        #print(R_fc)
        R_fc[2]=self.relprop(fA[3], fc[2], R_fc[3])
        #print(R_fc)
        R_fc[1]=self.relprop(fA[1], fc[1], R_fc[2])
        R_fc[0]=self.relprop(A[8], fc[0], R_fc[1])
        R_f = R_fc[0].reshape(1, 64, 84, -1)       
        #print(R_fc)
        R=[None]*5
        
        R[3]=self.relprop(A[6], convlayers[3], R_f)
        R[2]=self.relprop(A[4], convlayers[2], R[3])
        R[1]=self.relprop(A[2], convlayers[1], R[2])
        R[0]=self.relprop(A[0], convlayers[0], R[1])
        
        print(R[0])
        p=R[0]
        p.numpy()
        p = np.reshape(p, newshape=(p.shape[2], p.shape[3]))
        print(p.shape)
        tv=test_v.cpu()
        
        tv.numpy()
        tv = np.reshape(tv, newshape=(tv.shape[2], tv.shape[3]))
        print(tv.shape)
        
        savetxt('relevance4910.csv', p, delimiter=',')
        savetxt('input4910.csv', tv, delimiter=',')
        
        '''
        
        A_LA[0] = (A_LA[0].data).requires_grad_(True)
        A_LL[0] = (A_LL[0].data).requires_grad_(True)
        A_N[0] = (A_N[0].data).requires_grad_(True)
        A_RA[0] = (A_RA[0].data).requires_grad_(True)
        A_RL[0] = (A_RL[0].data).requires_grad_(True)

        lb = (A[0].data*0+(0-mean)/std).requires_grad_(True)
        hb = (A[0].data*0+(1-mean)/std).requires_grad_(True)

z = layers[0].forward(A[0]) + 1e-9                                     # step 1 (a)
z -= utils.newlayer(layers[0],lambda p: p.clamp(min=0)).forward(lb)    # step 1 (b)
z -= utils.newlayer(layers[0],lambda p: p.clamp(max=0)).forward(hb)    # step 1 (c)
s = (R[1]/z).data                                                      # step 2
(z*s).sum().backward(); c,cp,cm = A[0].grad,lb.grad,hb.grad            # step 3
R[0] = (A[0]*c+lb*cp+hb*cm).data  
       
        
        if self.config["dataset"]=='mocap':
                idx_LA = np.arange(12, 24)
                idx_LA = np.concatenate([idx_LA, np.arange(36, 42)])
                idx_LA = np.concatenate([idx_LA, np.arange(54, 66)])
                in_LA = test_v[ :, :, :, idx_LA]
                
                idx_LL = np.arange(24, 36)
                idx_LL = np.concatenate([idx_LL, np.arange(42, 54)])
                in_LL = test_v[ :, :, :, idx_LL]
                
                idx_N = np.arange(0, 12)
                idx_N = np.concatenate([idx_N, np.arange(120, 126)])
                in_N = test_v[:, :, :, idx_N]
                
                idx_RA = np.arange(66, 78)
                idx_RA = np.concatenate([idx_RA, np.arange(90, 96)])
                idx_RA = np.concatenate([idx_RA, np.arange(108, 120)])
                in_RA = test_v[:, :, :, idx_RA]
                
                idx_RL = np.arange(78, 90)
                idx_RL = np.concatenate([idx_RL, np.arange(96, 108)])
                in_RL = test_v[:, :, :, idx_RL]
                
        elif self.config["dataset"]=='mbientlab':
                in_LA = test_v[:, :, :, 0:6]
                in_LL = test_v[:, :, :, 6:12]
                in_N = test_v[:, :, :, 12:18]
                in_RA = test_v[:, :, :, 18:24]
                in_RL = test_v[:, :, :, 24:30]
              
        #############################setting activation layers
        
        A_LA=[in_LA] + [None]*(cl1*2)
        A_LL=[in_LL] + [None]*(cl2*2)
        A_N=[in_N] + [None]*(cl3*2)
        A_RA=[in_RA] + [None]*(cl4*2)
        A_RL=[in_RL] + [None]*(cl5*2)
        
        ###################################conv layer activations
        
        j=1
        for i in range(cl1):
            A_LA[j]= convlayers1[i].forward(A_LA[j-1])
            A_LA[j+1]=F.relu(A_LA[j])
            j+=2
        #print(A_LA[8].shape)    
        A_LA[8] = A_LA[8].reshape(-1, A_LA[8].size()[1] * A_LA[8].size()[2] * A_LA[8].size()[3])
        t1=trans[0].forward(A_LA[8])
        A_t1=[t1]+[F.relu(t1)]
       
        j=1
        for i in range(cl2):
            A_LL[j]= convlayers2[i].forward(A_LL[j-1])
            A_LL[j+1]=F.relu(A_LL[j])
            j+=2
        #print(A_LL[8].shape)
        A_LL[8] = A_LL[8].reshape(-1, A_LL[8].size()[1] * A_LL[8].size()[2] * A_LL[8].size()[3])
        t2=trans[1].forward(A_LL[8])
        A_t2=[t2]+[F.relu(t2)]
        
        j=1
        for i in range(cl3):
            A_N[j]= convlayers3[i].forward(A_N[j-1])
            A_N[j+1]=F.relu(A_N[j])
            j+=2
        #print(A_N[8].shape)
        A_N[8] = A_N[8].reshape(-1, A_N[8].size()[1] * A_N[8].size()[2] * A_N[8].size()[3])
        t3=trans[2].forward(A_N[8])
        A_t3=[t3]+[F.relu(t3)]
           
        j=1
        for i in range(cl4):
            A_RA[j]= convlayers4[i].forward(A_RA[j-1])
            A_RA[j+1]=F.relu(A_RA[j])
            j+=2
        #print(A_RA[8].shape)
        A_RA[8] = A_RA[8].reshape(-1, A_RA[8].size()[1] * A_RA[8].size()[2] * A_RA[8].size()[3])
        t4=trans[3].forward(A_RA[8])
        A_t4=[t4]+[F.relu(t4)]
        
        j=1
        for i in range(cl5):
            A_RL[j]= convlayers5[i].forward(A_RL[j-1])
            A_RL[j+1]=F.relu(A_RL[j])
            j+=2
    
        A_RL[8] = A_RL[8].reshape(-1, A_RL[8].size()[1] * A_RL[8].size()[2] * A_RL[8].size()[3])  
        t5=trans[4].forward(A_RL[8])
        A_t5=[t5]+[F.relu(t5)]
        
        grouped= torch.cat((A_t1[1], A_t2[1], A_t3[1], A_t4[1], A_t5[1]), 1)
        
        fc4=fc[0].forward(grouped)
        A_fc4=[fc4]+[F.relu(fc4)]
        #print(A_fc4)
        A_fc5=fc[1].forward(A_fc4[1])
        print(A_fc5)
        sml=fc[2].forward(A_fc5)
        print(sml)
        
        print("Relevance part")
        
        T = A_fc5.cpu().detach().numpy().tolist()[0]
        index = T.index(max(T))
        T = np.abs(np.array(T)) * 0
        T[index] = 1
        T = torch.FloatTensor(T)
        # Create the list of relevances with (L + 1) elements and assign the value of the last one 
        R_fc = [None] * (2) + [(sml.cpu() * T).data + 1e-6]
        #print(R_fc)
        R_fc[1]=self.relprop(A_fc4[1], fc[1], R_fc[2])
        R_fc[0]=self.relprop(grouped, fc[0], R_fc[1])
        temp=R_fc[0]
        indx_LA=[i for i in range(0,256)]
        indx_LL=[i for i in range(256, 512)]
        indx_N=[i for i in range(512, 768)]
        indx_RA=[i for i in range(768, 1024)]
        indx_RL=[i for i in range(1024, 1280)]
        rfc_LA=temp[:, indx_LA]
        rfc_LL=temp[:, indx_LL]
        rfc_N=temp[:, indx_N]
        rfc_RA=temp[:, indx_RA]
        rfc_RL=temp[:, indx_RL]
        
        R_LA=[None]*5
        R_LL=[None]*5
        R_N=[None]*5
        R_RA=[None]*5
        R_RL=[None]*5
        
        R_LA[4]=self.relprop(A_LA[8], trans[0], rfc_LA)
        R_LL[4]=self.relprop(A_LL[8], trans[1], rfc_LL)
        R_N[4]=self.relprop(A_N[8], trans[2], rfc_N)
        R_RA[4]=self.relprop(A_RA[8], trans[3], rfc_RA)
        R_RL[4]=self.relprop(A_RL[8], trans[4], rfc_RL)
        
        R_LA[4] = R_LA[4].reshape(1, 64, 84, -1) 
        R_LL[4] = R_LL[4].reshape(1, 64, 84, -1) 
        R_N[4] = R_N[4].reshape(1, 64, 84, -1) 
        R_RA[4] = R_RA[4].reshape(1, 64, 84, -1) 
        R_RL[4] = R_RL[4].reshape(1, 64, 84, -1) 
       
        R_LA[3]=self.relprop(A_LA[6], convlayers1[3], R_LA[4])
        R_LL[3]=self.relprop(A_LL[6], convlayers2[3], R_LL[4])
        R_N[3]=self.relprop(A_N[6], convlayers3[3], R_N[4])
        R_RA[3]=self.relprop(A_RA[6], convlayers4[3], R_RA[4])
        R_RL[3]=self.relprop(A_RL[6], convlayers5[3], R_RL[4])
        
        R_LA[2]=self.relprop(A_LA[4], convlayers1[2], R_LA[3])
        R_LL[2]=self.relprop(A_LL[4], convlayers2[2], R_LL[3])
        R_N[2]=self.relprop(A_N[4], convlayers3[2], R_N[3])
        R_RA[2]=self.relprop(A_RA[4], convlayers4[2], R_RA[3])
        R_RL[2]=self.relprop(A_RL[4], convlayers5[2], R_RL[3])
        
        R_LA[1]=self.relprop(A_LA[2], convlayers1[1], R_LA[2])
        R_LL[1]=self.relprop(A_LL[2], convlayers2[1], R_LL[2])
        R_N[1]=self.relprop(A_N[2], convlayers3[1], R_N[2])
        R_RA[1]=self.relprop(A_RA[2], convlayers4[1], R_RA[2])
        R_RL[1]=self.relprop(A_RL[2], convlayers5[1], R_RL[2])
        
        R_LA[0]=self.relprop(A_LA[0], convlayers1[0], R_LA[1])
        R_LL[0]=self.relprop(A_LL[0], convlayers2[0], R_LL[1])
        R_N[0]=self.relprop(A_N[0], convlayers3[0], R_N[1])
        R_RA[0]=self.relprop(A_RA[0], convlayers4[0], R_RA[1])
        R_RL[0]=self.relprop(A_RL[0], convlayers5[0], R_RL[1])
        '''
        '''
        A_LA[0] = (A_LA[0].data).requires_grad_(True)
        A_LL[0] = (A_LL[0].data).requires_grad_(True)
        A_N[0] = (A_N[0].data).requires_grad_(True)
        A_RA[0] = (A_RA[0].data).requires_grad_(True)
        A_RL[0] = (A_RL[0].data).requires_grad_(True)

        lb = (A[0].data*0+(0-mean)/std).requires_grad_(True)
        hb = (A[0].data*0+(1-mean)/std).requires_grad_(True)

z = layers[0].forward(A[0]) + 1e-9                                     # step 1 (a)
z -= utils.newlayer(layers[0],lambda p: p.clamp(min=0)).forward(lb)    # step 1 (b)
z -= utils.newlayer(layers[0],lambda p: p.clamp(max=0)).forward(hb)    # step 1 (c)
s = (R[1]/z).data                                                      # step 2
(z*s).sum().backward(); c,cp,cm = A[0].grad,lb.grad,hb.grad            # step 3
R[0] = (A[0]*c+lb*cp+hb*cm).data  
'''
        
  
        return
       
    #def newlayer(self, layer, g):
    def newlayer(self, layer):
        """Clone a layer and pass its parameters through the function g."""
        #print("copying layer")
        layer = copy.deepcopy(layer)
        '''
        print(layer.weight)
        layer.weight = torch.nn.Parameter(g(layer.weight))
        print(layer.weights)
        layer.bias = torch.nn.Parameter(g(layer.bias))
        print(layer.bias)
        '''
        return layer
        
    def relprop(self, A, layers, R_1):
            '''
            print("A")
            print(A.shape)
            print("layers")
            print(layers)
            print("R_1")
            print(R_1.shape)
            '''
            rho= lambda p: p
            #A[layer] = A[layer].data.requires_grad_(True)
            A = A.data.requires_grad_(True)
            #print(A)
            #cpy=self.newlayer(layer=layers, g=rho)
            cpy=self.newlayer(layer=layers)
            # Step 1: Transform the weights of the layer and executes a forward pass
            z = cpy.forward(A) + 1e-9
            # Step 2: Element-wise division between the relevance of the next layer and the denominator
            s = (R_1.to(self.device) / z).data
            # Step 3: Calculate the gradient and multiply it by the activation layer
            (z * s).sum().backward()
            c = A.grad  	
            out = (A * c).cpu().data 
            
            return out

    '''
    def hook( m, i, o):
            print( m._get_name() )

        for ( mo ) in model.modules():
            mo.register_forward_hook(hook)
    
    '''

    ##################################################
    ############  evolution_evaluation  ##############
    ##################################################
    
    def evolution_evaluation(self, ea_iter=0, testing = False):
       '''
        Organises the evolution, training, testing or validating

        @param ea_itera: evolution iteration
        @param testing: Setting testing in training or only testing
        @return results: dict with validating/testing results
        @return confusion_matrix: dict with validating/testing results
        @return best_itera: best iteration for training
       '''
     
       logging.info('        Network_User: Evolution evaluation iter {}'.format(ea_iter))

       #confusion_matrix = 0
       
       #best_itera = 0
       
       #results, confusion_matrix, c_pos, c_neg = self.test(ea_iter)
       self.lrp()
       
       '''
       if testing:
            logging.info('        Network_User: Testing')
            results, confusion_matrix, c_pos, c_neg = self.test(ea_iter)
       else:
            if self.config['usage_modus'] == 'train':
                logging.info('        Network_User: Training')

                results, best_itera, c_pos, c_neg = self.train(ea_iter)

            elif self.config['usage_modus'] == 'fine_tuning':
                logging.info('        Network_User: Fine Tuning')
                results, best_itera = self.train(ea_iter)

            elif self.config['usage_modus'] == 'test':
                logging.info('        Network_User: Testing')

                results, confusion_matrix = self.test(ea_iter)

            else:
                logging.info('        Network_User: Not selected modus')
        
       '''
       #return results, confusion_matrix, best_itera, c_pos, c_neg
       return