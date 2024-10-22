# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:37:33 2022

example: 

    # main_route = 'E:\\Project_ExpNNFEM\\P1\\'
    main_route = 'C:\\westame\\Project_ExpNNFEM\\' + main_project + '\\'
    file_name ='Ep'+ str(Params.all_epoch) + '_' + Params.activate_function + '_Lr'+ str(Params.learning_r) + '_Hid' + str(Params.hid_layers)
    
    #(x, y), (x_val, y_val) = datasets.mnist.load_data() 
    train_route = main_route + 'Matlab\\DataNeuNet\\' + r_date + '\\train_10000_test_1000\\'
    # create 'train_route' related date to train MatPort_NN
    MatPort_NN = MatPort(train_route)
    # call 'auto_call_fem'  to read FEM database
    (x, x_val, y, y_val, info_xy) = MatPort_NN.auto_call_fem()

@author: westame
"""

from scipy.io import loadmat


class MatPort(object):
# matlab porter
    def __init__(self, main_route):
        self.main_route = main_route
    
    def one_call(self, file_name):
        # read .mat file without '.mat'
        call_file = self.main_route + file_name + '.mat'
        orig_mat = loadmat(call_file)
        data_mat = orig_mat[file_name]
        return data_mat
    
    def auto_call_fem(self):
        # read data .mat
        sample_train = self.one_call('sample_train')
        sample_test = self.one_call('sample_test')
        type_train = self.one_call('type_train')
        type_train = type_train[:,0]
        type_test = self.one_call('type_test')
        type_test = type_test[:,0]
        sample_info = self.one_call('sample_info')
        return (sample_train, sample_test, type_train, type_test, sample_info)
    
    def auto_call_exp(self):
        # read data .mat
        sample_go = self.one_call('sample_go')
        sample_back = self.one_call('sample_back')
        type_go = self.one_call('type_go')
        type_go = type_go[:,0]
        type_back = self.one_call('type_back')
        type_back = type_back[:,0]
        sample_info = self.one_call('sample_info')
        return (sample_go, sample_back, type_go, type_back, sample_info)

    def auto_call_2(self):
        # read data .mat
        sample_field = self.one_call('x_train_1')
        sample_loc = self.one_call('y_train')
        sample_dmg = self.one_call('z_train')
        test_field = self.one_call('x_test')
        test_loc = self.one_call('y_test')
        test_dmg = self.one_call('z_test')
        return (sample_field, sample_loc, sample_dmg, test_field, test_loc, test_dmg)