# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 21:47:45 2018

@author: DART_HSU
"""
import pandas as pd
import numpy as np

class ProcessData:
       
    def __init__(self, path_ = './data/201507-201606.csv'):
        self.path = path_
        self.df = pd.read_csv(self.path)
     
    def get_nparray(self):
        df = self.df
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        nparray = df.groupby([pd.Grouper(key='Date', freq='H')]).sum().values
        
        return nparray
    
    '''
    Get the index of train and test set.
    '''
    def get_npindex(self, rate=0.7, timestep=24):
        df = self.get_nparray()
        np_idx = np.arange(len(df))
        
        train_size = int(len(np_idx)/timestep * rate) * timestep
        
        train_idx, test_idx = np_idx[0:(train_size)], np_idx[train_size:(len(np_idx)-timestep)]
        
        return  train_idx, test_idx
    
    '''
    Get holiday set.
    e.g. MON, TUE, WED, THU, FRI=0 ; 
         SUN, SAT=1 
    '''
    def get_npdayofweek(self):
        df = self.df
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.groupby([pd.Grouper(key='Date', freq='H')]).sum()
        
        df['Date'] = df.index
        df['dayofweek'] = pd.to_datetime(df['Date'], errors='coerce').dt.dayofweek
        
        # e.g. The day of the week with Monday=0, Sunday=6; Therefore, it's 1 when day is SUN, SAT.
        df['dayofweek'] = df.dayofweek>5
        # bool to int, True->1, false=0
        df['dayofweek'] = df['dayofweek'].astype(int)
        npdayofweek = df['dayofweek'].values
        
        train_idx, test_idx = self.get_npindex()
        train_day, test_day = np.zeros(np.shape(train_idx)), np.zeros(np.shape(test_idx))
        
        for i in range(len(train_idx)):
            train_day[i] = npdayofweek[train_idx[i]]

        for i in range(len(test_idx)):
            test_day[i] = npdayofweek[test_idx[i]]

        return train_day, test_day

    '''
    Get date date 
    remain information about date which train & test numpy is.
    *Important: it's relation with 'look_back' on funtion process_timeseriesXY.
    '''
    def get_record_date(self, rate=0.7, timestep=24):
        df = self.df
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        datetimeIndex = df.groupby([pd.Grouper(key='Date', freq='H')]).sum()
        
        train_idx, test_idx = self.get_npindex()
        
        '''
        *Important :if look_back is 24
        '''
        train_idx_Y_head, train_idx_Y_tail= train_idx[24], train_idx[-1]
        test_idx_Y_head, test_idx_Y_tail = test_idx[24], test_idx[-1]
        
        # Because it is index, it need to add 1
        date_train_Y = datetimeIndex[train_idx_Y_head: train_idx_Y_tail+1].index
        date_test_Y = datetimeIndex[test_idx_Y_head: test_idx_Y_tail+1].index
        
        return date_train_Y, date_test_Y
    
    '''
    Add validation set, 
    Interval 24
    it is used when the mode predict 24 step(one day)
    '''    
    def get_taxi_data_24_v(self, train_r=0.6, validate_r=0.2, test_r=0.2, timestep=24):
        dt_np = self.get_nparray()
        # a mutiple of 'timestep'
        train_size = int(len(dt_np)/timestep * train_r) * timestep
        validate_size = int(len(dt_np)/timestep * (train_r + validate_r)) * timestep
        
        training, validation, testing = dt_np[0:(train_size), :], dt_np[train_size:validate_size, :], dt_np[validate_size:(len(dt_np)-timestep), :]
        
        training_X, training_Y = self.process_timeseriesXY(training, timestep)
        validation_X, validation_Y = self.process_timeseriesXY(validation, timestep)
        testing_X, testing_Y = self.process_timeseriesXY(testing, timestep)
    
        return training_X, training_Y, validation_X, validation_Y, testing_X, testing_Y       
    
    '''
    Interval 24
    it is used when the mode predict 24 step(one day)
    '''    
    def get_taxi_data_24(self, rate=0.7, timestep=24):
        dt_np = self.get_nparray()
        # a mutiple of 'timestep'
        train_size = int(len(dt_np)/timestep * rate) * timestep
        
        training, testing = dt_np[0:(train_size), :], dt_np[train_size:(len(dt_np)-timestep), :]
        
        training_X, training_Y = self.process_timeseriesXY(training, timestep)
        testing_X, testing_Y = self.process_timeseriesXY(testing, timestep)
    
        return training_X, training_Y, testing_X, testing_Y       
    
    '''
    Interval 1
    it is used when the mode predict 1 step
    '''
    def get_taxi_data(self, rate=0.7, timestep=24):
        dt_np = self.get_nparray()
        # a mutiple of 'timestep'
        train_size = int(len(dt_np)/timestep * rate) * timestep
        # [0:index+1] and foward one time step ; Because last index(t+1) is null, we put forward one timestep.  
        training, testing = dt_np[0:(train_size+1), :], dt_np[train_size:(len(dt_np)-timestep+1),:]
        
        training_X, training_Y = self.process_timeseriesXY(training)
        testing_X, testing_Y = self.process_timeseriesXY(testing)
    
        return training_X, training_Y, testing_X, testing_Y
    
    '''
    # function procss_XY
    # e.g. time series df[1,2,3,4,5] -> df_X[1,2,3,4], df_Y[2,3,4,5] 
    '''    
    def process_timeseriesXY(self, df, look_back=1):
        df_X = df[0:(len(df)-look_back), :]
        df_Y = df[look_back:len(df), :]
        
        return df_X, df_Y

    '''
    # function split_2layers
    # e.g. [10,2,3,4,5]>3 -> [10,0,0,4,5], df_Y[0,2,3,0,0] 
    ''' 
    def split_2layers(self, layer, benchmark=50):
        condlist1 = [layer>benchmark, layer<=benchmark]
        choicelist1 = [layer, 0]
        layer1 = np.select(condlist1, choicelist1)
        
        condlist2 = [layer>benchmark, layer<=benchmark]
        choicelist2 = [0, layer]
        layer2 = np.select(condlist2, choicelist2)
        
        return layer1, layer2
    
    '''
    # function prediction_to_csv
    # e.g. numpy to df to .csv file
    '''
    def prediction_to_csv(self, pred, name='test'):
        date_train_Y, date_test_Y = self.get_record_date()
        
        df = pd.DataFrame(data=pred, index=date_test_Y)
        df.to_csv(name + '.csv')