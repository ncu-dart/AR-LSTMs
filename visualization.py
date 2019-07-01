# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 13:45:38 2019

@author: DART_HSU
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from process_data import ProcessData as pdata

class Graph():
    
    def __init__(self, num=0):
        self.figure = plt.figure(num, figsize=(9, 6))
        
    def line_graph_forRMSE_MAPE(self, timestep_x, value_y, ylabel='', label=''):
        plt.figure(self.figure.number)
        
        if ylabel is 'MAPE':
            plt.ylim(0, 0.6)
        if ylabel is 'RMSE':
            plt.ylim(0, 30)
            
        plt.ylabel(ylabel)
        plt.xlabel('Hour')
        plt.plot(timestep_x, value_y, label=label)
        plt.legend(loc=1)
        
    def to_fig(self, name):
        self.figure.savefig('./asset/' + name + '.pdf')

def MAPE(y, y_hat):
    return np.mean(np.abs((y - y_hat) / (y + 1.0))) 

def RMSE(y, y_hat):
    return np.sqrt(np.mean(np.square(y - y_hat)))

if __name__=='__main__':
    
    g1 = Graph(1)
    g2 = Graph(2)
    timestep = 24
    read_files = ['MultiLSTM(2layers)','MultiLSTM(4layers)', 'DMVST-Net', 'Residual-LSTMs', 'AR_LSTMs',
                  'History_Average', 'Linear_Regression', 'ARIMA', 'XGBoost']
#    read_files = ['History_Average', 'Linear_Regression', 'ARIMA', 'XGBoost', 'AR_LSTMs']
#    read_files = ['MultiLSTM(2layers)','MultiLSTM(4layers)', 'DMVST-Net', 'Residual-LSTMs', 'AR_LSTMs']
    
    # 55688_real.csv or real.csv
    real = pd.read_csv('./data/real.csv')
    real_ARIMA = pd.read_csv('./data/real_ARIMA.csv')
    
    real = real.drop(['Date'], axis=1).values
    real_ARIMA = real_ARIMA.drop(['Date'], axis=1).values
    
    for f in read_files:
        timestep_x = []
        value_y_MAPE = []
        value_y_RMSE = []
        
        f_ = pd.read_csv('./pred_csv/vs/' + f + '.csv')
        f_ = f_.drop(['Date'], axis=1).values
        
        if 'ARIMA' in f:
            real_ = real_ARIMA
        else :
            real_ = real
            
        for i in range(timestep):
            timestep_x.append(i)
            value_y_MAPE.append(MAPE(real_[i::timestep], f_[i::timestep]))
            
        
        for i in range(timestep):
            value_y_RMSE.append(RMSE(real_[i::timestep], f_[i::timestep]))
            
        print('%s\' var: %.6f' % ('MAPE', np.var(value_y_MAPE)))
        print('%s\' var: %.4f' % ('RMSE', np.var(value_y_RMSE)))
        g1.line_graph_forRMSE_MAPE(timestep_x, value_y_MAPE, 'MAPE', f)
        g2.line_graph_forRMSE_MAPE(timestep_x, value_y_RMSE, 'RMSE', f) 
        
        mape = MAPE(real_, f_)
        rmse = RMSE(real_, f_)
        print('%s: %.3f, %.3f'%(f, mape, rmse))
        print('---')
        
    g1.to_fig('MAPE')
    g2.to_fig('RMSE')
    plt.show()

        
    
