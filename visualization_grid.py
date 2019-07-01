# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 20:17:32 2019

@author: DART_HSU
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def visualization2_Grid(img1_, img2_, size=(60, 60)):
    # scale 
    img_ = np.concatenate((img1_, img2_))
    vmin_ = np.amin(img_)
    vmax_ = np.amax(img_)
    
    imgs1 = np.reshape(img1_, (-1, size[0], size[1]))
    imgs2 = np.reshape(img2_, (-1, size[0], size[1]))
    
    fig = plt.figure(1, (24,8))

    visualization2_Grid_component(fig, imgs1, 210, vmin_, vmax_)
    visualization2_Grid_component(fig, imgs2, 211, vmin_, vmax_)   
    
    plt.suptitle('Taxi Demand', fontsize=20)
    
    fig.savefig('result_55688_workday.pdf')
    plt.draw()
    plt.show()
    
def visualization2_Grid_component(fig, imgs, plotid, vmin_, vmax_):
    grid = ImageGrid(fig, plotid,  # similar to subplot(111)
                     nrows_ncols=(2, 12),  # creates 2x2 grid of axes 
                     axes_pad=0,  # pad between axes in inch.
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="7%",
                     cbar_pad="2%"
                     )
    
    for i in range(24):  
        '''
        NYC: 12,10
        55688: 0.6 0.5
        '''
        grid[i].text(0.6, 0.5 , str(i)+'-'+str(i+1),
            horizontalalignment='center',
            bbox=dict(facecolor='white', alpha=0.6),
            fontsize=10)
        '''
        # rotate direction, it make image like direction of real image.
        '''
        a = imgs[i]
        a = np.rot90(a)
        a = np.flipud(a)
        
        im = grid[i].matshow(a, cmap = plt.get_cmap('Reds'), vmin=vmin_, vmax=vmax_)
        
        grid[i].tick_params(    
                axis='both',       # changes apply to the x-axis, y-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                right=False,
                left=False,
                top=False,         # ticks along the top edge are off
                labelleft=False,
                labelbottom=False, # labels along the bottom edge are off
                labeltop=False)
        grid[i].grid(True)
        
    plt.colorbar(im, cax = grid.cbar_axes[0])


if __name__=='__main__':
    real = pd.read_csv('./data/55688_real.csv')
    pred = pd.read_csv('./pred_csv/vs/AR_LSTMs.csv')
    
    real['Date'] = pd.to_datetime(real['Date'], errors='coerce')
    pred['Date'] = pd.to_datetime(pred['Date'], errors='coerce')

    # select visualization period
    real = real[real['Date']>=pd.to_datetime('2017/1/24  00:00:00')]
    real = real[real['Date']<=pd.to_datetime('2017/1/24  23:00:00')]
    
    pred = pred[pred['Date']>=pd.to_datetime('2017/1/24  00:00:00')]
    pred = pred[pred['Date']<=pd.to_datetime('2017/1/24  23:00:00')]
    ###
    
    real = real.drop('Date', axis=1)
    pred = pred.drop('Date', axis=1)
   
    visualization2_Grid(real.values, pred.values, size=(5, 5))
    