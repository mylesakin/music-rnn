import math
import numpy as np
import pandas as pd

filepath = '/Users/alexonderdonk/Dropbox/Research/Recurrent NN Music/tabs/I Dont Wanna Go Down to the Basement.csv'
n_bars = 8

def one_hot(tabs):
    one_hot = np.zeros((14*6 , tabs.shape[1])) #12 frets plus open and muted
    
    for i in range(tabs.shape[0]):
        
        for j in range(tabs.shape[1]):
            
            #treat mutes like fret 13
            if tabs[i,j] == -1:
                one_hot[14*(i+1)-1, j ] = 1
            else:
                one_hot[14 * i + tabs[i,j] , j] = 1
    
    return one_hot
    
def tabs(one_hot):
    tabs = np.zeros((6, one_hot.shape[1]))
    
    for i in range(one_hot.shape[0]):
        
        for j in range(one_hot.shape[1]):
            
            if one_hot[i,j] == 1:
                
                if i % 14 == 13:
                    
                    tabs[math.floor(i / 14), j] = -1
                    
                else:
                    
                    tabs[math.floor(i / 14), j] = i % 14
                    
    return tabs
    
    
#test

csv = pd.read_csv(filepath, header=None) 
test_bars = csv.values[:, 0:n_bars]

test_one_hot = one_hot(test_bars)

test_tabs = tabs(test_one_hot)