import numpy as np
from keras.preprocessing import image
import cv2
from keras.utils import to_categorical
import os

image_adr = 'SMALL_5K_IMAGES/'
mask_adr = 'SMALL_5K_MASKS/'
heatmap_adr = 'SMALL_5K_HEATMAPSOFTMAX/'

images = sorted(os.listdir(image_adr))
masked = sorted(os.listdir(mask_adr))
heatmaps = sorted(os.listdir(heatmap_adr))

def generator(X_data, y_data, order):
    
    samples_per_epoch = len(order)
    number_of_batches = samples_per_epoch
    
    counter=0
    
    while 1:
        
        X_batch = cv2.cvtColor(cv2.imread(image_adr + X_data[order[counter]]), cv2.COLOR_BGR2RGB).astype('float32')
        X_batch /= 255
        
        r = X_batch.shape[0]
        c = X_batch.shape[1]
        r_pad = 0
        c_pad = 0
        
        while r % 16 != 0:
            r_pad += 1
            r += 1
            
        while c % 16 != 0:
            c_pad += 1
            c += 1
            
        X_batch = np.pad(X_batch, ((0, r_pad), (0, c_pad), (0, 0)), mode='constant', constant_values=0)
        X_batch = np.expand_dims(X_batch, axis=0)

        y_heatmap = np.load(heatmap_adr + heatmaps[order[counter]])  # For Heatmaps
        y_heatmap = np.pad(y_heatmap, ((0, r_pad), (0, c_pad)), mode='constant', constant_values=0)
        y_map = np.zeros(y_heatmap.shape + (26,))
        
        for i in range(26):
            y_map[(y_heatmap == i), i] = 1
        
        y_map = np.expand_dims(y_map, 0)
        
        y_mask = (cv2.imread(mask_adr + y_data[order[counter]], 0) > 200).astype('float32') # For Mask
        y_mask = np.pad(y_mask, ((0, r_pad), (0, c_pad)), mode='constant', constant_values=0)
        y_mask = to_categorical(y_mask, 2) # For Mask
        y_mask = np.expand_dims(y_mask, 0)

        counter += 1
        
        yield X_batch, {'output1': y_mask, 'output2': y_map}

        #restart counter to yeild data in the next epoch as well
        if counter >= number_of_batches:
            counter = 0