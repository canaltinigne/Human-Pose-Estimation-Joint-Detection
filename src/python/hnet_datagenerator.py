import numpy as np 
import os
import cv2
from keras.utils import to_categorical
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

image_adr = 'SMALL_5K_IMAGES/'
mask_adr = 'SMALL_5K_MASKS/'

images = sorted(os.listdir(image_adr))
masked = sorted(os.listdir(mask_adr))

class Images(Dataset):

    def __init__(self, X_names, y_names, indices):
        self.X_names = X_names
        self.y_names = y_names
        self.indices = indices
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        
        X_name = image_adr + self.X_names[self.indices[idx]]
        y_name = mask_adr + self.y_names[self.indices[idx]]
        
        X = cv2.cvtColor(cv2.imread(X_name), cv2.COLOR_BGR2RGB).astype('float32')
        X /= 255
        
        r = X.shape[0]
        c = X.shape[1]
        r_pad = 0
        c_pad = 0
        
        while r % 16 != 0:
            r_pad += 1
            r += 1
            
        while c % 16 != 0:
            c_pad += 1
            c += 1
            
        X = np.pad(X, ((0, r_pad), (0, c_pad), (0, 0)), mode='constant', constant_values=0)
        X = self.to_tensor(X)
        
        y = (cv2.imread(y_name, 0) > 200).astype('float32')
        y = np.pad(y, ((0, r_pad), (0, c_pad)), mode='constant', constant_values=0)
        y = to_categorical(y, 2)
        y = self.to_tensor(y)

        return {'X': X, 'y': y}