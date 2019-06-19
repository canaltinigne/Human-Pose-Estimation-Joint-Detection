import numpy as np 
import os
import cv2
import sys
from keras.utils import to_categorical
import time
import torch
import torch.nn as nn
from torch_localize import localized_module
from torch_dimcheck import dimchecked
from harmonic.d2 import HConv2d, ScalarGate2d, avg_pool2d, BatchNorm2d, upsample_2d, cat2d, Dropout2d, HConv2dTranspose
from harmonic.cmplx import from_real
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm_notebook as tqdm
import torch.nn.functional as F

@localized_module
class FinalConvBlock(nn.Module):
    def __init__(self, in_repr, out_repr, size=3, pad=False):
        super(FinalConvBlock, self).__init__()

        self.in_repr = in_repr
        self.out_repr = out_repr
        
        self.conv1 = HConv2d(in_repr, out_repr, size, conv_kwargs={'padding':(size-1)//2})

    def forward(self, x):
        y = x
        
        y = self.conv1(y)
        y = y[0, ...]
        y = torch.nn.Softmax(1)(y)

        return y

@localized_module
class HNetConvBlock(nn.Module):
    def __init__(self, in_repr, out_repr, size=3, pad=False):
        super(HNetConvBlock, self).__init__()

        self.in_repr = in_repr
        self.out_repr = out_repr
        
        self.nonl1 = ScalarGate2d(out_repr)
        self.nonl2 = ScalarGate2d(out_repr)
        
        self.bnorm1 = BatchNorm2d(out_repr)
        self.bnorm2 = BatchNorm2d(out_repr)

        self.conv1 = HConv2d(in_repr, out_repr, size, conv_kwargs={'padding':(size-1)//2})
        self.conv2 = HConv2d(out_repr, out_repr, size, conv_kwargs={'padding':(size-1)//2})
        self.dropout = Dropout2d(p=0.1)

    def forward(self, x):
        y = x
        
        y = self.conv1(y)
        y = self.bnorm1(y)
        y = self.nonl1(y)
        y = self.conv2(y)
        y = self.bnorm2(y)
        y = self.nonl2(y)
        y = self.dropout(y)

        return y
    
@localized_module
class HNetUpSampleBlock(nn.Module):
    def __init__(self, in_repr, out_repr, size=3):
        super(HNetUpSampleBlock, self).__init__()

        self.in_repr = in_repr
        self.out_repr = out_repr
        self.nonl = ScalarGate2d(out_repr)
        self.bnorm1 = BatchNorm2d(out_repr)
        self.conv = HConv2d(in_repr, out_repr, size, conv_kwargs={'padding':(size-1)//2})
        #self.conv = HConv2dTranspose(in_repr, out_repr, 2, conv_kwargs={'stride':2})


    def forward(self, x):
        y = x
        y = upsample_2d(y)
        y = self.conv(y)
        y = self.bnorm1(y)
        y = self.nonl(y)

        return y

layers = [(3,),
         (32,32),
         (64,64),
         (128,128),
         (256,256),
         (512,512)]

class HNet(nn.Module):

    def __init__(self, kernel):
        super(HNet, self).__init__()
        
        self.kernel_size = kernel
        
        self.conv_down1 = HNetConvBlock(layers[0], layers[1])
        self.conv_down2 = HNetConvBlock(layers[1], layers[2])
        self.conv_down3 = HNetConvBlock(layers[2], layers[3])
        self.conv_down4 = HNetConvBlock(layers[3], layers[4])
        self.conv_down5 = HNetConvBlock(layers[4], layers[5])
        
        self.conv_upsample1 = HNetUpSampleBlock(layers[5], layers[4], self.kernel_size)
        self.conv_up1 = HNetConvBlock(layers[5], layers[4])
        
        self.conv_upsample2 = HNetUpSampleBlock(layers[4], layers[3], self.kernel_size)
        self.conv_up2 = HNetConvBlock(layers[4], layers[3])
        
        self.conv_upsample3 = HNetUpSampleBlock(layers[3], layers[2], self.kernel_size)
        self.conv_up3 = HNetConvBlock(layers[3], layers[2])
        
        self.conv_upsample4 = HNetUpSampleBlock(layers[2], layers[1], self.kernel_size)
        self.conv_up4 = HNetConvBlock(layers[2], layers[1])

        self.conv_up5 = FinalConvBlock(layers[1], (2,))
        

    @dimchecked
    def forward(self, x: ['n', 3, 'wi', 'hi']) -> ['n', -1, 'wo', 'ho']:
        x = from_real(x)
        y = x
        
        conv1 = self.conv_down1(y)
        pool1 = avg_pool2d(conv1, kernel_size=2)

        conv2 = self.conv_down2(pool1)
        pool2 = avg_pool2d(conv2, kernel_size=2)
        
        conv3 = self.conv_down3(pool2)
        pool3 = avg_pool2d(conv3, kernel_size=2)
        
        conv4 = self.conv_down4(pool3)
        pool4 = avg_pool2d(conv4, kernel_size=2)
        
        conv5 = self.conv_down5(pool4)

        up6 = self.conv_upsample1(conv5)
        merge6 = cat2d(conv4, layers[4], up6, layers[4])
        conv6 = self.conv_up1(merge6)
        
        up7 = self.conv_upsample2(conv6)
        merge7 = cat2d(conv3, layers[3], up7, layers[3])
        conv7 = self.conv_up2(merge7)

        up8 = self.conv_upsample3(conv7)
        merge8 = cat2d(conv2, layers[2], up8, layers[2])
        conv8 = self.conv_up3(merge8)
        
        up9 = self.conv_upsample4(conv8)
        merge9 = cat2d(conv1, layers[1], up9, layers[1])
        conv9 = self.conv_up4(merge9)
        
        output = self.conv_up5(conv9)
        
        return output