import torch

def dice_coef(input_, target):
    smooth = 1e-6
    iflat = input_[:,1,:,:]
    tflat = target[:,1,:,:]
    intersection = (iflat * tflat).sum(dim=(2,1))
    return torch.mean((2. * intersection + smooth) / (iflat.sum(dim=(2,1)) + tflat.sum(dim=(2,1)) + smooth))

def dice_loss(input_, target):
    return 1-dice_coef(input_, target)

def dice_coef_hard(input_, target):
    iflat = torch.argmax(input_, dim=1).type(torch.cuda.FloatTensor)
    tflat = target[:,1,:,:]
    intersection = (iflat * tflat).sum(dim=(2,1))
    return torch.mean((2. * intersection) / (iflat.sum(dim=(2,1)) + tflat.sum(dim=(2,1))))