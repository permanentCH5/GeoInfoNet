#import skimage
import cv2 as cv
import torch
import random

import numpy as np



class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
        select one of the transformations!
        select one of the transformations!
        select one of the transformations!

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        len_trans = len(self.transforms)
        selectedTransformation = self.transforms[int(random.random() * len_trans)]
        img = selectedTransformation(img)
        return img


#single layer
class GammaTransform(object):
    def __init__(self,gammaParam):
        #gammaParam is a list
        self.gammaParam = gammaParam

    def __call__(self,img):
        #input uint8
        """
        gammaListLen = len(self.gammaParam)
        selectedParam = self.gammaParam[int(random.random() * gammaListLen)]
        img = img.astype('float64')/255.0
        img = np.power(img,selectedParam)
        img = np.minimum(img*255,np.ones((img.shape[0],img.shape[1],3))*255)
        img = img.astype('uint8')
        return img
        """

        #input uint16 1024 4-bands      
        gammaListLen = len(self.gammaParam)
        selectedParam = self.gammaParam[int(random.random() * gammaListLen)]
        img = img.astype('float64')/255.0
        img = np.power(img,selectedParam)
        img = np.minimum(img*255,np.ones((img.shape[0],img.shape[1],4))*255)
        img = img.astype('uint8')
        return img
        




class ContrastTransform(object):
    def __init__(self,contrastParam):
        #contrastParam is a list
        self.contrastParam = contrastParam

    def __call__(self,img):
        #input uint8
        """
        contrastListLen = len(self.contrastParam)
        selectedParam = self.contrastParam[int(random.random() * contrastListLen)]
        average = 127 * np.ones((img.shape[0],img.shape[1],3),dtype='float32')
        percentage = selectedParam / 100.0
        img = img.astype('float32')
        if percentage>0:
            img = average + (img - average)/(1 - percentage)
        else:
            img = average + (img - average)*(1 + percentage)
        img = np.maximum(np.minimum(img,np.ones((img.shape[0],img.shape[1],3))*255),np.zeros((img.shape[0],img.shape[1],3)))
        img = img.astype('uint8')
        return img
        """

        #input uint16 - 4 bands
        
        contrastListLen = len(self.contrastParam)
        selectedParam = self.contrastParam[int(random.random() * contrastListLen)]
        average = 127 * np.ones((img.shape[0],img.shape[1],4),dtype='float32')
        percentage = selectedParam / 100.0
        img = img.astype('float32')
        if percentage>0:
            img = average + (img - average)/(1 - percentage)
        else:
            img = average + (img - average)*(1 + percentage)
        img = np.maximum(np.minimum(img,np.ones((img.shape[0],img.shape[1],4))*255),np.zeros((img.shape[0],img.shape[1],4)))
        img = img.astype('uint8')
        return img
        



class SaturationTransform(object):
    def __init__(self,saturationParam):
        #saturationParam is a list
        self.saturationParam = saturationParam

    def __call__(self,img):
        #input uint8
        """
        saturationListLen = len(self.saturationParam)
        selectedParam = self.saturationParam[int(random.random() * saturationListLen)]
        percentage = selectedParam/100.0

        b = img[:,:,0].astype('float32')
        g = img[:,:,1].astype('float32')
        r = img[:,:,2].astype('float32')
        rgbMax = np.maximum(b,np.maximum(g,r))
        rgbMin = np.minimum(b,np.minimum(g,r))
        delta = (rgbMax - rgbMin).astype('float32')/255.0
        value = np.maximum((rgbMax + rgbMin).astype('float32')/255.0,np.ones((img.shape[0],img.shape[1]))*0.000001)
        L = value/2.0
        tmp = (L<0.5).astype('uint8')
        s = delta/value * tmp + delta/(2-value) * (1-tmp)
        if(percentage>=0):
            tmp2 = ((percentage + s)>1).astype('uint8')
            alpha = s*tmp2 + (1-percentage)*(1-tmp2)
            alpha = 1/alpha -1
            r = r+(r-L*255)*alpha
            g = g+(g-L*255)*alpha
            b = b+(b-L*255)*alpha
        else:
            alpha = s
            r = L*255 + (r-L*255)*(1+alpha)
            g = L*255 + (g-L*255)*(1+alpha)
            b = L*255 + (b-L*255)*(1+alpha)

        img = img.astype('float32')
        img[:,:,0] = b
        img[:,:,1] = g
        img[:,:,2] = r
        img = np.maximum(np.minimum(img,np.ones((img.shape[0],img.shape[1],3))*255),np.zeros((img.shape[0],img.shape[1],3)))
        img = img.astype('uint8')
        return img
        """

        #input uint16 4-bands 1024
        
        saturationListLen = len(self.saturationParam)
        selectedParam = self.saturationParam[int(random.random() * saturationListLen)]
        percentage = selectedParam/100.0

        b = img[:,:,0].astype('float32')
        g = img[:,:,1].astype('float32')
        r = img[:,:,2].astype('float32')
        ir = img[:,:,3].astype('float32')
        rgbMax = np.maximum(np.maximum(b,np.maximum(g,r)),ir)
        rgbMin = np.minimum(np.minimum(b,np.minimum(g,r)),ir)
        #rgbMax = np.maximum(b,np.maximum(g,r))
        #rgbMin = np.minimum(b,np.minimum(g,r))



        delta = (rgbMax - rgbMin).astype('float32')/255.0
        value = np.maximum((rgbMax + rgbMin).astype('float32')/255.0,np.ones((img.shape[0],img.shape[1]))*0.000001)
        L = value/2.0
        tmp = (L<0.5).astype('float32')
        s = delta/value * tmp + delta/(2-value) * (1-tmp)
        if(percentage>=0):
            tmp2 = ((percentage + s)>1).astype('uint8')
            alpha = s*tmp2 + (1-percentage)*(1-tmp2)
            alpha = 1/alpha -1
            r = r+(r-L*255)*alpha
            g = g+(g-L*255)*alpha
            b = b+(b-L*255)*alpha
            ir = ir+(ir-L*255)*alpha
        else:
            alpha = s
            r = L*255 + (r-L*255)*(1+alpha)
            g = L*255 + (g-L*255)*(1+alpha)
            b = L*255 + (b-L*255)*(1+alpha)
            ir = L*255 + (ir-L*255)*(1+alpha)

        img = img.astype('float32')
        img[:,:,0] = b
        img[:,:,1] = g
        img[:,:,2] = r
        img[:,:,3] = ir
        img = np.maximum(np.minimum(img,np.ones((img.shape[0],img.shape[1],4))*255),np.zeros((img.shape[0],img.shape[1],4)))
        img = img.astype('uint8')
        return img
        




class DiskBlurTransform(object):
    def __init__(self,radiusParam):
        self.radiusParam = radiusParam
        #only support 1,2,3

    def __call__(self,img):
        #input uint8
###For future, disk blur can be used for any kernel size.

        radiusListLen = len(self.radiusParam)
    #   print gammaListLen

        selectedParam = self.radiusParam[int(random.random() * radiusListLen)]
    #   print selectedParam

        if(selectedParam==1):
            kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
            kernel = kernel.astype('float32')
            kernel = kernel/5.0
        if(selectedParam==2):
            kernel = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]])
            kernel = kernel.astype('float32')
            kernel = kernel/13.0
        if(selectedParam==3):
            kernel = np.array([[0,0,0,1,0,0,0],[0,0,1,1,1,0,0],[0,1,1,1,1,1,0],[1,1,1,1,1,1,1],[0,1,1,1,1,1,0],[0,0,1,1,1,0,0],[0,0,0,1,0,0,0]])
            kernel = kernel.astype('float32')
            kernel = kernel/25.0
        img = cv.filter2D(img,ddepth=-1,kernel=kernel)
        #input uint8
        img = img.astype('uint8')

        #input uint16 1024 4-band
        #img = img.astype('float32')

        
        return img

class ColorBiasTransform(object):
    def __init__(self,colorBiasParam):
        #colorBiasParam is a list
        self.colorBiasParam = colorBiasParam

    def __call__(self,img):
        #input uint8
        
        colorBiasListLen = len(self.colorBiasParam)
        selectedParam = self.colorBiasParam[int(random.random() * colorBiasListLen)]
        img = img.astype('float64')/255.0
        img = img * selectedParam
        img = np.minimum(img*255,np.ones((img.shape[0],img.shape[1],4))*255)
        img = img.astype('uint8')
        return img
        

        #input uint16 1024 4-band
        """
        colorBiasListLen = len(self.colorBiasParam)
        selectedParam = self.colorBiasParam[int(random.random() * colorBiasListLen)]
        img = img.astype('float64')/255.0
        img = img * selectedParam
        img = np.minimum(img*1024,np.ones((img.shape[0],img.shape[1],3))*1024)
        img = img.astype('float32')
        """
        return img






























        