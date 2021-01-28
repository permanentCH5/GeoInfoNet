import sys
sys.path.append('../dataset')
sys.path.append('../datatransfer')
sys.path.append('../network')
sys.path.append('../test')
sys.path.append('../util')
import torch
import torch.nn as nn
import cv2 as cv
import gin as gin
import os
from misc import check_rmfile,check_mkdir
import glob
import numpy as np
import gdal
import math
from torch.utils.data import DataLoader
import torch.nn.functional as F
np.set_printoptions(edgeitems=8)


def getImageName(imagePath):
    slash = imagePath.rfind('/')
    dot = imagePath.rfind('.')
    imageName = imagePath[slash+1:dot]
    return imageName



def toTensor(pic):
    if isinstance(pic, np.ndarray):
    # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return img.float()



args = {
    'model':'../models/gin.pth',
    'outputdirRoot': '../output/',
    'testimagetxt': './test_levir_cloud_snow_dataset_version3_withdem.txt',
}


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda:0')

def test():
    txtFile = open(args['testimagetxt'],'r')
    imgPath=[]
    demPath = []
    while 1:
        line = txtFile.readline()
        if not line:
            break
        line=line.strip('\n')
        (imagePath_s,demPath_s) = line.split()
        imgPath.append(imagePath_s)
        demPath.append(demPath_s)


    print(imgPath) 


    net = gin.gin()
    net.to(device)

    net.load_state_dict(torch.load(args['model']))
    net.eval()

    slash = args['model'].rfind('/')
    dot = args['model'].rfind('.')
    expName = args['model'][slash+1:dot]
    outputdir = args['outputdirRoot']+expName+'/'

    check_mkdir(outputdir)

    for singleImagePath in imgPath:
        imageName = getImageName(singleImagePath)
        imageIndex = imgPath.index(singleImagePath)
        print(imageName) 
        imgDataset = gdal.Open(singleImagePath)
        imgGeotransform = imgDataset.GetGeoTransform()
        tmp = imgDataset.GetRasterBand(1).ReadAsArray()
        height = tmp.shape[0]
        width = tmp.shape[1]
        datacube = np.zeros( (height,width,7) ).astype('float32') 
        
        for i in range(0,4):
            datacube[:,:,i] = imgDataset.GetRasterBand(i+1).ReadAsArray()
            datacube[:,:,i] = (datacube[:,:,i]>1023)*1023 + (datacube[:,:,i]<=1023)*datacube[:,:,i]
            datacube[:,:,i] = datacube[:,:,i]/1023.0
        imgDataset=None
        demDataset = gdal.Open(demPath[imageIndex])
        datacube[:,:,4] = ( (demDataset.GetRasterBand(1).ReadAsArray()).astype('float32') ) /10000.0
        demDataset = None
        datacube[:,:,5]= (imgGeotransform[0]+imgGeotransform[1]*np.tile(np.arange(width),(height,1)) + imgGeotransform[2]*(  (np.ones( (width,height) )*np.arange(height)).transpose()     )  +180.0)/360.0
        datacube[:,:,6]= (imgGeotransform[3]+imgGeotransform[4]*np.tile(np.arange(width),(height,1)) + imgGeotransform[5]*(  (np.ones( (width,height) )*np.arange(height)).transpose()     )  +90.0)/180.0
        cv.imwrite(outputdir+imageName+'_ori.png',(datacube[:,:,0:3]*255.0).astype('uint8') )
        imageCover = np.zeros((height,width,3)).astype('uint8')
        for i in range(0,3):
            imageCover[:,:,i] = (0.7*255*datacube[:,:,i].copy()).astype('uint8')
        input_height=600
        input_width=600
        deltaHeight=100#half unit of filed of view
        deltaWidth=100#half unit of filed of view
        actual_height = input_height - 2*deltaHeight
        actual_width = input_width - 2*deltaWidth
        tmp_height= int(math.ceil(float(height)/actual_height)*actual_height)
        tmp_width= int(math.ceil(float(width)/actual_width)*actual_width)

        imageProbMax_t = np.zeros((tmp_height,tmp_width)).astype('uint8')
        imageCloudProb_t = np.zeros((tmp_height,tmp_width)).astype('uint8')
        imageSnowProb_t = np.zeros((tmp_height,tmp_width)).astype('uint8')

        imageProbMax = np.zeros((height,width)).astype('uint8')
        imageCloudProb = np.zeros((height,width)).astype('uint8')
        imageSnowProb = np.zeros((height,width)).astype('uint8')

        imageCloudMask = np.zeros((height,width)).astype('uint8')
        imageSnowMask = np.zeros((height,width)).astype('uint8')
        imageRoiMap = np.zeros((height,width)).astype('uint8')

        paddatacube = np.zeros((tmp_height+2*deltaHeight,tmp_width+2*deltaWidth,7))
        paddatacube[deltaHeight:deltaHeight+height,deltaWidth:deltaWidth+width,:] = datacube.copy()

        for i in range(0,tmp_height,actual_height):
            for j in range(0,tmp_width,actual_width):
                img_block = paddatacube[i:i+input_height,j:j+input_width,:].copy()       
                img_block = toTensor(img_block)
                testset=[]
                testset.append(img_block)
                test_loader = DataLoader(testset, batch_size=1, num_workers=1, shuffle=False)

                for p,img_block in enumerate(test_loader):

                    with torch.no_grad():
                        img_block = img_block.to(device)
                        output_block = net(img_block)
                        output_block = output_block.float()

                        compensation_len = 0
                        output_block = F.interpolate(output_block,
                                    size=(input_height+compensation_len,input_width+compensation_len),
                                    mode='bilinear',align_corners=True)

                        prob = F.softmax(output_block,dim=1)
                        selectHeight = int(deltaHeight+compensation_len/2)
                        selectWidth = int(deltaWidth+compensation_len/2)

                        imageProbMax_t[i:i+actual_height,j:j+actual_width] = (prob.data[0,:,selectHeight:selectHeight+actual_height,selectWidth:selectWidth+actual_width].cpu().numpy().argmax(axis=0))
                        imageCloudProb_t[i:i+actual_height,j:j+actual_width] = (prob.data[0,1,selectHeight:selectHeight+actual_height,selectWidth:selectWidth+actual_width].cpu().numpy()*255)
                        imageSnowProb_t[i:i+actual_height,j:j+actual_width] = (prob.data[0,2,selectHeight:selectHeight+actual_height,selectWidth:selectWidth+actual_width].cpu().numpy()*255)

        imageProbMax = imageProbMax_t[0:height,0:width]
        imageCloudProb = imageCloudProb_t[0:height,0:width]
        imageSnowProb = imageSnowProb_t[0:height,0:width]

        imageRoiMap = 128*(imageProbMax==1).copy()+255*(imageProbMax==2).copy()

        imageCloudMask = 255*(imageProbMax==1).copy()
        imageSnowMask = 255*(imageProbMax==2).copy()
        
        for i in range(0,2):
            tmp = imageCover[:,:,i] + 0.3*imageCloudMask
            imageCover[:,:,i] = (tmp>255)*255 + (tmp<=255)*tmp
            imageCover[:,:,i] = imageCover[:,:,i].astype('uint8')

        for i in range(1,3):
            tmp = imageCover[:,:,i] + 0.3*imageSnowMask
            imageCover[:,:,i] = (tmp>255)*255 + (tmp<=255)*tmp
            imageCover[:,:,i] = imageCover[:,:,i].astype('uint8')



        cv.imwrite(outputdir+imageName+'_cover.png',imageCover)
        cv.imwrite(outputdir+imageName+'_cloud_prob.png',imageCloudProb)
        cv.imwrite(outputdir+imageName+'_cloud_mask.png',imageCloudMask)
        cv.imwrite(outputdir+imageName+'_snow_prob.png',imageSnowProb)
        cv.imwrite(outputdir+imageName+'_snow_mask.png',imageSnowMask)
        cv.imwrite(outputdir+imageName+'_allMask.png',imageRoiMap)




def main(argv):
    test()

if __name__ == '__main__':
    main(sys.argv)
