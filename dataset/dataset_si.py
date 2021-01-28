import glob
import cv2 as cv
import torch
import numpy as np
#from torchvision import models,transforms
from torch.utils.data import DataLoader
import gdal


#for input image
def toTensor(pic):
    if isinstance(pic, np.ndarray):
    # handle numpy array
        #print pic.shape
        pic = pic.transpose((2, 0, 1))
        #print pic.shape

        img = torch.from_numpy(pic.astype('float32'))
        # backward compatibility
        #for 8 bit images
        #return img.float().div(255.0)
        #for 16 bit images
        # return img.float().div(255.0)
        return img.float()


# def normalize(tensor,mean):
#     for t,m in zip(tensor,mean):
#         t.sub_(m)
#     return tensor

def maskToTensor(img):
    return torch.from_numpy(np.array(img,dtype=np.int32)).long()



class LevirCS_MSS_CloudSnowDatasetV3(object):
    #image: mss 16 bit
    #label: 0,1 8 bit

    def __init__(self,trainTxtPath='/home/c307/pytorch_learning/cloud_snow_detection_trans/train_mss_levirCSV2_cloudsnow.txt',rootPath='',
        joint_transform=None,image_transform=None,mean=[257.0/1024.0,268.0/1024.0,276.0/1024.0,238.0/1024.0]):
        #bands
        """
        image 4 bands  -> 0-1023 ->  0-1
        dem 1 band   -> /10000.0 -> almost 0-1
        geo 2 bands  -> 1st long -180-180 -> 0-1 ; 2nd lat -90 - 90 -> 0-1 
        time 1 band  -> 1.1-12.31  /365.0 or 366.0 ->  0-1
        """

        #only for train
        """
        mean_value: 257 #514
        mean_value: 268 #536
        mean_value: 276 #552
        mean_value: 238 #475
        """
        # self.type = type
        # self.demFlag = self.type/4
        # self.geoFlag = ((self.type%4)>1)*1
        # self.timeFlag = self.type%2 

        txtFile = open(trainTxtPath,'r')
        self.imgPath=[]
        self.demPath=[]
        self.seglabelPath=[]
        
        # self.imgs=[]
        # self.dems=[]
        # self.geos=[]
        # self.times=[]
        # self.datacubes = []
        # self.seglabels = []

        print( "loading data paths......")
        while 1:
            line = txtFile.readline()
            if not line:
                break
            (imgPath_s,demPath_s,seglabelPath_s) = line.split()
            #print imgPath_s
            #print labelPath_s
            self.imgPath.append(rootPath+imgPath_s)
            self.demPath.append(rootPath+demPath_s)
            self.seglabelPath.append(rootPath+seglabelPath_s)
            
        print( "loading data paths finished......")
        # print len(self.img)
        # print len(self.seglabel)
        # print len(self.scenelabel)


        self.joint_transform = joint_transform
        self.image_transform = image_transform
        self.mean = mean

    def __getitem__(self,index):

        # img = self.imgs[index]
        imgPath_s = self.imgPath[index]
        demPath_s = self.demPath[index]
        seglabelPath_s = self.seglabelPath[index]

        imgDataset = gdal.Open(imgPath_s)
        imgGeotransform = imgDataset.GetGeoTransform()
        tmp = imgDataset.GetRasterBand(1).ReadAsArray()

        datacube = np.zeros( (tmp.shape[0],tmp.shape[1],7) ).astype('float32') 

        for i in range(0,4):
            datacube[:,:,i] = imgDataset.GetRasterBand(i+1).ReadAsArray()
            datacube[:,:,i] = (datacube[:,:,i]>1023)*1023 + (datacube[:,:,i]<=1023)*datacube[:,:,i]
            datacube[:,:,i] = datacube[:,:,i]/1023.0
        imgDataset=None

        demDataset = gdal.Open(demPath_s)

        datacube[:,:,4] = ( (demDataset.GetRasterBand(1).ReadAsArray()).astype('float32') ) /10000.0
        demDataset = None

        datacube[:,:,5]= (imgGeotransform[0]+imgGeotransform[1]*np.tile(np.arange(tmp.shape[1]),(tmp.shape[0],1)) + imgGeotransform[2]*(  (np.ones( (tmp.shape[1],tmp.shape[0]) )*np.arange(tmp.shape[0])).transpose()     )  +180.0)/360.0
        datacube[:,:,6]= (imgGeotransform[3]+imgGeotransform[4]*np.tile(np.arange(tmp.shape[1]),(tmp.shape[0],1)) + imgGeotransform[5]*(  (np.ones( (tmp.shape[1],tmp.shape[0]) )*np.arange(tmp.shape[0])).transpose()     )  +90.0)/180.0
        


        seglabel = cv.imread(seglabelPath_s,0)

        if self.joint_transform is not None:
            datacube, seglabel = self.joint_transform(datacube, seglabel)

        #then transform the input image to tensor
        if self.image_transform is not None:
            datacube = self.image_transform(datacube)
        
        datacube = toTensor(datacube)

        seglabel = maskToTensor(seglabel)

        return datacube,seglabel

    def __len__(self):
        return len(self.seglabelPath)
