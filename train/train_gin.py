import sys
sys.path.append('../dataset')
sys.path.append('../datatransfer')
sys.path.append('../network')
sys.path.append('../test')
sys.path.append('../util')

import dataset_si as LevirCS
import torch
import torch.nn as nn
import torch.optim as optim
import dataTransfer as inputTransform
import dataTransferJoint as jointTransform
import cv2 as cv
import gin as gin
#import myRes50 as res
import math
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datetime
from misc import check_mkdir,  AverageMeter, BatchStat
import numpy as np

#need to set
args = {
    'train_batch_size':4,
    'lr': 0.001,
    'lr_decay':0.9,
    'crop_size':240,#wait for design
    'weight_decay':1e-4,
    'momentum':0.9,
    'display':10,
    'max_epoch': 250,
    'max_iter':200000,
    'snapshot': 'gin',
    'snapshot_freq': 20000,
    'snapshotInitName': 'densenet169.pth',
    'model_dir': '../models/',
    'log_dir': '../log/',
    'exp_name': 'gin',
    'print_freq': 1,
    'is_transfer': True,
    'curr_epoch': 1,
    'trainTxtPath':'./train_levir_cloud_snow_dataset_version3_withdem.txt',
}
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda:0')

def main():

    train_joint_transform = jointTransform.Compose([
        jointTransform.RandomImgCrop(args['crop_size']), #imgcrop:h,w
        jointTransform.RotateTransform(), #random 90
    ])

    train_set = LevirCS.LevirCS_MSS_CloudSnowDatasetV3(trainTxtPath=args['trainTxtPath'],rootPath='',
        joint_transform=train_joint_transform,image_transform=None)

    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=2, shuffle=True)
    curr_epoch = args['curr_epoch']

    net = gin.gin(is_transfer=args['is_transfer'],curr_epoch=args['curr_epoch'],snapshotInitPath=args['model_dir']+args['snapshotInitName']).to(device)
    print(net)
    net.train()

    criterion_cls = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(),lr=args['lr'],weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)
    print( optimizer)

    check_mkdir(args['model_dir'])
    check_mkdir(args['log_dir'])

    args['time'] = str(datetime.datetime.now().timestamp())
    open(os.path.join(args['log_dir'],  args['time']+ '_'+args['snapshot']+'.txt'), 'a').write(str(args) + '\n\n')

    train(train_loader, net,criterion_cls, optimizer, curr_epoch, args)

def train(train_loader, net,criterion_cls, optimizer, curr_epoch, train_args):

    while curr_epoch<=train_args['max_epoch']:
        main_loss_recoder = AverageMeter()
        cls_loss_recoder = AverageMeter()        
        batch_stat = BatchStat()

        curr_iter = (curr_epoch - 1) * len(train_loader)
        for i, data in enumerate(train_loader):

            optimizer.param_groups[0]['lr'] = train_args['lr'] * (1 - float(curr_iter) / train_args['max_iter']
                                                                      ) ** train_args['lr_decay']

            inputs, gts_seg = data

            inputs = inputs.to(device)
            optimizer.zero_grad()
            
            outputs_cls= net(inputs)
            gts_seg = gts_seg.to(device)
            cls_loss = criterion_cls(outputs_cls,gts_seg)
            loss = cls_loss
            loss.backward()
            optimizer.step()
            main_loss_recoder.update(loss.data.cpu().numpy(), inputs.size(2) * inputs.size(3))
            batch_stat.stat_update(outputs_cls,gts_seg)
            curr_iter += 1

            if (i + 1) % train_args['display'] == 0:
                
                mainLossOutput = '[epoch %d], [iter %d / %d], [train main loss %.5f],  [lr %.10f]' % (
                    curr_epoch, i + 1, len(train_loader), main_loss_recoder.avg,
                    optimizer.param_groups[0]['lr'])
                print(mainLossOutput)

                open(os.path.join(args['log_dir'],  args['time']+ '_'+args['snapshot']+'.txt'), 'a').write(mainLossOutput)

                batch_statOutput = '[iter %d] [recall %.5f,%.5f,%.5f], [precision %.5f,%.5f,%.5f], [iou %.5f,%.5f,%.5f]\n' % (i+1,
                    batch_stat.recall[0],batch_stat.recall[1],batch_stat.recall[2],batch_stat.precision[0],
                    batch_stat.precision[1],batch_stat.precision[2],batch_stat.iou[0],batch_stat.iou[1],batch_stat.iou[2])
                print(batch_statOutput)
                open(os.path.join(args['log_dir'],  args['time']+ '_'+args['snapshot']+'.txt'), 'a').write(batch_statOutput)

            if curr_iter >= train_args['max_iter']:
                return

            if curr_iter % train_args['snapshot_freq'] == 0 and curr_iter/train_args['snapshot_freq'] >0:
                torch.save(net.state_dict(),train_args['model_dir']+train_args['snapshot']+'_'+str(curr_iter)+'_epoch'+str(curr_epoch)+'.pth')

        curr_epoch += 1

    return


if __name__ == '__main__':
    main()
