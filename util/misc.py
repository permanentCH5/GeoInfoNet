import os
import numpy as np


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def check_rmfile(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)


class BatchStat(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.recall = 0.0
        self.precision = 0.0
        self.iou = 0.0

    def stat_update(self,inputs,targets):
        #for only two class
        confusionInt = np.zeros((3,3))
        recall = np.zeros((1,3))
        precision = np.zeros((1,3))
        iou = np.zeros((1,3))
        resultImg = inputs.data.max(1)[1].squeeze_(0).cpu().numpy()
        labelImg = targets.data.cpu().numpy()
        batch_size = resultImg.shape[0]
        for i in range(0,3):
            for j in range(0,3):
                for k in range(0,batch_size):
                    labelHit = labelImg[k,:,:]==i
                    resultHit = resultImg[k,:,:]==j
                    confusionInt[i,j] = confusionInt[i,j] + sum(sum(labelHit&resultHit))
        for i in range(0,3):
            recall[0,i] = confusionInt[i,i] / (confusionInt[0,i] + confusionInt[1,i]+confusionInt[2,i]+0.001)
            precision[0,i] = confusionInt[i,i] / (confusionInt[i,0] + confusionInt[i,1]+confusionInt[i,2]+0.001)
            iou[0,i] = confusionInt[i,i] / (confusionInt[0,i]  + confusionInt[i,0] +  confusionInt[i,1] + confusionInt[1,i]+confusionInt[i,2] + confusionInt[2,i] -confusionInt[i,i] +0.001)

        self.recall = (recall[0,0],recall[0,1],recall[0,2])
        self.precision = (precision[0,0],precision[0,1],precision[0,2])
        self.iou = (iou[0,0],iou[0,1],iou[0,2])

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

