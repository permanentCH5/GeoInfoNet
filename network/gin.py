import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class GIN(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    # def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
    #              num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
    def __init__(self):
        # growth_rate=32, block_config=(6, 12, 24, 16),
        # num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000
# num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32)
        super(GIN, self).__init__()

        # First convolution

        self.conv0 = nn.Conv2d(4, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.norm0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv0_feat = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Each denseblock
        self.denseblock1 = _DenseBlock(num_layers=6, num_input_features=64,
                                bn_size=4, growth_rate=32, drop_rate=0)
        self.dense1_feat = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.transition1 = _Transition(num_input_features=256, num_output_features=128)
        self.denseblock2 = _DenseBlock(num_layers=12, num_input_features=128,
                                bn_size=4, growth_rate=32, drop_rate=0)
        self.dense2_feat = nn.Conv2d(512, 64, kernel_size=3, padding=1)
        self.transition2 = _Transition(num_input_features=512, num_output_features=256)
        self.denseblock3 = _DenseBlock(num_layers=32, num_input_features=256,
                                bn_size=4, growth_rate=32, drop_rate=0)
        self.dense3_feat = nn.Conv2d(1280, 64, kernel_size=3, padding=1)
        self.transition3 = _Transition(num_input_features=1280, num_output_features=640)
        self.denseblock4 = _DenseBlock(num_layers=32, num_input_features=640,
                                bn_size=4, growth_rate=32, drop_rate=0)
        self.dense4_feat = nn.Conv2d(1664, 64, kernel_size=3, padding=1)

        self.conv0_aux = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.norm0_aux = nn.BatchNorm2d(64)
        self.relu0_aux = nn.ReLU(inplace=True)
        self.conv0_feat_aux = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool0_aux = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Each denseblock
        self.denseblock1_aux = _DenseBlock(num_layers=6, num_input_features=64,
                                bn_size=4, growth_rate=32, drop_rate=0)
        self.dense1_feat_aux = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.transition1_aux = _Transition(num_input_features=256, num_output_features=128)
        self.denseblock2_aux = _DenseBlock(num_layers=12, num_input_features=128,
                                bn_size=4, growth_rate=32, drop_rate=0)
        self.dense2_feat_aux = nn.Conv2d(512, 64, kernel_size=3, padding=1)
        self.transition2_aux = _Transition(num_input_features=512, num_output_features=256)
        self.denseblock3_aux = _DenseBlock(num_layers=32, num_input_features=256,
                                bn_size=4, growth_rate=32, drop_rate=0)
        self.dense3_feat_aux = nn.Conv2d(1280, 64, kernel_size=3, padding=1)
        self.transition3_aux = _Transition(num_input_features=1280, num_output_features=640)
        self.denseblock4_aux = _DenseBlock(num_layers=32, num_input_features=640,
                                bn_size=4, growth_rate=32, drop_rate=0)
        self.dense4_feat_aux = nn.Conv2d(1664, 64, kernel_size=3, padding=1)



        self.score = nn.Conv2d(640, 3, kernel_size=1, padding=0)

        # no transition4
         # self.transition4 = _Transition(num_input_features=1664, num_output_features=832)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):

        #inputTensor: 
        # batch channel height width
        # img channel 0-3
        # dem channel 4
        # geo channel 5-6
        # time channel 7
        imageSize = x.shape[2:]

        img = x[:,:4,:,:]

        aux = x[:,4:,:,:]

        img = self.conv0(img)
        img = self.norm0(img)
        img = self.relu0(img)
        img0 = self.conv0_feat(img)
        img0_upsample = F.interpolate(img0,size=imageSize,mode='bilinear',align_corners=True)

        img = self.pool0(img)
        img = self.denseblock1(img)
        img1 = self.dense1_feat(img)
        img1_upsample = F.interpolate(img1,size=imageSize,mode='bilinear',align_corners=True)
        img = self.transition1(img)

        img = self.denseblock2(img)
        img2 = self.dense2_feat(img)
        img2_upsample = F.interpolate(img2,size=imageSize,mode='bilinear',align_corners=True)
        img = self.transition2(img)

        img = self.denseblock3(img)
        img3 = self.dense3_feat(img)
        img3_upsample = F.interpolate(img3,size=imageSize,mode='bilinear',align_corners=True)
        img = self.transition3(img)

        img = self.denseblock4(img)
        img4 = self.dense4_feat(img)
        img4_upsample = F.interpolate(img4,size=imageSize,mode='bilinear',align_corners=True)

        aux = self.conv0_aux(aux)
        aux = self.norm0_aux(aux)
        aux = self.relu0_aux(aux)
        aux0 = self.conv0_feat_aux(aux)
        aux0_upsample = F.interpolate(aux0,size=imageSize,mode='bilinear',align_corners=True)

        aux = self.pool0_aux(aux)
        aux = self.denseblock1_aux(aux)
        aux1 = self.dense1_feat_aux(aux)
        aux1_upsample = F.interpolate(aux1,size=imageSize,mode='bilinear',align_corners=True)
        aux = self.transition1_aux(aux)

        aux = self.denseblock2_aux(aux)
        aux2 = self.dense2_feat_aux(aux)
        aux2_upsample = F.interpolate(aux2,size=imageSize,mode='bilinear',align_corners=True)
        aux = self.transition2_aux(aux)

        aux = self.denseblock3_aux(aux)
        aux3 = self.dense3_feat_aux(aux)
        aux3_upsample = F.interpolate(aux3,size=imageSize,mode='bilinear',align_corners=True)
        aux = self.transition3_aux(aux)

        aux = self.denseblock4_aux(aux)
        aux4 = self.dense4_feat_aux(aux)
        aux4_upsample = F.interpolate(aux4,size=imageSize,mode='bilinear',align_corners=True)

        x_final = self.score(torch.cat((img0_upsample,img1_upsample,img2_upsample,img3_upsample,img4_upsample,aux0_upsample,aux1_upsample,aux2_upsample,aux3_upsample,aux4_upsample),1))

        self.coreHeight = x_final.shape[2]
        self.coreWidth = x_final.shape[3]       

        return x_final


def gin(is_transfer=False,curr_epoch=2,snapshotInitPath='/home/c307/pytorch_learning/multitask_task_learning_cloud/model/vgg16_bn.pth', **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GIN(**kwargs)
    if is_transfer == False:
        # print('Training from start ......')
        curr_epoch = 1
    elif curr_epoch==1:
        print('Training from resumed pths ......')
        net_dict = model.state_dict()
        pretrained_dict = torch.load(snapshotInitPath)

        conv1Weight = pretrained_dict['features.conv0.weight'].detach().numpy()
        if conv1Weight.shape[1]==4:
            pass
        elif conv1Weight.shape[1]==3:
            newConv1Weight = np.zeros((conv1Weight.shape[0],4,conv1Weight.shape[2],conv1Weight.shape[3]))
            for i in range(0,conv1Weight.shape[0]):
                newConv1Weight[i,0,:,:] = conv1Weight[i,0,:,:]
                newConv1Weight[i,1,:,:] = conv1Weight[i,1,:,:]
                newConv1Weight[i,2,:,:] = conv1Weight[i,2,:,:]
                newConv1Weight[i,3,:,:] = conv1Weight[i,0,:,:]/3.0 + conv1Weight[i,1,:,:]/3.0 + conv1Weight[i,2,:,:]/3.0
            pretrained_dict['features.conv0.weight'] = torch.from_numpy(newConv1Weight)
        else:
            print('Error! Conv0.weight channel is not 3 or 4!\n')

        spStrings = ['norm1','relu1','conv1','norm2','relu2','conv2']
        for i in range(0,conv1Weight.shape[0]):
	        net_dict['conv0_aux.weight'][i,0,:,:] = pretrained_dict['features.conv0.weight'][i,3,:,:]
	        net_dict['conv0_aux.weight'][i,1,:,:] = pretrained_dict['features.conv0.weight'][i,3,:,:]
	        net_dict['conv0_aux.weight'][i,2,:,:] = pretrained_dict['features.conv0.weight'][i,3,:,:]

        for target_net_key in net_dict.keys():
            # print target_net_key
            if target_net_key == 'conv0_aux.weight':
            	continue

            target_net_key_temp = target_net_key
            if target_net_key.find('_aux')!=-1:
            	start = target_net_key.find('_aux')
            	target_net_key_temp = target_net_key[:start]+target_net_key[start+len('_aux'):]
            
            target_net_key_pretrained = 'features.'+ target_net_key_temp
            for spString in spStrings:
                if target_net_key_temp.find(spString)!=-1:
                    dot = target_net_key_temp.rfind('.')
                    target_net_key_pretrained = 'features.'+target_net_key_temp[:dot-1]+ '.'+target_net_key_temp[dot-1:dot]+'.' +target_net_key_temp[dot+1:]                    
                    break     

            if target_net_key_pretrained in pretrained_dict.keys():
                net_dict[target_net_key] = pretrained_dict[target_net_key_pretrained]

        model.load_state_dict(net_dict)
        
    else:
        net_dict = model.state_dict()
        pretrained_dict = torch.load(snapshotInitPath)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
        net_dict.update(pretrained_dict)
        model.load_state_dict(net_dict)
        


    return model

