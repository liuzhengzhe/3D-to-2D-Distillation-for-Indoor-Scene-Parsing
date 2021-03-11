import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import model.resnet as models
import glob
import model.resnet18 as models_origin
class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)



class PSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, pretrained=True):
        super(PSPNet, self).__init__()
        assert layers in [18, 50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        self.criterion_reg = nn.MSELoss(reduce=False)
        models.BatchNorm = BatchNorm

        if layers == 18:
            resnet = models_origin.resnet18(pretrained=True)
        elif layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        '''self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 512
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, BatchNorm)
            fea_dim *= 2'''


        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048


        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, BatchNorm)
            fea_dim *= 2

        '''self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )'''


        self.cls3 = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 96*2, kernel_size=1),
            BatchNorm(96*2),
            nn.ReLU(inplace=True)
        )

        self.cls2 = nn.Sequential(
            nn.Conv2d(96*2+96, 96, kernel_size=3, padding=1, bias=False),
            BatchNorm(96),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(96, classes, kernel_size=1)
        )


        self.reg = nn.Sequential(
                nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
                BatchNorm(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 96, kernel_size=1)
        )

        self.reg2 = nn.Sequential(
                nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
                BatchNorm(96),
                nn.ReLU(inplace=True),
                nn.Conv2d(96, 96, kernel_size=1),
                nn.ReLU(inplace=True)
        )
        self.reg3 = nn.Sequential(
                nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
                BatchNorm(96),
                nn.ReLU(inplace=True),
                nn.Conv2d(96, 96, kernel_size=1),
                nn.ReLU(inplace=True)
        )


        





        self.bn=BatchNorm(96)
        self.bn2=BatchNorm(96)

        self.mean=torch.tensor(np.load('../data/meanvar/mean.npy'), requires_grad=False)
        self.var=torch.tensor(np.sqrt(np.load('../data/meanvar/var.npy')), requires_grad=False)


        self.mean_2d=torch.tensor(np.load('../data/meanvar/mean_2d.npy'), requires_grad=False) #.cuda().float()
        self.var_2d=torch.tensor(np.sqrt(np.load('../data/meanvar/var_2d.npy')), requires_grad=False) #.cuda().float()
        self.var=self.var[0,:]

        
        self.mean=torch.unsqueeze(self.mean,0)
        self.mean=torch.unsqueeze(self.mean,2)
        self.mean=torch.unsqueeze(self.mean,2)
        self.mean=self.mean.repeat(1,1,60,60)
        self.var=torch.unsqueeze(self.var,0)
        self.var=torch.unsqueeze(self.var,2)
        self.var=torch.unsqueeze(self.var,2)
        self.var=self.var.repeat(1,1,60,60)
        

    def forward(self, x, y=None, feat=None, feat_idx=None):
        x_size = x.size()

        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)
        inp=x.clone()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        if self.use_ppm:
            x = self.ppm(x)


        #DN_1
        reg = self.reg(x)
        mean=self.mean.repeat(x.shape[0],1,1,1)
        var=self.var.repeat(x.shape[0],1,1,1)
        reg=self.bn(reg)*var.cuda().float()+mean.cuda().float()
        reg_totrain=self.reg2(reg)
        

        #DN_2
        reg=self.bn2(reg_totrain)
        reg=reg*(self.var_2d.cuda().float())+self.mean_2d.cuda().float()   #reg*(0.037**0.5)+0.1515
        reg=self.reg3(reg)

        #concat 2d and 3d feature
        x=self.cls3(x)
        reg_x=torch.cat((x,reg),1)

        #repeat the valid 3d projected indexes
        if self.training:
            
            feat_idx=torch.unsqueeze(feat_idx,1)
            number=torch.sum(feat_idx)
            featidxs=feat_idx.repeat(1,96,1,1)


        
        x2 = self.cls2(reg_x)

        

        if self.zoom_factor != 1:
          x2 = F.interpolate(x2, size=(h, w), mode='bilinear', align_corners=True)
          
        if self.training:
            reg_loss=self.criterion_reg(reg_totrain.cuda()*featidxs.cuda(),feat.cuda()*featidxs.cuda()).sum()
            if number==0:
                reg_loss=reg_loss*0
            else:
                reg_loss=reg_loss*0.03/number

            

            feat_idx=feat_idx[:,0,:,:]

            main_loss2 = self.criterion(x2, y)
            
            
            
            return x2.max(1)[1], reg_loss*0, reg_loss*0 , reg_loss.cuda(), main_loss2
        else:
            
            return x2 


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    input = torch.rand(4, 3, 473, 473).cuda()
    model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=1, use_ppm=True, pretrained=True).cuda()
    model.eval()
    #print(model)
    output = model(input)
    print('PSPNet', output.size())
