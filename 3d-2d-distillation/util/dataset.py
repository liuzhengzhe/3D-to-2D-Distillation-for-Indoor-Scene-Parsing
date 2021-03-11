import os
import os.path
import cv2,torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        scene=image_path.split('/')[-3]
        #room='_'.join(image_path.split('/')[7].split('_')[2:4])
        #frame=image_path.split('/')[-1]#.split('_')[5]
        imname=image_path.split('/')[-1].split('.')[0]
        if os.path.exists('../data/featidx2/'+scene+'_'+imname+'.npy')==0:
          featidx=np.zeros((968,1296))
          if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]: # or image.shape[0]!=featidx.shape[0]  or image.shape[1]!=featidx.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
          if self.transform is not None:
            image, label, featidx = self.transform(image, label, featidx)
          feat=np.zeros((96,60,60))
          featidx=np.zeros((60,60))
          return image, label, torch.Tensor(feat), torch.Tensor(featidx).double()



        featidx=np.load('../data/featidx2/'+scene+'_'+imname+'.npy').astype('int32')


        
        
        
        #kernel = np.ones((5,5), np.uint8) 
        #featidx = cv2.dilate(featidx, kernel, iterations=1)
        
        eachfeature=np.load('../data/eachfeature/'+scene+'_'+imname+'.npy')
        #print ('each',eachfeature.shape,flush=True)
        #featidx=np.load('/media/lzz/faef702d-ca59-4dad-a70a-089cc782183e/featidx2_pred/Area_'+area+'_'+imname+'.npy')
        #eachfeature=np.load('/media/lzz/faef702d-ca59-4dad-a70a-089cc782183e/eachfeature_pred/Area_'+area+'_'+imname+'.npy')


        featidx=cv2.resize(featidx,(1296,968),interpolation=cv2.INTER_NEAREST)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]: 
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label, featidx = self.transform(image, label, featidx)

        #print (featidx.dtype,torch.sum(featidx),'a',flush=True)
        
        featidx=featidx.unsqueeze(0).unsqueeze(0).double()
        

        featidx=F.interpolate(featidx, (60,60), mode='nearest')

        featidx=featidx.squeeze(0).squeeze(0)
        
        
        #print (torch.sum(featidx),'b',flush=True)
        featidx=featidx.long()



        feat=eachfeature[featidx,:]

        feat=np.transpose(feat,(2,0,1))
        
        '''np.save('featidx.npy',featidx)
        np.save('feat.npy',feat)
        np.save('im.npy',image.detach().cpu().numpy())
        exit()'''



        featidx[featidx>0]=1
        featidx=featidx.double()

        return image, label, torch.Tensor(feat), featidx

class SemDataTest(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label,label = self.transform(image, label,label)
        return image, label
