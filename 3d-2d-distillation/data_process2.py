import glob,os
import cv2
import numpy as np

try:
  os.mkdir('../data/gt_ignore')
except:
  pass

paths=glob.glob('../data/scannet_frames_25k/gt/*.png')

for path in paths:
  name=path.split('/')[-1]
  if os.path.exists('../data/scannet_frames_25k/gt_ignore/'+name)==1:
    print ('exist')
    continue

  im=cv2.imread(path)
  im[np.where(im==0)]=255
  im[np.where(im==13)]=255
  im[np.where(im==15)]=255
  im[np.where(im==17)]=255
  im[np.where(im==18)]=255
  im[np.where(im==19)]=255
  im[np.where(im==20)]=255
  im[np.where(im==21)]=255
  im[np.where(im==22)]=255
  im[np.where(im==23)]=255
  im[np.where(im==25)]=255
  im[np.where(im==26)]=255
  im[np.where(im==27)]=255
  im[np.where(im==29)]=255
  im[np.where(im==30)]=255
  im[np.where(im==31)]=255
  im[np.where(im==32)]=255
  im[np.where(im==35)]=255
  im[np.where(im==37)]=255
  im[np.where(im==38)]=255
  im[np.where(im==40)]=255
  cv2.imwrite('../data/scannet_frames_25k/gt_ignore/'+name,im)
