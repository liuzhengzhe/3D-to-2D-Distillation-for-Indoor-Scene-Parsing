import shutil
import glob,cv2
import numpy as np
import os
try:
  os.mkdir('../data/scannet_frames_25k/gt/')
except:
  pass

paths=glob.glob('../data/scannet_frames_25k/scene*/label/*.png')
for path in paths:
  im=cv2.imread(path)
  fd=path.split('/')[-3]
  name=path.split('/')[-1]
  shutil.copy(path,'../data/scannet_frames_25k/gt/'+fd+'_'+name)
  print (np.unique(im))
