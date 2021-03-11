import numpy as np
import cv2
from numpy.linalg import inv

import os
try:
  os.mkdir('proj')
except:
  pass


try:
  os.mkdir('featureidx')
except:
  pass

import open3d as o3d
import glob
paths=glob.glob('/local/xjqi/scannet/scans/scene0474_03')
for path in paths:

  name=path.split('/')[-1]
  if name[:2]=='gt':
    continue
  pc = o3d.io.read_point_cloud('scannet/scans/'+name+'/'+name+'_vh_clean_2.ply')
  coords =  np.array(pc.points[:, :3])
  feats = np.array(pc.colors)*255
  
  
  
  fin=open('scannet_frames_25k/'+name+'/intrinsics_color.txt')
  num=0
  K=np.zeros((4,4))
  for line in fin:
    line.strip()
    nums=line.split(' ')
    K[num,0]=float(nums[0])
    K[num,1]=float(nums[1])
    K[num,2]=float(nums[2])
    K[num,3]=float(nums[3])
    num+=1
  fin.close()
  
  coords=np.concatenate((coords,np.ones((coords.shape[0],1))),1)
  
  impaths=glob.glob('scannet_frames_25k/'+name+'/pose/*.txt')
  for impath in impaths:
    imname=impath.split('/')[-1].split('.')[0]
    
    
    label=cv2.imread('scannet_frames_25k/'+name+'/label/'+imname+'.png')
    
    h=label.shape[0]
    w=label.shape[1]
    
    fin=open('scannet_frames_25k/'+name+'/pose/'+imname+'.txt')
    num=0
    rt=np.zeros((4,4))
    for line in fin:
      line=line.strip()
      nums=line.split(' ')
      rt[num,0]=float(nums[0])
      rt[num,1]=float(nums[1])
      rt[num,2]=float(nums[2])
      rt[num,3]=float(nums[3])
      num+=1
    fin.close()
    rt=inv(rt)
    
    
    coords2=np.matmul(rt,np.transpose(coords))
    
    coords3=np.matmul(K,coords2)
    
    coords3=np.transpose(coords3)
  
    coords3[:,0]=coords3[:,0]/coords3[:,2]
    coords3[:,1]=coords3[:,1]/coords3[:,2]
    
    coords3=coords3.astype('int')
    
    
    im=np.zeros((w,h,3))
    
    
    feat=np.zeros((w,h))
    
    depth=cv2.imread('scannet_frames_25k/'+name+'/depth/'+imname+'.png',-1)/1000.0
    depth=cv2.resize(depth,(w,h),interpolation=cv2.INTER_NEAREST)
    for i in range(coords3.shape[0]):
        if coords3[i,1]>=h or coords3[i,1]<0 or coords3[i,0]>=w or coords3[i,0]<0 or coords3[i,2]<=0:
          continue
        if depth[coords3[i,1],coords3[i,0]]==0:
          continue
        if abs(coords3[i,2]-depth[coords3[i,1],coords3[i,0]])>2 or abs(coords3[i,2]-depth[coords3[i,1],coords3[i,0]])/depth[coords3[i,1],coords3[i,0]]>0.5:
          continue

        #We use depth to filter out the projected 3D points which should be occluded, but visible due to the sparsity of the point cloud. You can also try to filter them out by prediction and gt like the following:
        #if label[coords3[i,1],coords3[i,0]]!=pred:
        #  continue
        im[coords3[i,0],coords3[i,1],0]=feats[i,2]
        im[coords3[i,0],coords3[i,1],1]=feats[i,1]
        im[coords3[i,0],coords3[i,1],2]=feats[i,0]
        feat[int(coords3[i,0]),int(coords3[i,1])]=i
    im=np.transpose(im, (1,0,2))
    feat=np.transpose(feat, (1,0))
    
    np.save("featureidx/"+name+'_'+imname+".npy",feat)  
    cv2.imwrite("proj/"+name+'_'+imname+".jpg",im)
    
    print ('saved')
    #print (coords3)
