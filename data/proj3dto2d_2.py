import glob,os,cv2
import numpy as np
paths=glob.glob('featureidx/scene*_*_*.npy')
paths.sort()
curr_scene=''


import os
try:
  os.mkdir('eachfeature')
except:
  pass


try:
  os.mkdir('featidx2')
except:
  pass


for path in paths:
        print (path)
        name=path.split('/')[-1]

        scene='_'.join(name.split('_')[:2])
        if curr_scene!=scene:
          print ('loading...')
          rawfeat = np.looad('feat/'+scene+'.ply.npy')    
          rawsum=np.sum(rawfeat,1)
          print ('loaded')
          curr_scene=scene
        featidx_im=np.load(path).astype('int')
        kernel = np.ones((3,3), np.uint8) 

        featidx_im = cv2.dilate(featidx_im.astype('float32'), kernel, iterations=1)
        featidx_im = cv2.resize(featidx_im,(162,121),interpolation=cv2.INTER_NEAREST).astype('int')
        
        #featsum=rawsum[featidx]
        #featidx[np.where(featsum==-999)]=0
        




        featidx=featidx_im.astype('int').flatten().tolist()
        featidx=list(filter(lambda a:a!=0,featidx))
        if len(featidx)==0:
          continue
        else:
          featidx=np.asarray(featidx)
          inv=np.zeros((max(featidx)+1))
          for i in range(len(featidx)):
            inv[featidx[i]]=i













        feat=rawfeat[featidx,:]
        np.save('eachfeature/'+name,feat)



        featidx2=inv[featidx_im].astype('int')
        np.save('featidx2/'+name,featidx2)




