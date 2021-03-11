import glob,cv2
import numpy as np
f=open('ade20k_colors.txt')
l=[]
f1=open('/media/lzz/f1cc9be0-388f-421d-a473-5b33192a9893/PointWeb/dataset/s3dis/s3dis_names.txt')
for line in f1:
  l.append(line.strip())
print (len(l))
im=np.zeros((200,1500,3))
i=0
for line in f:
  c1=line.strip().split(' ')
  im[:,i*100:i*100+100,0]=c1[2]
  im[:,i*100:i*100+100,1]=c1[1]
  im[:,i*100:i*100+100,2]=c1[0]
  i+=1
  print (i)
  if i==15:
    break
for i in range(13):
  cv2.putText(im,l[i],(100*i,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
  if i==12:
    break

cv2.imwrite('vis.png',im)
