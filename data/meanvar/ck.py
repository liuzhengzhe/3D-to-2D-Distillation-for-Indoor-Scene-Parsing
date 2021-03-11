import numpy as np
m=np.zeros((1,))
m[0]=0.1515
v=np.zeros((1,))
v[0]=0.037

np.save('mean_2d.npy',m)
np.save('var_2d.npy',v)
