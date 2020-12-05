import numpy as np
testx = np.load('testx', allow_pickle=True)
testy = np.load('testy', allow_pickle=True)
for i in range(47236):
    print(testx[0, i])
print(testx.shape)
