# rcv1 dataset
from sklearn.datasets import fetch_rcv1
import pickle
rcv1 = fetch_rcv1(subset='train', download_if_missing=False)
Xarray = rcv1.data.A
Yarray = rcv1.target.A
test_Y = Yarray[0:10, :]
test_X = Xarray[0:10, :]
test_X.dump('testx')
test_Y.dump('testy')



# Xarray.shape = (23149, 47236)
# Yarray.shape = (23149, 103)
print("type = ", type(test_Y))
print("type = ", test_Y.shape)
# dir(rcv1) = ['DESCR', 'data', 'sample_id', 'target', 'target_names']
# data是 n_samples x n_feature的二维numpy.ndarray数组
# target是 n_samples一维numpy.ndarray数组





