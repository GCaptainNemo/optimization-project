import numpy as np
import torch
from sklearn.datasets import fetch_rcv1
from src.LBFGS import LogisticRegression

class DataAccessObject:
    def __init__(self):
        """
            dir(rcv1) = ['DESCR', 'data', 'sample_id', 'target', 'target_names']
            data是 n_samples x n_feature的二维numpy.ndarray数组
            target是 n_samples一维numpy.ndarray数组
        """
        self.load_data()

    def load_data(self):
        rcv1 = fetch_rcv1(subset='train', download_if_missing=False)
        x = rcv1.data.A  # numpy.float64
        x = x.astype(np.float32)  # 修改数据类型，否则就会出错
        self.xArray = torch.from_numpy(x)
        # csr_matrix -> numpy.ndarray -> torch.tensor
        y = rcv1.target.A
        y = y.astype(np.float32)  # 修改数据类型，否则就会出错
        self.yArray = torch.from_numpy(y)


the_model = torch.load("LBFGS_model_epoch1000.pt")
the_model.to(torch.device("cpu"))
print(the_model.linearTransform.weight[0])
print(type(the_model))


if __name__ == "__main__":
    DAO = DataAccessObject()

    # for i in range(DAO.xArray.shape[0]):
    output = the_model(DAO.xArray)
    mask = output.ge(0.5).float()
    correct = (mask == DAO.yArray).sum()
    acc = correct.item() / DAO.xArray.size(0)

    print("acc = ", acc) # 99.81640675623137
    print(DAO.xArray.size(0))  #23149
