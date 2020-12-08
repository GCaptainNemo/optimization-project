import torch
from torch import nn
from sklearn.datasets import fetch_rcv1
import numpy as np
import torch.utils.data as Data
import pickle

# 超参数
EPOCH = 100
LAMBDA = 0.01

# 加载数据
class DataAccessObject:
    def __init__(self):
        """
            dir(rcv1) = ['DESCR', 'data', 'sample_id', 'target', 'target_names']
            data是 n_samples x n_feature的二维numpy.ndarray数组
            target是 n_samples一维numpy.ndarray数组
        """
        self.load_data()
        # self.load_batch()

    def load_data(self):
        rcv1 = fetch_rcv1(subset='train', download_if_missing=False)
        x = rcv1.data.A  # numpy.float64
        x = x.astype(np.float32)  # 修改数据类型，否则就会出错
        self.xArray = torch.from_numpy(x)
        # csr_matrix -> numpy.ndarray -> torch.tensor
        y = rcv1.target.A
        y = y.astype(np.float32)  # 修改数据类型，否则就会出错
        self.yArray = torch.from_numpy(y)

# 定义LogisticRegression
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        # linearTransform = wx + b
        self.linearTransform = nn.Linear(47236, 103)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linearTransform(x)
        x = self.sigmoid(x)
        return x


class MyLossFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(-y * torch.log(x) + - (1 - y) * torch.log(1 - x))


# optimization process
if __name__ == '__main__':
    logistic_model = torch.load("BGD_model_100.pt")
    # logistic_model = LogisticRegression()
    DAO = DataAccessObject()

    criterion = MyLossFunction()
    # criterion = nn.BCELoss()
    use_gpu = torch.cuda.is_available()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    DAO.xArray = DAO.xArray.to(device)
    DAO.yArray = DAO.yArray.to(device)

    logistic_model.to(device)
    criterion.to(device)
    optimizer = torch.optim.SGD(logistic_model.parameters(),
                                lr=3)
    loss_lst = []
    for num_epoch in range(EPOCH):
        print(num_epoch)
        output = logistic_model(DAO.xArray)  # get_out for every net
        regularization_loss = 0
        for par in logistic_model.parameters():
            regularization_loss += \
                torch.sum(torch.pow(par, 2))
        classify_loss = criterion(output, DAO.yArray)  # compute loss for every net
        loss = classify_loss + LAMBDA * regularization_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # apply gradient
        loss_lst.append(loss.item()) # loss recoder
        print('Epoch [{}/{}], Loss: {:.4f}'.format(num_epoch + 1, EPOCH, loss.item()))
    torch.save(logistic_model, "BGD_model_100.pt")
    with open("loss_BGD.pkl", "wb") as f:
        pickle.dump(loss_lst, f)
    print(loss)

