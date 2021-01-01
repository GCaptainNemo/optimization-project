import torch
from torch import nn
from sklearn.datasets import fetch_rcv1
import numpy as np
import pickle
import argparse
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

# 超参数

def parse_config():
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('--epochs', '-e', type=int, default=100, required=False, help='number of epochs to train for')
    parser.add_argument('--lamda', '-l', type=float, default=0.01, required=False, help='regularization parameter')
    parser.add_argument('--device', choices=['gpu', 'cpu'], default='gpu', help='use gpu or cpu')
    parser.add_argument('--learning_rate', '-lr', type=float, default=3, required=False, help='learning rate')
    args = parser.parse_args()
    return args


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
        y = rcv1.target.A
        y = y.astype(np.float32)  # 修改数据类型，否则就会出错
        self.yArray = torch.from_numpy(y)


# 定义LogisticRegression
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        # linearTransform = wx + b
        self.linearTransform = nn.Linear(47236, 103, bias=True)
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
    args = parse_config()
    logistic_model = LogisticRegression()
    DAO = DataAccessObject()
    criterion = MyLossFunction()
    # criterion = nn.BCELoss()
    if args.device == "gpu":
        device = torch.device("cuda:0")
        DAO.xArray = DAO.xArray.to(device)
        DAO.yArray = DAO.yArray.to(device)
        logistic_model.to(device)
        criterion.to(device)
    else:
        device = torch.device("cpu")

    print(DAO.xArray.device)
    logistic_model.train()
    criterion.train()
    optimizer = torch.optim.SGD(logistic_model.parameters(),
                                lr=args.learning_rate)
    loss_lst = []
    for num_epoch in range(args.epochs):
        print(num_epoch)
        output = logistic_model(DAO.xArray)  # get_out for every net
        regularization_loss = 0
        logistic_model.named_parameters()
        for par in logistic_model.parameters():
            regularization_loss += \
                torch.sum(torch.pow(par, 2))
        classify_loss = criterion(output, DAO.yArray)  # compute loss for every net
        loss = classify_loss + args.lamda * regularization_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # apply gradient
        loss_lst.append(loss.item()) # loss recoder
        print('Epoch [{}/{}], Loss: {:.4f}'.format(num_epoch + 1, args.epochs, loss.item()))
    torch.save(logistic_model, "BGD_model_100.pt")
    with open("loss_BGD.pkl", "wb") as f:
        pickle.dump(loss_lst, f)
    print(loss)
