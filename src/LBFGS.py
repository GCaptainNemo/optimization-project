import torch
from torch import nn
from sklearn.datasets import fetch_rcv1
import numpy as np
import pickle
import argparse

# 超参数
# 加载数据
def parse_config():
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('--epochs', '-e', type=int, default=10, required=False, help='number of epochs to train for')
    parser.add_argument('--lamda', '-l', type=float, default=0.01, required=False, help='regularization parameter')
    parser.add_argument('--device', choices=['gpu', 'cpu'], default='gpu', help='use gpu or cpu')
    args = parser.parse_args()
    return args

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
        x = x.astype(np.float32)
        self.xArray = torch.from_numpy(x)
        print("length = ", len(self.xArray))
        # csr_matrix -> numpy.ndarray -> torch.tensor
        y = rcv1.target.A
        y = y.astype(np.float32)  # 修改数据类型，否则会出错
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
    args = parse_config()
    logistic_model = LogisticRegression()
    DAO = DataAccessObject()
    criterion = MyLossFunction()
    if args.device == "gpu":
        device = torch.device("cuda:0")
        DAO.xArray = DAO.xArray.to(device)
        DAO.yArray = DAO.yArray.to(device)
        logistic_model.to(device)
        criterion.to(device)
    else:
        device = torch.device("cpu")
    logistic_model.train()
    criterion.train()
    optimizer = torch.optim.LBFGS(logistic_model.parameters(),
                      lr=1, max_iter=20, max_eval=None,
                      tolerance_grad=1e-05, tolerance_change=1e-09,
                      history_size=100, line_search_fn=None)

    loss_lst = []
    for epoch in range(args.epochs):
        # 前向
        print("epoch = ", epoch)
        def closure():
            optimizer.zero_grad()
            out = logistic_model(DAO.xArray)
            classify_loss = criterion(out, DAO.yArray)
            regular_loss = 0
            for par in logistic_model.parameters():
                regular_loss += torch.sum(torch.pow(par, 2))
            loss = classify_loss + args.lamda * regular_loss
            loss.backward()
            loss_lst.append(loss.item())
            return loss
        optimizer.step(closure)
          # loss recoder
        print('*' * 10)
        print('epoch {}'.format(epoch + 1))
        print('loss is {:.4f}'.format(loss_lst[-1]))
    # print('acc is {:.4f}'.format(acc))
    torch.save(logistic_model, "LBFGS_model_100.pt")
    with open("loss_lbfgs.pkl", "wb") as f:
        pickle.dump(loss_lst, f)
