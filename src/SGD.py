import torch
from torch import nn
from torch.autograd import Variable
from sklearn.datasets import fetch_rcv1
import numpy as np
import torch.utils.data as Data
import pickle

# 超参数, BATCH_SIZE = 1即SGD
BATCH_SIZE = 1
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
        self.load_batch()

    def load_data(self):
        rcv1 = fetch_rcv1(subset='train', download_if_missing=False)
        x = rcv1.data.A  # numpy.float64
        x = x.astype(np.float32)  # 修改数据类型，否则就会出错
        self.xArray = torch.from_numpy(x)
        # csr_matrix -> numpy.ndarray -> torch.tensor
        y = rcv1.target.A
        y = y.astype(np.float32)  # 修改数据类型，否则就会出错
        self.yArray = torch.from_numpy(y)

    def load_batch(self):
        self.torch_dataset = Data.TensorDataset(self.xArray, self.yArray)
        self.loader = Data.DataLoader(
            dataset=self.torch_dataset,
            batch_size=1000,
            shuffle=True,  # true表示每个epoch需要洗牌
            num_workers=2,  # 每次训练有两个线程进行的
        )


class data_prefetcher():
    """ 给dataloader 训练加速"""
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        # with torch.cuda.stream(self.stream):
        #     self.next_data = self.next_data.cuda(non_blocking=True)

    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data


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

    logistic_model = LogisticRegression()
    DAO = DataAccessObject()
    criterion = MyLossFunction()
    # criterion = nn.BCELoss()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    logistic_model.to(device)
    criterion.to(device)
    optimizer = torch.optim.SGD(logistic_model.parameters(),
                                lr=1e-3)
    loss_lst = []
    per_iter_loss_lst = []
    # CPU = torch.device("cpu")
    DAO.xArray.to(device)
    DAO.yArray.to(device)

    for num_epoch in range(2):
        print(num_epoch)
        # prefetcher = data_prefetcher(DAO.loader)
        # data = prefetcher.next()
        # # i = 0
        # while data is not None:
        #     # print(i, len(batch_x))
        #     # i += 1
        #     data = prefetcher.next()
        #     batch_x, batch_y = data
        for step, (batch_x, batch_y) in enumerate(DAO.loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            for i in range(len(batch_x)):
                output = logistic_model(batch_x[i])  # get_out for every net
                classify_loss = criterion(output, batch_y[i])  # compute loss for every net
                regular_loss = 0
                for par in logistic_model.parameters():
                    regular_loss += torch.sum(torch.pow(par, 2))
                loss = classify_loss + LAMBDA * regular_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step() # apply gradient
                loss_lst.append(loss.item()) # loss recoder

                # #########################################
                # #########################################
                output = logistic_model(DAO.xArray)
                classify_loss = criterion(output, DAO.yArray)  # compute loss for every net
                regular_loss = 0
                for par in logistic_model.parameters():
                    regular_loss += torch.sum(torch.pow(par, 2))
                loss = classify_loss + LAMBDA * regular_loss
                per_iter_loss_lst.append(loss.item())
        print('Epoch [{}/{}], Loss: {:.4f}'.format(num_epoch + 1, EPOCH, loss.item()))
        # logistic_model.to(device)
    torch.save(logistic_model, "SGD_model_100.pt")
    # with open("loss_sgd.pkl", "wb") as f:
    #     pickle.dump(loss_lst, f)
    with open("loss_sgd_iter.pkl", "wb") as f:
        pickle.dump(per_iter_loss_lst, f)

    print(loss)







