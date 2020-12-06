import torch
from torch import nn
from torch.autograd import Variable
from sklearn.datasets import fetch_rcv1
import numpy as np
import torch.utils.data as Data
import pickle

# 超参数
BATCH_SIZE = 32
EPOCH = 1000

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
        x = x.astype(np.float32)
        self.xArray = torch.from_numpy(x)
        # csr_matrix -> numpy.ndarray -> torch.tensor
        y = rcv1.target.A
        y = y.astype(np.float32)  # 修改数据类型，否则会出错
        self.yArray = torch.from_numpy(y)

    def load_batch(self):
        self.torch_dataset = Data.TensorDataset(self.xArray, self.yArray)
        self.loader = Data.DataLoader(
            dataset=self.torch_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,  # true表示每个epoch需要洗牌
            num_workers=2,  # 每次训练有两个线程进行的
        )


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
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logistic_model.to(device)
    criterion.to(device)
    # 显存需大于等于4GB
    x_data = DAO.xArray.to(device)
    y_data = DAO.yArray.to(device)
    optimizer = torch.optim.LBFGS(logistic_model.parameters(),
                      lr=1, max_iter=20, max_eval=None,
                      tolerance_grad=1e-05, tolerance_change=1e-09,
                      history_size=100, line_search_fn=None)

    loss_lst = []
    for epoch in range(1000):
        # 前向
        out = logistic_model(x_data)
        loss = criterion(out, y_data)
        print_loss = loss.data.item()
        mask = out.ge(0.5).float()
        correct = (mask == y_data).sum()
        acc = correct.item() / x_data.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_lst.append(loss.item())  # loss recoder
        if (epoch + 1) % 20 == 0:
            print('*' * 10)
            print('epoch {}'.format(epoch + 1))
            print('loss is {:.4f}'.format(print_loss))
            print('acc is {:.4f}'.format(acc))
    torch.save(logistic_model, "LBFGS_model_epoch1000.pt")
    with open("loss_lbfgs.pkl", "wb") as f:
        pickle.dump(loss_lst, f)
