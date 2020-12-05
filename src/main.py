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
    # criterion = nn.BCELoss()
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logistic_model.to(device)
    criterion.to(device)

    # if(use_gpu):
    #     logistic_model = logistic_model.cuda()
    #     criterion = criterion.cuda()
    #     xArray = xArray.cuda()
    #     yArray = yArray.cuda()


    # torch.optim.LBFGS(logistic_model.parameters(),
    #                   lr=1, max_iter=20, max_eval=None,
    #                   tolerance_grad=1e-05, tolerance_change=1e-09,
    #                   history_size=100, line_search_fn=None)

    optimizer = torch.optim.SGD(logistic_model.parameters(),
                                lr=1e-3)
    loss_lst = []
    for num_epoch in range(1000):
        print(num_epoch)
        for step, (batch_x, batch_y) in enumerate(DAO.loader):
            # b_x = Variable(batch_x)
            # b_y = Variable(batch_x)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = logistic_model(batch_x)  # get_out for every net
            loss = criterion(output, batch_y)  # compute loss for every net
            # if don't call zero_grad,
            # the grad of each batch will be accumulated
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # apply gradient
            loss_lst.append(loss.item()) # loss recoder
        # if (num_epoch + 1) % 5 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(num_epoch + 1, EPOCH, loss.item()))
    torch.save(logistic_model, "mini_batchGD_model_epoch1000.pt")
    with open("loss.pkl", "wb") as f:
        pickle.dump(loss_lst, f)
    print(loss)


# for epoch in range(1000):
#     x_data = Variable(xArray)
#     y_data = Variable(yArray)
#
#     # 前向
#     out = logistic_model(x_data)
#     loss = criterion(out, y_data)
#     print_loss = loss.data.item()
#     mask = out.ge(0.5).float()
#     correct = (mask == y_data).sum()
#     acc = correct.item() / x_data.size(0)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if (epoch + 1) % 20 == 0:
#         print('*' * 10)
#         print('epoch {}'.format(epoch + 1))
#         print('loss is {:.4f}'.format(print_loss))
#         print('acc is {:.4f}'.format(acc))
#
# # 参数输出
# w0, w1 = logistic_model.linearTransform.weight[0]
# w0 = float(w0.item())
# w1 = float(w1.item())
# b = float(logistic_model.linearTransform.bias.item())
#
# print('w0:{}\n'.format(w0), 'w1:{}\n'.format(w1), 'b:{0}'.format(b))





