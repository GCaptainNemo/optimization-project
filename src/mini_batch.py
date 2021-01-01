import torch
from torch import nn
from sklearn.datasets import fetch_rcv1
import numpy as np
import torch.utils.data as Data
import pickle
import argparse



def parse_config():
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('--epochs', '-e', type=int, default=100, required=False, help='number of epochs to train for')
    parser.add_argument('--lamda', '-l', type=float, default=0.01, required=False, help='regularization parameter')
    parser.add_argument('--device', choices=['gpu', 'cpu'], default='gpu', help='use gpu or cpu')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=3, required=False, help='learning rate')
    args = parser.parse_args()
    return args

class DataAccessObject:
    def __init__(self, batch_size):
        """
            dir(rcv1) = ['DESCR', 'data', 'sample_id', 'target', 'target_names']
            data是 n_samples x n_feature的二维numpy.ndarray数组
            target是 n_samples一维numpy.ndarray数组
        """
        self.load_data()
        self.load_batch(batch_size)

    def load_data(self):
        rcv1 = fetch_rcv1(subset='train', download_if_missing=False)
        x = rcv1.data.A  # numpy.float64
        x = x.astype(np.float32)  # 修改数据类型，否则就会出错
        self.xArray = torch.from_numpy(x)
        # csr_matrix -> numpy.ndarray -> torch.tensor
        y = rcv1.target.A
        y = y.astype(np.float32)  # 修改数据类型，否则就会出错
        self.yArray = torch.from_numpy(y)

    def load_batch(self, batch_size):
        self.torch_dataset = Data.TensorDataset(self.xArray, self.yArray)
        self.loader = Data.DataLoader(
            dataset=self.torch_dataset,
            batch_size=batch_size*10,
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
    args = parse_config()
    logistic_model = LogisticRegression()
    DAO = DataAccessObject(args.batch_size)
    criterion = MyLossFunction()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.device == "gpu":
        device = torch.device("cuda:0")
        logistic_model.to(device)
        criterion.to(device)

    else:
        device = torch.device("cpu")

    optimizer = torch.optim.SGD(logistic_model.parameters(),
                                lr=args.learning_rate)
    per_iter_loss_lst = []
    loss_lst = []
    DAO.xArray = DAO.xArray.to(device)
    DAO.yArray = DAO.yArray.to(device)

    for num_epoch in range(args.epochs):
        print(num_epoch)
        for step, (batch_x, batch_y) in enumerate(DAO.loader):
            batch_x = batch_x.to(device)    # 512
            batch_y = batch_y.to(device)
            for i in range(16):   # 512 / 32
                output = logistic_model(batch_x[i * 32: (i+1) * 32])  # get_out for every net
                classify_loss = criterion(output, batch_y[i*32: (i+1) * 32])  # compute loss for every net
                regular_loss = 0
                for par in logistic_model.parameters():
                    regular_loss += torch.sum(torch.pow(par, 2))
                loss = classify_loss + args.lamda * regular_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step() # apply gradient

                # ############################################################
                # ############################################################
                output = logistic_model(DAO.xArray)
                classify_loss = criterion(output, DAO.yArray)  # compute loss for every net
                regular_loss = 0
                for par in logistic_model.parameters():
                    regular_loss += torch.sum(torch.pow(par, 2))
                loss = classify_loss + args.lamda * regular_loss
                per_iter_loss_lst.append(loss.item())
                print('iter [{}], Loss: {:.4f}'.format(i, loss.item()))
        with open("loss_minibatch_iter.pkl", "wb") as f:
            pickle.dump(per_iter_loss_lst, f)

        print('Epoch [{}/{}], Loss: {:.4f}'.format(num_epoch + 1, args.epochs, loss.item()))
    torch.save(logistic_model, "minibatch_model.pt")
    with open("loss_minibatch_iter.pkl", "wb") as f:
        pickle.dump(per_iter_loss_lst, f)







