import torch
from torch import nn
from torch.autograd import Variable

# 构造数据
n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data)
y1 = torch.ones(100)

x = torch.cat((x0, x1)).type(torch.FloatTensor)
y = torch.cat((y0, y1)).type(torch.FloatTensor)


# 定义LogisticRegression
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linearTransform = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linearTransform(x)
        x = self.sigmoid(x)
        return x
logistic_model = LogisticRegression()
criterion = nn.BCELoss()
use_gpu = torch.cuda.is_available()


# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model=model.to(device)
# x=x.to(device)
# y=y.to(device)

if(use_gpu):
    logistic_model = logistic_model.cuda()
    criterion = criterion.cuda()
    x = x.cuda()
    y = y.cuda()


optimizer = torch.optim.SGD(logistic_model.parameters(), lr=1e-3, momentum=0.9)

# 训练过程
print("torch.cuda = ", torch.cuda)
for epoch in range(1000):
    x_data = Variable(x)
    y_data = Variable(y)

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

    if (epoch + 1) % 20 == 0:
        print('*' * 10)
        print('epoch {}'.format(epoch + 1))
        print('loss is {:.4f}'.format(print_loss))
        print('acc is {:.4f}'.format(acc))

# 参数输出
w0, w1 = logistic_model.linearTransform.weight[0]
w0 = float(w0.item())
w1 = float(w1.item())
b = float(logistic_model.linearTransform.bias.item())

print('w0:{}\n'.format(w0), 'w1:{}\n'.format(w1), 'b:{0}'.format(b))





