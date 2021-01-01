import pickle
import matplotlib.pyplot as plt
from pylab import mpl
import torch

mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题
with open("../add_regular_result/loss_minibatch_iter.pkl", "rb") as f:
    loss = pickle.load(f)
    print(len(loss))
print([i for i, name in enumerate(loss) if name < 0.6439][0])

sgd_loss = []
with open("../add_regular_result/loss_sgd_iter.pkl", "rb") as f:
    sgd_loss = pickle.load(f)



plt.figure(0)
plt.plot([i for i in range(1, len(loss) + 1)], loss,
         label='步长为3')
plt.legend()
plt.title("Mini-batch Gradient Descent loss-iteration curve")
plt.xlabel("iteration")

plt.ylabel("loss")
plt.show()

