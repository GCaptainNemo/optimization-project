# 大规模机器学习优化方法比较
## 实验内容：
用带二范数正则化函数的logistic回归解决RCV1数据集上的多标签文本主题分类问题，测试比较了可微严格凸函数上，大规模机器学习优化算法的收敛速率，包括：
1. BGD
2. mini-batch GD
3. SGD
4. L-BFGS

## 对应代码：
1. src/BGD.py
2. src/mini_batch.py
3. src/SGD.py
4. src/LBFGS.py

## 环境：
python3.6 + pytorch + sklearn + numpy

其中RCV1数据集通过sklearn的fetch_rcv1函数下
## 安装与使用
a. python依赖库安装
```
pip install -r requirements.txt
```
b. 参数查询
```
python xxx.py -h
```

## 参考文献
[1] H. Robbins and S. Monro, A Stochastic Approximation Method, The Annals of Mathematical
Statistics, 1951, vo.22, pp.400-407. 

[2]L. Bottou, F. E. C.urtis, J. Nocedal, Optimization Methods for large-Scale Machine Learning,
http://arxiv.org/abs/1606.04838 Submit to SIAM Review.

[3]D. D. Lewis, Y. Yang, T. G. Rose, and F. Li, RCV1: A New Benchmark Collection of Text
Categorization Research, Journal of Machine Learning Research, 2004, vo.5, pp. 361-397.

[4]R. H. Byrd，G. M. Chin, J. Nocedal，F Oztoprak A Family of Second-Order Methods for Convex
l1-Regularized Optimization, Mathematical Programming, 2016, vo. 159, no. 1, pp. 435-467. 

[5]Frank E. Curtis_ Katya Scheinberg, Optimization Methods for Supervised Machine Learning:
From Linear Models to Deep Learning, https://arxiv.org/abs/1706.10207
