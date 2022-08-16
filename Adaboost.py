import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

# 生成2维正态分布，生成的数据按分位数分为两类，500个样本,2个样本特征，协方差系数为2
x1,y1 = make_gaussian_quantiles(cov=3.0,n_samples=500,n_features=2,n_classes=2,random_state=1)
# 生成2维正正态分布，生成的数据按照分位数分为两类，400个样本，2个样本特征均值都为3，协方差系数为2
x2,y2 = make_gaussian_quantiles(mean=(4, 3), cov=1.5,n_samples=400, n_features=2, n_classes=2, random_state=1)

#两组数据合成一组数据
X = np.concatenate((x1,x2))
y = np.concatenate((y1,-y2+1))

print(X)
print(y)

plt.scatter(X[:,0],X[:,1],marker='o',c=y)
plt.show()
plt.savefig("test3.jpg")