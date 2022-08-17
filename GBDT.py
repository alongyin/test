#利用GBDT算法实现波士顿房价预测
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_squared_error
import numpy as np

boston = load_boston()
X,y = boston.data,boston.target
feature_name = boston.feature_names

#划分数据集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


#设置GBDT的参数
params = {
    'n_estimators':500, #弱分类器的个数
    'max_depth':3, #弱分类器（CART回归树）的最大深度
    'min_samples_split':5, #分裂内部节点所需的最小样本数
    'learning_rate':0.04, # 学习率
    'loss':'ls' #损失函数：均方误差损失函数
}

#模型实例化，并训练
GBDTreg = GradientBoostingRegressor(**params)
GBDTreg.fit(X_train,y_train)

#输出并可视化
y_predict = GBDTreg.predict(X_test)
mpl.rcParams['font.sans-serif'] = ['KaiTi','SimHei','FangSong'] # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
mpl.rcParams['font.size'] = 12 #字体大小
plt.plot(y_predict,label='predict price')
plt.plot(y_test,label='real price')
plt.legend()
plt.savefig("test5.jpg")

#计算预测房价和真实房价之间的误差
mse = mean_squared_error(y_test,y_predict)
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

#可视化随着弱学习期的增多时偏差的变化
test_score = np.zeros((params['n_estimators'],),dtype=np.float64)
for i,y_pred in enumerate(GBDTreg.staged_predict(X_test)):
    test_score[i] = GBDTreg.loss_(y_test,y_pred)

#绘制偏差随弱学习器个数的变化
fig = plt.figure(figsize=(6,6))
plt.subplot(1,1,1)
plt.title('the difference varies with the number of weak learners')
plt.plot(np.arange(params['n_estimators']) + 1, GBDTreg.train_score_, 'b-',
         label='training set bias')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='test set bias')
plt.legend(loc='upper right')
plt.xlabel('number of weak learners')
plt.ylabel('deviation')
fig.tight_layout()
fig.savefig("test6.jpg")

#可视化特征重要性