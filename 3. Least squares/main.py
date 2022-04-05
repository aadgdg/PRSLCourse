from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
X.shape,y.shape
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# 直接解法
a = np.ones(len(X_train))
XX_train = np.c_[a,X_train]
y_mytest = np.c_[np.ones(len(X_test)),X_test].dot(np.linalg.pinv(XX_train).dot(y_train.reshape(-1,1)))
# 计算 RMSE
rmse = np.sqrt(1/len(X_test)*np.sum((y_test.reshape(-1,1)-y_mytest)**2))
print(rmse)


y_mytest = np.c_[X_test,np.ones(X_test.shape)].dot(np.linalg.pinv(XX_train).dot(y_train))
rmse_nt = ((y_mytest - y_test) ** 2).sum()
rmse_nt = np.sqrt(rmse_nt / y_mytest[0])
## -------------------梯度下降法----------------------------
# 使用 RMSProp
# 初始化参数 Beta 和 学习率 alpha，mu, vt
row,col = XX_train.shape
Beta = np.random.random([col,1])
vt = np.ones([col,1])
alpha = 0.5
mu = 0.9

# 梯度的负方向
delta_Beta = 2/row *( XX_train.T.dot(y_train.reshape(row,1)) \
                              - XX_train.T.dot(XX_train).dot(Beta))

# 更新Beta 运用RMSProp
new_vt = mu*vt + (1-mu)* delta_Beta**2
new_Beta = Beta + alpha *  delta_Beta/(np.sqrt(new_vt)+np.spacing(1))
new_Beta = Beta + alpha*delta_Beta

loss = []
for idx in range(100000):
    tmp_loss = 1/row * np.linalg.norm(y_train.reshape(row,1)-XX_train.dot(Beta))**2
    loss.append(tmp_loss)
    if tmp_loss < 50:
        print(idx)
        break
    else:
        Beta = new_Beta
        vt = new_vt
        delta_Beta = 2/row *( XX_train.T.dot(y_train.reshape(row,1)) \
                              - XX_train.T.dot(XX_train).dot(Beta))
        new_vt = mu*vt + (1-mu)* delta_Beta**2
        new_Beta = Beta + alpha * delta_Beta/(np.sqrt(new_vt)+np.spacing(1))
        new_Beta = Beta + alpha*delta_Beta

print(Beta)
print(np.linalg.pinv(XX_train).dot(y_train).reshape(col,1))
y_mytest_GD = np.c_[np.ones(len(X_test)),X_test].dot(Beta)
rmse_GD = np.sqrt(1/len(X_test)*np.sum((y_test.reshape(-1,1)-y_mytest_GD.reshape(-1,1))**2))
print('the rmse of least squares with Gradient Descent is %f' %rmse_GD)