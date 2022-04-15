from sklearn.model_selection import train_test_split
from sklearn import datasets, model_selection
import numpy as np
import matplotlib.pyplot as plt

def LeastSquare(data, target):
    row, col = data.shape
    target = np.transpose(np.array([target]))
    alpha = 0.001
    threshold = 0.01
    iteration = 1000
    beta = np.random.rand(col, 1)
    delta = np.dot(np.transpose(data), target) - np.dot(np.dot(np.transpose(data), data), beta)
    beta_new = beta + alpha * delta
    loss = np.zeros(iteration)
    for i in range(iteration):
        loss[i] = np.linalg.norm(target - np.dot(data, beta_new))
        if loss[i] < threshold:
            break
        else:
            beta = beta_new
            delta = np.dot(np.transpose(data), target) - np.dot(np.dot(np.transpose(data), data), beta)
            beta_new = beta + alpha * delta
    plt.plot(loss)
    plt.show()
    return beta

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
X.shape,y.shape
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#拟合
beta = LeastSquare(X_train, y_train)
# 直接解法
a = np.ones(len(X_train))
XX_train = np.c_[a,X_train]
y_mytest = np.c_[np.ones(len(X_test)),X_test].dot(np.linalg.pinv(XX_train).dot(y_train.reshape(-1,1)))
# 计算 RMSE
rmse = np.sqrt(1/len(X_test)*np.sum((y_test.reshape(-1,1)-y_mytest)**2))
print("rmse:")
print(rmse)
