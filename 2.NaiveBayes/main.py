import numpy as np
from sklearn import model_selection
import scipy.stats as st


def dividetf(test):
    row = len(test)
    column = len(test[0]) - 1
    target = np.zeros(row)
    feature = np.zeros([row, column])
    for i in range(row):
        target[i] = test[i][0]
        for j in range(column):
            feature[i][j] = test[i][j + 1]
    return row, column, target, feature


def naivebayes(train):
    num_data, num_feature, target, feature = dividetf(train)
    for i in range(num_data):
        for j in range(num_feature):
            feature[i][j] = train[i][j + 1]
    label = np.unique(target)
    num_label = label.size
    mean = np.zeros([num_label, num_feature])
    std = np.zeros([num_label, num_feature])
    prior = np.zeros(num_label)
    for i in range(num_label):
        temp = np.sum(target == label[i])
        prior[i] = temp / num_data
        tmp_index = np.where(target == label[i])[0]
        for j in range(num_feature):
            tmp = np.zeros([1, len(tmp_index)])
            for k in range(len(tmp_index)):
                tmp[0][k] = feature[tmp_index[k]][j]
            mean[i][j] = np.mean(tmp)
            std[i][j] = np.std(tmp)
    return prior, mean, std, num_label


def naivebayesclassify(score, prior, mean, std, label):
    row, col = score.shape
    post = np.zeros([row, label])
    estimated = np.zeros(row)
    for i in range(row):
        for j in range(label):
            p_product = 1
            for k in range(col):
                p_product = p_product * st.norm.pdf(score[i][k], mean[j][k], std[j][k])
            post[i][j] = p_product * prior[j]
        estimated[i] = np.argmax(post[i])
    return estimated


dataset = [
    [1, 6, 180, 12],
    [1, 5.92, 190, 11],
    [1, 5.58, 170, 12],
    [1, 5.92, 165, 10],
    [0, 5, 100, 6],
    [0, 5.5, 150, 8],
    [0, 5.42, 130, 7],
    [0, 5.75, 150, 9]
]
dataset = np.array(dataset)
np.random.shuffle(dataset)
kfold = model_selection.KFold(n_splits=3, shuffle=False)
for train_idx, test_idx in kfold.split(dataset):
    data_train = dataset[train_idx]
    data_test = dataset[test_idx]
test_num, fea_num, test_lab, score = dividetf(data_test)
prior, mean, std, label = naivebayes(data_train)
y_test = naivebayesclassify(score, prior, mean, std, label)
right = y_test == test_lab
rate = np.sum(right) / test_num
print('Training accuracy is {:.2%}.'.format(rate))

print("输入需要判定的人数：")
amount = int(input())
data_info = np.zeros([amount, fea_num])
print('输入体型数据：')
for i in range(amount):
    data_info[i] = input().split(" ")
result_info = naivebayesclassify(data_info, prior, mean, std, label)
for i in range(amount):
    print('Data: %.2f  %.2f %.2f is ' % (data_info[i][0], data_info[i][1], data_info[i][2]), end=" ")
    if result_info[i] == 1:
        print('male')
    else:
        print('female')
print(result_info)
