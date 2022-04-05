import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
#from sklearn.model_selection import train_test_split
#pandas 读取 csv 文件
data = pd.read_csv('PlayTennis.csv')
#指定列
#data.columns = ['outlook','temp','humidity','windy','play']
#sparse=False是不产生稀疏矩阵
vec=DictVectorizer(sparse=False)
#先用 pandas 对每行生成字典，然后进行向量化
feature = data[['outlook','temp','humidity','windy']]
X_train = vec.fit_transform(feature.to_dict(orient='record'))
Y_train = data['play'].to_numpy()
#打印各个变量
print('show feature\n',feature)
print('show vector\n',X_train)
print('show vector name\n',vec.get_feature_names())
#train_x, test_x, train_y, test_y = train_test_split(X_train, Y_train, test_size = 0.3)
clf = tree.DecisionTreeClassifier(criterion='gini')
clf.fit(X_train,Y_train)
#保存成 dot 文件
with open("out.dot", 'w') as f :
    f = tree.export_graphviz(clf, out_file = f,
            feature_names = vec.get_feature_names())