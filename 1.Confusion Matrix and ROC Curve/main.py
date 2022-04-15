import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

with open('score.csv', newline='') as f:
    reader = csv.reader(f)
    s = list(reader)
tmp = s[0]
score = np.array([float(item) for item in tmp])

with open('label.csv', newline='') as f:
    reader = csv.reader(f)
    l = list(reader)
tmp = l[0]
label = np.array([float(item) for item in tmp])
#将label为1的作为P类，把label为0的作为N类，阈值设为0.05时，计算混淆矩阵，计算TPR，FPR，Precision，Recall，F1-score，Accuracy
threshold = 0.05
label_pred = score.copy()
label_pred[label_pred > threshold] = 1
label_pred[label_pred < threshold] = 0
cm = confusion_matrix(label, label_pred)
print('混淆矩阵: \n',cm)
tpr = cm[1][1]/(cm[0][0]+cm[1][1])
print('TPR: ', tpr)
fpr = cm[0][1]/(cm[0][1]+cm[1][0])
print('FPR: ', fpr)
precision = precision_score(label, label_pred)
print('Precision: ', precision)
recall = recall_score(label, label_pred)
print('Recall: ', recall)
f1 = f1_score(label, label_pred)
print('F1 score: ', f1)
accuracy = accuracy_score(label, label_pred)
print('Accuracy: ', accuracy)
#绘制ROC曲线，计算AUC
plt.figure()
fpr, tpr, threshold = roc_curve(label, score)
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)
plt.plot(
    fpr,
    tpr,
    color="red",
    label="ROC Curve (Area = '%0.2f')" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()

