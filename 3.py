import pandas as pd
data = pd.read_csv('/content/zoo_data.csv')
data.head()
y=data["1.7"]

import numpy as np
y = np.array(y)
print(y)

x= data.drop(columns=["1.7"])
print(x)
x
x = np.array(x)
x

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3)
X_train
X_test
Y_train
Y_test
inst = DecisionTreeClassifier()
inst = inst.fit(X_train,Y_train)
Y_pred = inst.predict(X_test)
Y_pred

from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
acc = metrics.accuracy_score(Y_test,Y_pred)
print(acc)
print("Precision,recall,F score is",precision_recall_fscore_support(Y_test,Y_pred, average ='macro'))

import matplotlib.pyplot as plt
from sklearn import tree
plt.figure(figsize=(10,10))
tree.plot_tree(inst,fontsize=8)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,Y_pred))

from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))
