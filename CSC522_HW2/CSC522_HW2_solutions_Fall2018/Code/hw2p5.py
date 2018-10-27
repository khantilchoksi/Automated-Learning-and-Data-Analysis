import pandas as pd
import numpy as np
import random
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import graphviz 



filename = 'h2p5data.csv'
data = pd.read_csv(filename)
row_index = data.index.tolist()
feature_list = list(data)[1:-1]

folders = []
for i in range(1, 6):
    temp = []
    for j in row_index:
        #print(j)
        Id = j + 1
        if Id % 5 == i-1:
            temp.append(j)

    folders.append(temp)


print("5-fold CV Result of Naive Bayes Classifier:")

NB_actual_class = []
NB_predicted_class = []
for idx, folder in enumerate(folders):
    folderID = idx + 1
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    gnb = GaussianNB()
    for i in row_index:
        if i not in folder:
            x_train.append(data.loc[row_index[i], feature_list])
            y_train.append(data.loc[row_index[i], 'Class'])
        else:
            x_test.append(data.loc[row_index[i], feature_list])
            y_test.append(data.loc[row_index[i], 'Class'])

    model = gnb.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    NB_actual_class.extend(y_test)
    NB_predicted_class.extend(y_pred)
    print("---Print Probabilities---")
    print(model.predict_proba(x_test))
    

DT_actual_class = []
DT_predicted_class = []
for idx, folder in enumerate(folders):
    folderID = idx + 1
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    clf = tree.DecisionTreeClassifier()
    for i in row_index:
        if i not in folder:
            x_train.append(data.loc[row_index[i], feature_list])
            y_train.append(data.loc[row_index[i], 'Class'])
        else:
            x_test.append(data.loc[row_index[i], feature_list])
            y_test.append(data.loc[row_index[i], 'Class'])

    model = clf.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    DT_actual_class.extend(y_test)
    DT_predicted_class.extend(y_pred)
    #Visualize
    dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=feature_list,  
                                class_names=target_names,  
                                filled=True, rounded=True,  
                                special_characters=True)  
    graph = graphviz.Source(dot_data)  
    print("--Tree--")
    print("See Tree"+str(idx)+".pdf")
    graph.render("Tree"+str(idx))
    
    
#NB Metrics
print("confusion_matrix")
print(confusion_matrix(NB_actual_class, NB_predicted_class, labels=[1, 0]))
#yes = 1; no = 0
target_names = ['1','0']
print(classification_report(NB_actual_class, NB_predicted_class, target_names=target_names))
print("NB Accuracy: ")
print(accuracy_score(NB_actual_class, NB_predicted_class))



print("5-fold CV Result of Decision Tree Classifier:")
#DT Metrics
print("confusion_matrix")
print(confusion_matrix(DT_actual_class, DT_predicted_class, labels=[1, 0]))
target_names = ['1','0']
print(classification_report(DT_actual_class, DT_predicted_class, target_names=target_names))
print("DT Accuracy: ")
print(accuracy_score(DT_actual_class, DT_predicted_class))


# Final Training on the whole dataset
print("Final Training on the whole dataset")

filename = 'h2p5data.csv'
data = pd.read_csv(filename)
x_data = []
y_data = []

gnb2 = GaussianNB()
    
x_data = data.loc[:, feature_list]
y_data = data.loc[:, 'Class']
      
model = gnb2.fit(x_data, y_data)


final_predicted_class = model.predict(x_data)
final_actual_class = y_data   
    
#NB Metrics
print("confusion_matrix")
print(confusion_matrix(final_actual_class, final_predicted_class, labels=[1, 0]))
target_names = ['1','0']
print(classification_report(final_actual_class, final_predicted_class, target_names=target_names))
print("Accuracy: ")
print(accuracy_score(final_actual_class, final_predicted_class))
print("---Print Probabilities---")
print(model.predict_proba(x_data))