
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
#%matplotlib inline

## (a)
# Load the data
train_df = pd.read_csv('q1_train.csv')
test_df = pd.read_csv('q1_test.csv')

print('Training samples: ', len(train_df))
print('Testing samples: ', len(test_df))
print('\n')

print('Rock in training samples: ', len(train_df[train_df.Class == 'R']))
print('Mine in training samples: ', len(train_df[train_df.Class == 'M']))
print('Rock in testing samples: ', len(test_df[test_df.Class == 'R']))
print('Mine in testing samples: ', len(test_df[test_df.Class == 'M']))

x_train = np.array(train_df.loc[:,train_df.columns[:-1]])
x_test = np.array(test_df.loc[:,train_df.columns[:-1]])
y_train = np.array(train_df.loc[:,train_df.columns[-1]])
y_test = np.array(test_df.loc[:,train_df.columns[-1]])

## (b)
pre_dict = {0: 'Normalization', 1: 'Standardization'}
sel_pre = [0,1]

for i in sel_pre: 
    if pre_dict[i] == 'Normalization': 
        # Normalization for training data
        norm_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0))
        print('Nomalization of training data: \n', norm_train[0:5,0:5])
        # Use the min and max of training data to normalization testing data
        min_train = np.min(x_train, axis = 0)
        max_train = np.max(x_train, axis = 0)
        norm_test = (x_test - min_train) / (max_train - min_train)
        print('Nomalization of testing data: \n', norm_test[0:5,0:5], '\n') 
        
    elif pre_dict[i] == 'Standardization': 
        # Standardize for training data
        z_train = stats.zscore(x_train)
        print('Z-score of training data: \n', z_train[0:5,0:5])
        # Use the mean and std of training data to standardize testing data
        mean_train = np.mean(x_train, axis = 0)
        std_train = np.std(x_train, axis = 0)
        z_test = (x_test - mean_train) / std_train
        print('Z-score of testing data: \n', z_test[0:5,0:5], '\n')

## (c)
# Calculate the covariance matrix of the new training data
for i in sel_pre: 
    if pre_dict[i] == 'Normalization': 
        print('Normalization: ')
        cov_norm_train = np.cov(norm_train.T)
        print('Size of covariance matrix: ', np.shape(cov_norm_train))
        print('Covariance matrix of training data: \n', cov_norm_train[0:5,0:5], '\n')
    
    elif pre_dict[i] == 'Standardization': 
        print('Standardization: ')        
        cov_z_train = np.cov(z_train.T)
        print('Size of covariance matrix: ', np.shape(cov_z_train))
        print('Covariance matrix of training data: \n', cov_z_train[0:5,0:5], '\n')

## (d)
# Calculate the eigenvalues and eigenvectors
for i in sel_pre:  
    if pre_dict[i] == 'Normalization': 
        print('Normalization: ')
        norm_eig_vals, norm_eig_vecs = np.linalg.eig(cov_norm_train)
        print('Eigenvalues: \n', norm_eig_vals[0:5])
        print('Eigenvectors: \n', norm_eig_vecs[0:5,0:5], '\n')
        fig = plt.figure(figsize=(8,6))
        plt.plot(np.arange(len(norm_eig_vals)), norm_eig_vals)
        
    elif pre_dict[i] == 'Standardization': 
        print('Standardization: ')
        z_eig_vals, z_eig_vecs = np.linalg.eig(cov_z_train)
        print('Eigenvalues: \n', z_eig_vals[0:5])
        print('Eigenvectors: \n', z_eig_vecs[0:5,0:5], '\n')
        fig = plt.figure(figsize=(8,6))
        plt.plot(np.arange(len(z_eig_vals)), z_eig_vals)
    
    ax = plt.gca()
    ax.grid(linestyle='--')
    plt.xlabel('$i^th$ Principle Component', fontsize=16)
    plt.ylabel('Eigenvalues', fontsize=16)
    plt.title('Eigenvalues of PCs with ' + pre_dict[i], fontsize=16)
    plt.savefig('elbow_'+ pre_dict[i] + '.png')

## (e)
pc_list = [2,4,8,10,20,40,60]

from sklearn.neighbors import KNeighborsClassifier

for i in sel_pre: 
    if pre_dict[i] == 'Normalization': 
        print('Normalization: ')
        eig_vecs = norm_eig_vecs
        train_data = norm_train; test_data = norm_test
    elif pre_dict[i] == 'Standardization': 
        print('Standardization: ')        
        eig_vecs = z_eig_vecs
        train_data = z_train; test_data = z_test
    
    acc = []
    for k in range(len(pc_list)):
        temp_pc_num = pc_list[k]
        pca_train = train_data.dot(eig_vecs[:,0:temp_pc_num])
        pca_test = test_data.dot(eig_vecs[:,0:temp_pc_num])
        clf = KNeighborsClassifier(n_neighbors=3)
        clf = clf.fit(pca_train, y_train)
        y_pred = clf.predict(pca_test)
        acc.append(float(sum(y_pred == y_test) / len(y_test)))
    
    print('Accuracy when PC = 10: ', acc[3])
    fig = plt.figure(figsize=(8,6))
    plt.plot(pc_list, acc)
    ax = plt.gca()
    ax.grid(linestyle='--')
    plt.xlabel('Dimensions', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy across different PCs by KNN with ' + pre_dict[i], fontsize=16)
    plt.savefig('acc_curve_' + pre_dict[i] + '.png')

