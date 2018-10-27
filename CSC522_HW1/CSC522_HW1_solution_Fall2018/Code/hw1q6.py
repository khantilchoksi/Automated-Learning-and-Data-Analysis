import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt

# Load the dataset
data_df = pd.read_csv('hw1q6_data.csv')

# Select the columns of features
data_col = data_df.columns
sel_col = data_col[0:6]

# (a) What are the size of diabetic and nondiabetic patients in the dataset?
print('# of diabetic patients: ', len(data_df.loc[data_df['Class'] == 1]))
print('# of nondiabetic patients: ', len(data_df.loc[data_df['Class'] == 0]))
print('\n')

# (b) Whatis the missing rate (%) for each feature?
print('Missing rate for each feature: ')
for i in range(len(sel_col)):
    #print('== Variable: ', sel_col[i], ' ==')
    print(sel_col[i], ': ', (data_df[sel_col[i]] == 0).sum() / len(data_df), '\n')
    
# Remove the rows with missing values
rm_df = data_df.loc[~(data_df[sel_col] == 0).any(axis = 1)]
print('Remaining patients size: ', len(rm_df))

# (c) What are the size of diabetic and nondiabetic patients?
print('# of diabetic patients: ', len(rm_df.loc[rm_df['Class'] == 1]))
print('# of nondiabetic patients: ', len(rm_df.loc[rm_df['Class'] == 0]))
print('\n')

# (d) Compute the mean, median, standard deviation, range, 
# 25th percentiles, 50th percentiles, 75th percentiles for each feature.
for i in range(len(sel_col)):
    print(sel_col[i], ': ')
    tmpData_df = rm_df[sel_col[i]]
    print('Mean: ', np.mean(tmpData_df))
    print('Median: ', np.median(tmpData_df))
    print('Standard Deviation: ', np.std(tmpData_df))
    print('Range: ', np.max(tmpData_df) - np.min(tmpData_df))
    print('25th percentile: ', np.percentile(tmpData_df, 25))
    print('50th percentile: ', np.percentile(tmpData_df, 50))
    print('75th percentile: ', np.percentile(tmpData_df, 75))
    print('\n')

# (e) Create a histogram plot using 10 bins for the two featuresBloodPressure and DiabetesPedigreeFunction, respectively.
sel_feat = ['BloodPressure', 'DiabetesPedigreeFunction']

for i in range(len(sel_feat)):
    print('== Variable: ', sel_feat[i], ' ==')
    tmp_df = rm_df[sel_feat[i]]
    tmp_df = tmp_df.loc[tmp_df != 0]
    
    plt.hist(tmp_df, bins = 10)
    plt.title("Histogram for " + sel_feat[i])
    plt.savefig("Histogram for " + sel_feat[i])
    plt.show()
    
# (f) Create a quantile-quantile plot for the two featuresBloodPressure and DiabetesPedigreeFunction, respectively. 
for i in range(len(sel_feat)):
    print('== Variable: ', sel_feat[i], ' ==')
    tmp_df = rm_df[sel_feat[i]]
    tmp_df = tmp_df.loc[tmp_df != 0]
    
    stats.probplot(tmp_df, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot for " + sel_feat[i])
    plt.savefig("Normal Q-Q plot for " + sel_feat[i])
    plt.show()