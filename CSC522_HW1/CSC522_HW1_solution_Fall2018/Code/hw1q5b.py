# -*- coding: utf-8 -*-
"""
ALDA Fall 2018
HW1 Question#5: sample solution (NORMALIZED DATASET)
"""
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial
from numpy.linalg import inv
from pandas import Series
from sklearn.preprocessing import MinMaxScaler

def Normalize(y_raw):
    # define contrived series
    series = Series(y_raw)
    values = series.values
    values = values.reshape((len(values), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))#normalized
    scaler = scaler.fit(values)
    print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))#normalize
    norm = scaler.transform(values)
    y_norm = []
    for each in norm:
        y_norm.extend(np.round(each,4))

    y = np.array(y_norm)
    return(y)

xlabel = 'area_A'
ylabel = 'kernel_width'

#5(a) Load the file and read '' and '' column
data = pd.read_csv('seeds_dataset.csv', usecols=[xlabel, ylabel])
data[xlabel] = Normalize(data[xlabel])
data[ylabel] = Normalize(data[ylabel])
#5(b)2D plot which label X axis as lat and y as longitude
fig = plt.figure(5)
plt.plot(data[xlabel], data[ylabel], 'b.')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.show()

#5(c) Compute mean of x and y
P = data.mean()
v2 = [P[xlabel], P[ylabel]]
print("P: ",P)
# for each row in the data frame, transfer it to vector.
def v1(row):
    v1 = [row[xlabel],row[ylabel]]
    return v1
        
#5(C)-a Calculate each row's euclidean distance   
ed = data.apply(lambda row: spatial.distance.euclidean(v1(row),v2), axis =1)

#5(c)-a Calculate Mahalanobis distance  
md = data.apply(lambda row:scipy.spatial.distance.mahalanobis(v1(row), v2, inv(np.cov(data[xlabel],data[ylabel]))), axis =1)

#5(c)-a Calculate City block metric
cbm = data.apply(lambda row: scipy.spatial.distance.cityblock(v1(row), v2), axis =1)

#5(c)-a Minkowski metric(for p=3)   
mkski = data.apply(lambda row: scipy.spatial.distance.minkowski(v1(row), v2, 3), axis =1)

#5(c)-a Chebyshev distance    
cd = data.apply(lambda row: scipy.spatial.distance.chebyshev(v1(row), v2), axis =1)

#5(c)-a Cosine distance   
cosined = data.apply(lambda row: scipy.spatial.distance.cosine(v1(row), v2), axis =1)

#5(c)-a canberra distance   
canberra = data.apply(lambda row: scipy.spatial.distance.canberra(v1(row), v2), axis =1)

#5(c)-b Define the function to get the 10 points which are closed to the point P
def min_dist(dist,x):
    datacopy = data.copy()
    datacopy[x] = dist
    min_dist10 = datacopy.nsmallest(10, columns = x)
    return min_dist10
    
ed10 = min_dist(ed,'ed')  
md10 = min_dist(md,'md')   
cbm10 = min_dist(cbm,'cbm') 
mkski10 = min_dist(mkski,'mkski') 
cd10 = min_dist(cd,'cd') 
cosined10 = min_dist(cosined,'cosined') 
canberra10 = min_dist(canberra,'canberrajaccard') 

print ('\n10 points closed to the point P using euclidean distance \n', ed10)
print ('\n10 points closed to the point P using Mahalanobis distance \n', md10)
print ('\n10 points closed to the point P using City block metric is \n', cbm10)
print ('\n10 points closed to the point P using Minkowski metric(for r=3) \n', mkski10)
print ('\n10 points closed to the point P using Chebyshev distance \n', cd10)
print ('\n10 points closed to the point P using Cosine distance \n', cosined10)
print ('\n10 points closed to the point P using canberra distance \n', canberra10)
    
#5(c)-b-a Plot 10 points and P, and connect them
def graph10(dist, P, string):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(data[xlabel], data[ylabel], color = '#00ff00')
    plt.scatter(P[xlabel], P[ylabel], color = 'r', marker = 'x')
    ax.annotate('P', xy=(P[xlabel], P[ylabel]), xytext=(P[xlabel]-0.25, P[ylabel] +0.25),
            )
    plt.scatter(dist[xlabel], dist[ylabel], color = 'b', marker = 'o')
    for index, row in dist.iterrows():
        plt.plot([row[xlabel],P[xlabel]],[row[ylabel],P[ylabel]],'#008000')
    plt.title(string)    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.axis('equal') # set x, y the same scale
    plt.grid()
    plt.show()

graph10(ed10,P,'Euclidean distance')
graph10(md10,P,'Mahalanobis distance')
graph10(cbm10,P,'City block metric')
graph10(mkski10,P,'Minkowski metric(for p=3)')
graph10(cd10,P,'Chebyshev distance')
graph10(cosined10,P,'Cosine distance')
graph10(canberra10,P,'canberra distance')