import numpy as np

A = np.identity(5)
print A
A[:, 1] = 3
print A


sum = 0
for (x,y), value in np.ndenumerate(A):
    sum += value
print sum


A = A.T
print A

print np.sum(A[2])
print np.sum(np.diagonal(A))

B = np.random.normal(5, 3, (5, 5))
print B

C = np.matrix([B[1]*B[0], B[2]+B[3]-B[4]])
print C

t = np.array([2,3,4,5,6])
D = np.multiply(C, t)
print D


X = np.array([[2, 4, 6, 8], [6, 5, 4, 3], [1, 3, 5, 7]])
print np.cov(X)


x = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
print sum([i*i for i in x]) / len(x)

st = np.std(x)
mean = np.mean(x)
print st*st + mean*mean