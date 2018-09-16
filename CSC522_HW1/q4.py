import random
import math
import sys
import os


def pprint(mat):
	pad = '{:>' + str(len(str(max([max(mat[i]) for i in range(len(mat))])))) + '}'
	print('[' + '\n '.join(['[' + ', '.join([pad.format(mat[i][j]) for j in range(len(mat[0]))]) + ']' for i in range(len(mat))]) + ']')


def identity_matrix(dim):
	return [[1 if j==i else 0 for j in range(dim)] for i in range(dim)]


def column_manipulate(mat, col, val):
	for row in mat:
		row[col] = val
	return mat


def matrix_sum(mat):
	cnt = 0
	for row in mat:
		for col in row:
			cnt += col
	return cnt


def transpose(mat):
	result = [[None for j in range(len(mat))] for i in range(len(mat[0]))]
	for i in range(len(mat)):
		for j in range(len(mat[i])):
			result[j][i] = mat[i][j]
	return result


def row_diagonal_sum(mat, row):
	cnt=0
	for i in range(len(mat)):
		for j in range(len(mat[i])):
			if i==j or i==row:
				cnt += mat[i][j]
	return cnt


def gaussian_matrix(dim, mean, variance, dtype):
	if dtype=="int":
		return [[math.floor(random.gauss(mean, math.sqrt(variance))) for j in range(dim)] for i in range(dim)]
	else:
		return [[random.gauss(mean, math.sqrt(variance)) for j in range(dim)] for i in range(dim)]


def matrix_multiply(a, b):
	result = [[0 for j in range(len(b[0]))] for i in range(len(a))]
	for i in range(len(a)):
		for j in range(len(b[0])):
			for k in range(len(b)):
				result[i][j] += a[i][k] * b[k][j]
	return result


def multiply(a, b):
	result = [[0 for j in range(len(a[0]))] for i in range(len(a))]
	for i in range(len(a)):
		for j in range(len(a[0])):
			result[i][j] = a[i][j] * b[i][j]
	return result


def add(a, b):
	result = [[0 for j in range(len(a[0]))] for i in range(len(a))]
	for i in range(len(a)):
		for j in range(len(a[0])):
			result[i][j] = a[i][j] + b[i][j]
	return result


def row_shift(mat):
	return mat[1:] + [mat[0]]


def covariance(x, y):
	mean_x = sum(x)/len(x)
	mean_y = sum(y)/len(y)
	sumo = 0.0
	for i in range(len(x)):
		sumo += (x[i] - mean_x) * (y[i] - mean_y)
	sumo /= (len(x) - 1)
	return sumo


def main():
	print('Q4 - (a)')
	A = identity_matrix(5)
	pprint(A)

	print('\nQ4 - (b)')
	A = column_manipulate(A, 1, 3)
	pprint(A)

	print('\nQ4 - (c)')
	print(matrix_sum(A))

	print('\nQ4 - (d)')
	A = transpose(A)
	pprint(A)

	print('\nQ4 - (e)')
	print(row_diagonal_sum(A, 2))

	print('\nQ4 - (f)')
	B = gaussian_matrix(5, 5, 3, "int")
	pprint(B)

	print('\nQ4 - (g)')
	C1 = multiply(matrix_multiply([[1,0,0,0,0]]*5, row_shift(B)), identity_matrix(5))
	C2 = matrix_multiply(matrix_multiply([[1,0,0,0,0], [0,0,0,0,0]], B), C1)
	C3 = matrix_multiply([[0,0,0,0,0], [0,0,1,1,-1]], B)
	C = add(C2, C3)
	pprint(C)

	print('\nQ4 - (h)')
	D1 = multiply([[2,3,4,5,6]]*5, identity_matrix(5))
	D = matrix_multiply(C, D1)
	pprint(D)

	print('\nQ4 - (g)')
	print('X\tY\tZ')
	print('-\t-\t-')
	print('{}\t{}\t{}'.format(2, 6, 1))
	print('{}\t{}\t{}'.format(4, 5, 3))
	print('{}\t{}\t{}'.format(6, 4, 5))
	print('{}\t{}\t{}'.format(8, 3, 7))
	X = [2, 4, 6, 8]
	Y = [6, 5, 4, 3]
	Z = [1, 3, 5, 7]
	print('\nCovariance Matrix')
	print('\tX\tY\tZ')
	print('X\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(covariance(X, X), covariance(X, Y), covariance(X, Z)))
	print('Y\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(covariance(Y, X), covariance(Y, Y), covariance(Y, Z)))
	print('Z\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(covariance(Z, X), covariance(Z, Y), covariance(Z, Z)))




if __name__=='__main__':
	main()
