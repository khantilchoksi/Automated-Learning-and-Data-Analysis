from pprint import pprint
import random
import math
import sys
import os


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
	for i in range(len(mat)):
		for j in range(i+1, len(mat[i])):
			mat[i][j], mat[j][i] = mat[j][i], mat[i][j]
	return mat


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


def multiply(a, b):
	result = [[0 for j in range(len(b[0]))] for i in range(len(a))]
	for i in range(len(a)):
		for j in range(len(b[0])):
			for k in range(len(b)):
				result[i][j] += a[i][k] * b[k][j]
	return result


def row_shift(mat):
	return mat[1:] + [mat[0]]


def main():
	A = identity_matrix(5)
	A = column_manipulate(A, 1, 3)
	matrix_sum(A)
	A = transpose(A)
	row_diagonal_sum(A, 2)
	B = gaussian_matrix(5, 5, 3, "int")
	pprint(B)
	C1 = multiply([[1,0,0,0,0]], row_shift(B))
	pprint(C1)


if __name__=='__main__':
	main()
