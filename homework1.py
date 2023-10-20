import numpy as np

def Z_R(matrix, length): # matrix为一维矩阵Z，length为R矩阵的列数
    num = len(matrix) - length + 1
    R = np.empty((0, length), int)
    for i in range(num):
        R = np.append(R, np.array([matrix[i: (i + length)]]), axis=0)
    return R
def freq(arr): # 查找数组中最频繁的值。
    length = len(arr)
    max = arr[0]
    for i in arr: # 求得数组中的最大值
        if i > max:
            max = i
    temp = np.zeros(max + 1)
    for i in arr: # 统计数组中各个数值的个数
        temp[i] = temp[i] + 1
    value = np.zeros(1)
    for i in range(max + 1):
        if temp[i] > temp[int(value[0])]:
            value = np.array([i])
        elif temp[i] == temp[int(value[0])]:
            value = np.append(value, np.array([i]), axis=0)
    return value
def cal_block(matrix): # 获得矩阵块的和
    sum = np.zeros((4, 4), int)
    for i in range(4):
        for j in range(4):
            sum = sum + matrix[i * 4:(i + 1) * 4,j * 4: (j + 1) * 4]
    return sum
def diff_row(matrix): # 输出没有相同元素的所有的行
    row, rol = matrix.shape
    M = np.empty((0, rol), int)
    for i in range(row):
        temp = matrix[i, :]
        max = temp[0]
        judge = 1 # 当判断到有重复数字后，judge值变为0
        for j in temp:  # 求得数组中的最大值
            if j > max:
                max = j
        num = np.zeros(max + 1)
        for n in temp:
            if num[n] == 0:
                num[n] = 1
            elif num[n] == 1:
                judge = 0
                break
        if judge == 0:
            continue
        elif judge == 1:
            M = np.append(M, np.array([temp]), axis=0)
    return M

def my_inner(A, B):
    return np.einsum('i,i', A, B)
def my_outer(A, B):
    return np.einsum('i,j->ij', A, B)
def my_sum(A, B):
    return A + B
def my_mul(A, B):
    return np.einsum('i,i->i', A, B)


'''
Z = np.arange(1, 15)
print(Z_R(Z, 4))
arr = np.array([1, 2, 3, 4, 0, 5, 9, 1, 2, 4, 4, 5, 5])
print(freq(arr))
matrix = np.ones((16, 16), int)
print(cal_block(matrix))
a = np.array([[1, 2, 3], [2, 2, 3], [3, 2, 4], [4, 1, 4]], int)
print(diff_row(a))
A = np.array([2, 3, 4])
B = np.array([1, 1, 2])
print(my_outer(A, B))
print(my_inner(A, B))
print(my_sum(A, B))
print(my_mul(A, B))
'''

