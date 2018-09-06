import math
import numpy as np
# import pylab as pl
# import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse, Circle
import time
from math import radians, cos, sin, asin, sqrt

# 数据集：每四个一组，分别是编号，经度，纬度和时间戳，读取后变成经度、纬度、时间戳三元组
filename = "/home/zhanghao/projectfile/demo_01/dbscan/data.txt"
arr = []
t1 = time.time()
with open(filename) as f:
    lines = f.readlines()
    for line in lines:
        a1 = line.strip().split(",")
        arr.append((a1[1], a1[2], a1[3]))
# print(arr[0:2])

# 数据处理成（x1, x2, timestamp）列表的形式
dataset = [(float(i[0]), float(i[1]), int(i[2])) for i in arr]
dataset.sort(key=lambda x:x[2])
# print("dataset: ", dataset[0:10])
t2 = time.time()
print("load dataset cost: ", str(t2 - t1))


# 计算欧式距离
def dist(a, b):
    lng1 = a[0]
    lat1 = a[1]
    lng2 = b[0]
    lat2 = b[1]
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon = lng2 -lng1
    dlat = lat2 - lat1
    x = sin(dlat/2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon/2) ** 2
    dis = 2 * asin(sqrt(x)) * 6371*1000
    return dis

def duration(a, b):
    return math.fabs(a[2] - b[2]) / 1000

# 获取满足聚类条件的邻近点
def getNeighbors(i, D, ceps, eps):
    neightbors = []
    neightbor_nums = []
    for j in range(i + 1, len(D)):
        if dist(D[i], D[j]) <= eps:
            neightbors.append(D[j])
            neightbor_nums.append(j)
        elif dist(D[i], D[j]) > ceps:
            break
    return neightbors, neightbor_nums

def DBSCAN(D, ceps, eps, Minpts):
    T_tmp = []
    k = 0
    C = []
    P = set([i for i in range(len(D))])
    neib_dict = {}
    neibnum_dict = {}
    t3 = time.time()
    print("init args cost: ", str(t3 - t2))

    for i in range(len(D)):
        neighbors, neightbor_nums = getNeighbors(i, D, ceps, eps)
        neibnum_dict[i] = neightbor_nums
        neib_dict[i] = neighbors
        if len(neighbors) >= Minpts:
            T_tmp.append(i)
    T = set(T_tmp)
    t4 = time.time()
    print("calculate core dot cost: ", str(t4 - t3))
    # print("neibnum_dict: ", neibnum_dict)
    # print("T: ", T)

    # 开始聚类
    while len(T):
        P_old = P
        t = min(list(T))
        o = [t]

        P = P - set(o)
        Q = []
        Q.append(t)
        while len(Q):
            q = Q[0]
            Nq = neibnum_dict[q]
            if len(Nq) >= Minpts:
                S = P & set(Nq)
                Q += (list(S))
                P = P - S
            Q.remove(q)
        k += 1
        Ck = list(P_old - P)
        ts = T & set(Ck)
        T = T - set(Ck)

        arr = set(Ck)
        for t1 in ts:
            arr = arr | set(neibnum_dict[t1])
        C.append(list(arr))

    t5 = time.time()
    print("calculater cluster cost: ", str(t5 - t4))
    return C


# def draw(C_data, C):
#     colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
#     for i in range(len(C_data)):
#         coo_X = []
#         coo_Y = []
#
#         for j in range(len(C_data[i])):
#             coo_X.append(C_data[i][j][0])
#             coo_Y.append(C_data[i][j][1])
#         plt.scatter(coo_X, coo_Y, marker='x', color=colValue[i % len(colValue)], label=i)
#
#         n = np.asanyarray(C[i])
#         for j, txt in enumerate(n):
#             plt.annotate(txt, [coo_X[j], coo_Y[j]])
#     plt.legend(loc='upper right')
#     # plt.xlim(0.2, 0.8)
#     # plt.ylim(0, 0.5)
#     plt.show()
#
#
# def draw_dataset(dataset):
#     # colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
#     coo_X = []
#     coo_Y = []
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#
#     for i in range(len(dataset)):
#         coo_X.append(dataset[i][0])
#         coo_Y.append(dataset[i][1])
#         # x = np.arange(dataset[i][0] - 0.15, dataset[i][0] + 0.15, 0.01)
#         # y = dataset[i][1] + np.sqrt(0.15**2 - (x-dataset[i][0])**2)
#         # plt.plot(x, y)
#         # if i>=14 and i<21:
#         #     x0 = coo_X[i]
#         #     y0 = coo_Y[i]
#         #     r = 0.15
#         #     theta = np.arange(0, 2 * np.pi, 0.01)
#         #     x = x0 + r * np.cos(theta)
#         #     y = y0 + r * np.sin(theta)
#         #     plt.plot(x, y)
#         #     plt.axis("equal")
#         #
#     plt.scatter(coo_X, coo_Y, marker='x')
#     n = range(len(coo_X))
#     for j, txt in enumerate(n):
#         plt.annotate(txt, [coo_X[j], coo_Y[j]])
#     # pl.legend(loc='upper right')
#     # pl.show()
#     plt.plot(coo_X, coo_Y, "b--")
#     # plt.xlim(0.2, 0.8)
#     # plt.ylim(0, 0.5)
#     plt.show()

ceps = 10000
eps = 2000
minpts = 5

C = DBSCAN(dataset, ceps, eps, minpts)
print("C.length: ", len(C))
print("C: ", C)

# draw_dataset(dataset)
# C_data = []
# for arr in C:
#     c_tmp = []
#     for i in arr:
#         c_tmp.append(dataset[i])
#     C_data.append(c_tmp)
# print('C_data: ', C_data)
# draw(C_data, C)
