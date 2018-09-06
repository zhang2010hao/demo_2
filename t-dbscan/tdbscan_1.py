import math
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

# 数据集：每三个一组，分别是编号，和两维度数据
data = """1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,6,0.403,0.237,7,0.481,0.149,8,0.437,0.211,9,0.666,0.091,10,0.243,0.267,11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,0.37,16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,21,0.748,0.232,22,0.714,0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,0.445,30,0.446,0.459"""
a = data.split(",")
# print("a: ", a)

# 数据处理成（x1, x2）列表的形式
dataset = [(float(a[i]), float(a[i + 1])) for i in range(1, len(a) - 1, 3)]
print("dataset: ", dataset)


# 计算欧式距离
def dist(a, b):
    return math.sqrt((math.pow((a[0] - b[0]), 2) + math.pow((a[1] - b[1]), 2)))


# 获取满足聚类条件的邻近点
def getNeighbors(d, D, e):
    neightbors = [i for i in D if dist(d, i) <= e]
    return neightbors


# 获取满足聚类条件的邻近点
def getNeighbors_1(i, D, ceps, eps):
    neightbors = []
    neightbor_nums = []
    for j in range(i + 1, len(D)):
        if dist(D[i], D[j]) <= eps:
            neightbors.append(D[j])
            neightbor_nums.append(j)
        elif dist(D[i], D[j]) > ceps:
            break
    return neightbors, neightbor_nums


# 扩展聚类
# def expandCluster():


def DBSCAN(D, ceps, eps, Minpts):
    # T = set() # 核心对象集合
    # k = 0     # 聚类个数
    # C = []    # 聚类集合
    # P = set(D)#未访问集合
    #
    # for d in D:
    #     # 计算核心对象并加入到T
    #     if len(getNeighbors(d, D, e)) >= Minpts:
    #         T.add(d)
    # # 开始聚类
    # while len(T):
    #     P_old = P
    #     o = list(T)[np.random.randint(0, len(T))]
    #     P = P - set(o)
    #     Q = []
    #     Q.append(o)
    #     while len(Q):
    #         q = Q[0]
    #         Nq = getNeighbors(q, D, e)
    #         if len(Nq) >= Minpts:
    #             S = P & set(Nq)
    #             Q += (list(S))
    #             P = P - S
    #         Q.remove(q)
    #     k += 1
    #     Ck = list(P_old - P)
    #     T = T - set(Ck)
    #     C.append(Ck)

    T_1 = set()
    k_1 = 0
    C_1 = []
    P_1 = set([i for i in range(len(D))])
    neib_dict = {}
    neibnum_dict = {}

    for i in range(len(D)):
        neighbors, neightbor_nums = getNeighbors_1(i, D, ceps, eps)
        neibnum_dict[i] = neightbor_nums
        neib_dict[i] = neighbors
        if len(neighbors) >= Minpts:
            T_1.add(i)
    print("neibnum_dict: ", neibnum_dict)
    print("T_1: ", T_1)
    # 开始聚类
    while len(T_1):
        P_old = P_1
        t = list(T_1)[0]
        o = [t]

        P_1 = P_1 - set(o)
        Q = []
        Q.append(t)
        while len(Q):
            q = Q[0]
            Nq = neibnum_dict[q]
            if len(Nq) >= Minpts:
                S = P_1 & set(Nq)
                Q += (list(S))
                P_1 = P_1 - S
            Q.remove(q)
        k_1 += 1
        Ck = list(P_old - P_1)
        T_1 = T_1 - set(Ck)
        C_1.append(Ck)
    return C_1


def draw(C):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(len(C)):
        coo_X = []
        coo_Y = []

        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        pl.scatter(coo_X, coo_Y, marker='x', color=colValue[i % len(colValue)], label=i)
    pl.legend(loc='upper right')
    pl.show()


def draw_dataset(dataset):
    # colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    coo_X = []
    coo_Y = []

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(dataset)):
        coo_X.append(dataset[i][0])
        coo_Y.append(dataset[i][1])
        # x = np.arange(dataset[i][0] - 0.15, dataset[i][0] + 0.15, 0.01)
        # y = dataset[i][1] + np.sqrt(0.15**2 - (x-dataset[i][0])**2)
        # plt.plot(x, y)
        if i>=14 and i<21:
            x0 = coo_X[i]
            y0 = coo_Y[i]
            r = 0.15
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = x0 + r * np.cos(theta)
            y = y0 + r * np.sin(theta)
            plt.plot(x, y)
            plt.axis("equal")

        plt.scatter(coo_X, coo_Y, marker='x')
        n = range(len(coo_X))
        for j, txt in enumerate(n):
            plt.annotate(txt, [coo_X[j], coo_Y[j]])
    # pl.legend(loc='upper right')
    # pl.show()
    plt.plot(coo_X, coo_Y, "b--")

    plt.show()

C = DBSCAN(dataset, 0.4, 0.15, 4)
# draw_dataset(dataset)
print("C: ", C)
# draw(C)
