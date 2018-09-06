import numpy as np

# rmd = np.random.RandomState(1)
# dataset_size = 2
# X = rmd.rand(dataset_size, 2)
# print(X)

X = [[1, 2], [3,4]]
Y = [[x1 + x2] for (x1, x2) in X]
print(Y)