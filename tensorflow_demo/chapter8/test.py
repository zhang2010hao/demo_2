import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

# np.random.seed(1000)
# y = np.random.standard_normal(10)
# print("y = %s"% y)
# x = range(len(y))
# print("x=%s"% x)
# y = np.array([1, 2,3,4]).squeeze()
# y_1 = np.array([4,5,6,8]).squeeze()
# plt.plot(y)
# plt.plot(y_1)
# plt.show()
data = [i for i in range(22)]
x = []
y = []
BATCH_SIZE = 7

for i in range(len(data) - 2):
    x.append([data[i:i+2]])
    y.append([data[i+2]])

print("x: ", x)
print("y: ", y)
with tf.Session() as sess:
    ds = tf.data.Dataset.from_tensor_slices((x, y))


    # print(sess.run(d))
    # for ex, ey in ds.make_one_shot_iterator():
    #     print(sess.run(ex))
    #     print(sess.run(ey))

    ds = ds.repeat()

    ds = ds.shuffle(10000)


    ds = ds.batch(BATCH_SIZE)
    d = ds.make_one_shot_iterator()
    while True:
        t = d.get_next()
        print(sess.run(t))

