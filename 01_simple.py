#%tensorflow_version 1.x
# (1)1-1/R
import tensorflow as tf

x = [1]
y = [1]

w = tf.Variable (tf.random_normal([1]))
h = w * x

E = (h - y) ** 2

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(E)
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #𝑤값 초기화

err_list = []
for i in range(101):
    w_val = sess.run(w)
    err = sess.run(E)
    print(i, 'w:', w_val, 'cost:', err)
    err_list.append(err)
    sess.run(train) #한번 경사하강(w 업데이트)

# ----- test(prediction)
print(sess.run(w * [3, 4, 6, 9]))

# show an error graph
import matplotlib.pyplot as plt
plt.plot(err_list)
plt.xlabel('Epoch')
plt.ylabel('Squared Error')
plt.show()

