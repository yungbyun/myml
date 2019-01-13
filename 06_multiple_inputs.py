# (3)2-1/R
import tensorflow as tf

from myplot import MyPlot

x = [[1., 1], [2, 2], [3, 3], [4, 4]]
y = [[1.], [2], [3], [4]]

#----- a neuron
w = tf.Variable(tf.random_normal([2, 1]))
b = tf.Variable(tf.random_normal([1]))
hypo = tf.matmul(x,  w)

#----- learning
cost = tf.reduce_mean((hypo - y) * (hypo - y))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_list = []
print(sess.run(w), sess.run(b), sess.run(cost))
for i in range(2001):
    sess.run(train)

    if i % 50 == 0:
        print("---")
        print(sess.run(w), sess.run(b), sess.run(cost))
        cost_list.append(sess.run(cost))

# Show the error
import matplotlib.pyplot as plt
plt.plot(cost_list)
plt.show();

#----- testing(prediction)
hypo2 = tf.matmul([[4.,4]], w) + b
print("Answer: ", sess.run(hypo2))

