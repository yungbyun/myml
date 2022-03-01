# (1)1-1/R
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [1]
y_data = [1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#----- a neuron
w = tf.Variable(tf.random_normal([1]))
hypo = w * X

#----- learning
cost = (hypo  - Y) ** 2
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_list = []
print(sess.run(w), sess.run(cost, feed_dict={X:x_data, Y:y_data}))
for i in range(1001):
    if i % 100 == 0:
        cost_val = sess.run(cost, feed_dict={X:x_data, Y:y_data})
        print(sess.run(w), cost_val)
        cost_list.append(cost_val)
    sess.run(train, feed_dict={X:x_data, Y:y_data})

# Show the error
import matplotlib.pyplot as plt
plt.plot(cost_list)
plt.show()

#----- testing(prediction)
print(sess.run(hypo, feed_dict={X:[3, 5, 7, 8, 10]}))
