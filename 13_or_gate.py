# (4)2-1/C(2)
import tensorflow as tf

x_data = [[0., 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [1]]

#------- a neuron
W = tf.Variable(tf.random_normal([2, 1]))
b = tf.Variable(tf.random_normal([1]))
hypo = tf.sigmoid(tf.matmul(x_data, W) + b)

#----- learning
cost = -tf.reduce_mean(y_data * tf.log(hypo) + tf.subtract(1., y_data) * tf.log(tf.subtract(1., hypo)))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_list = []
for step in range(10001):
    sess.run(train)

    if step % 500 == 0:
        print(step, sess.run(cost))
        cost_list.append(sess.run(cost))

# Show the error
import matplotlib.pyplot as plt
plt.plot(cost_list)
plt.show()

#----- testing(classification)
predicted = tf.cast(hypo > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y_data), dtype=tf.float32))

h = sess.run(hypo)
print("\nHypo: ", h)

p = sess.run(predicted)
print("Predicted: ", p)

a = sess.run(accuracy)
print("Accuracy(%): ", a * 100)


