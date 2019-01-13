# (4)1-1/C(2)
import tensorflow as tf

x_data = [[-2.], [-1], [1], [2]]
y_data = [[0.], [0], [1], [1]]

#------- a neuron
w = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))
hypo = tf.sigmoid(x_data * w + b)

#----- learning
cost = -tf.reduce_mean(y_data * tf.log(hypo) + tf.subtract(1., y_data) * tf.log(tf.subtract(1., hypo)))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_list = []
print(sess.run(w), sess.run(b), sess.run(cost))
for step in range(5001):
    sess.run(train)

    if step % 200 == 0:
        cost_val = sess.run(cost)
        print(sess.run(w), cost_val)
        cost_list.append(cost_val)

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
