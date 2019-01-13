# (4)2b-4S/C(4)
import tensorflow as tf

x_data = [[0., 0], [0, 1], [1, 0], [1, 1]]
y_data = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]

X = tf.placeholder(tf.float32, [4, 2])
Y = tf.placeholder(tf.float32, [4, 4])

#------- 2 inputs 4 neurons
W = tf.Variable(tf.random_normal([2, 4]))
b = tf.Variable(tf.random_normal([4]))
output = tf.matmul(X, W) + b

#----- learning
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=output, labels=Y))
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_list = []
for step in range(1001):
    if step % 100 == 0:
        cost_val = sess.run(cost, feed_dict={X: x_data, Y: y_data})
        print(step, cost_val)
        cost_list.append(cost_val)
    sess.run(train, feed_dict={X:x_data, Y:y_data})

# Show the error
import matplotlib.pyplot as plt
plt.plot(cost_list)
plt.show()

#----- testing(classification)
predicted = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(predicted, tf.float32))

h = sess.run(output, feed_dict={X:x_data})
print("\nLogits: ", h)

p = sess.run(predicted, feed_dict={X:x_data, Y:y_data})
print("Predicted: ", p)

a = sess.run(accuracy, feed_dict={X:x_data, Y:y_data})
print("Accuracy(%): ", a * 100)
