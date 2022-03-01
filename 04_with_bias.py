# (1)1-1/R
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [1]
y_data = [1]

#----- a neuron
w = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))
hypo = w * x_data + b

#----- learning
cost = (hypo - y_data) ** 2
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_list = []
for i in range(1001):
    if i % 50 == 0:
        error_val = sess.run(cost)
        print(sess.run(w), sess.run(b), error_val)
        cost_list.append(sess.run(cost))
    sess.run(train)

# Show the error
import matplotlib.pyplot as plt
plt.plot(cost_list)
plt.show();

#----- testing(prediction)
print(sess.run(w * [3]))
