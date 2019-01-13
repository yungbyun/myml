# (1)1-1/R
import tensorflow as tf

x_data = [1]
y_data = [1]

# ----- a neuron
w = tf.Variable(tf.random_normal([1]))
hypo = x_data * w

# ----- learning
cost = (hypo - y_data) ** 2
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_list = []
for i in range(1001):
    if i % 100 == 0:
        err_val = sess.run(cost)
        print('w:', sess.run(w), 'cost:', err_val)
        cost_list.append(err_val)
    sess.run(train)

# show an error graph
import matplotlib.pyplot as plt
plt.plot(cost_list)
plt.xlabel('Epoch')
plt.ylabel('Square Error')
plt.show();

# ----- test(prediction)
print(sess.run(w * [3, 4, 6, 9]))

x_data = [3, 4, 6, 9]
print(sess.run(hypo))

