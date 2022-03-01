# (4)2b-4S/C(4)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [[0., 0], [0, 1], [1, 0], [1, 1]]
y_data = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]

#------- 2 inputs 4 neurons
W = tf.Variable(tf.random_normal([2, 4]))
b = tf.Variable(tf.random_normal([4]))
output = tf.matmul(x_data, W) + b  # logit (?, 4)

#----- learning
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=output, labels=y_data))
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_list = []
for step in range(1001):
    if step % 100 == 0:
        print(step, sess.run(cost))
        cost_list.append(sess.run(cost))
    sess.run(train)

# Show the error
import matplotlib.pyplot as plt
plt.plot(cost_list)
plt.show()

#----- testing(classification)
logit = sess.run(output) #(?, 4)
print("\nLogits: ", logit)

hit = tf.equal(tf.argmax(output, 1), tf.argmax(y_data, 1))
accuracy = tf.reduce_mean(tf.cast(hit, tf.float32))

p = sess.run(hit)
print("\nhit: ", p)

a = sess.run(accuracy)
print("Accuracy(%): ", a * 100)


