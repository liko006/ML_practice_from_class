# 주어진 데이터로 7시간 공부하면 몇점을 받을지 예측
import tensorflow as tf

# training data set
x_data = [1,2,5,8,10]
y_data = [5,15,68,80,95]

# placeholder 설정 (이 값을 설정하지 않으면 고정값만 사용해야한다)
x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

H = (W * x) + b  

cost = tf.reduce_mean(tf.square(H-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(3000):
    _, cost_val = sess.run([train, cost], feed_dict={x:x_data, y:y_data})
    if step % 300 == 0:
        print('cost : {}'.format(cost_val))

# 예측
print("예측값: ", sess.run(H, feed_dict={x:7}))
