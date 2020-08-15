import pandas as pd
import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()

# data load
data = pd.read_csv('./data/ozone.csv', sep=',')
#display(data.head())
# 오존량 / 태양의 세기 / 바람의 세기 / 온도 / 월 / 일

# 결측치 제거
data = data.dropna(how='any')
#display(data.head())

# 필요한 컬럼 추출
df = data[["Solar.R", "Wind", "Temp", "Ozone"]]

# Standardization, Normalization
df['Solar.R_Stan'] = (df['Solar.R'] - df['Solar.R'].mean()) / df['Solar.R'].std()
df['Wind_Stan'] = (df['Wind'] - df['Wind'].mean()) / df['Wind'].std()
df['Temp_Stan'] = (df['Temp'] - df['Temp'].mean()) / df['Temp'].std()
df['Ozone_Stan'] = (df['Ozone'] - df['Ozone'].mean()) / df['Ozone'].std()
#display(df.head())

# training data set
x_data = df[['Solar.R_Stan', 'Wind_Stan', 'Temp_Stan']].values
y_data = df[['Ozone_Stan']].values.reshape(-1,1)

# placeholder
X = tf.compat.v1.placeholder(shape=[None,3], dtype=tf.float32)
Y = tf.compat.v1.placeholder(shape=[None,1], dtype=tf.float32)

# Weight & bias
W = tf.Variable(tf.random.normal([3,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# Hypothesis
H = tf.matmul(X,W) + b

# cost function
cost = tf.reduce_mean(tf.square(H-Y))

# train
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Session & 초기화
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# 학습
for step in range(3000):
    _, cost_val = sess.run([train, cost], feed_dict={X:x_data, Y:y_data})
    if step % 300 == 0:
        print("cost : {}".format(cost_val))
    
# prediction
# Solar.R, Wind, Temp
input_data = [[190.0, 7.4, 67.0]]
sess.run(H, feed_dict={X:input_data})
