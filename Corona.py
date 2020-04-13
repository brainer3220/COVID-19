import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py

import re

import tensorflow as tf
import numpy as np

import plotly.express as px

import matplotlib.pyplot as plt   # 그래프를 그리기 위해 사용됩니다.

url = 'https://raw.githubusercontent.com/jihoo-kim/Data-Science-for-COVID-19/master/dataset/Patient/PatientInfo.csv'
Korea_patient = pd.read_csv(url, error_bad_lines=False)

url = 'https://raw.githubusercontent.com/jihoo-kim/Data-Science-for-COVID-19/master/dataset/SearchTrend.csv'
Korea_Srch_Trend = pd.read_csv(url, error_bad_lines=False)

url = 'https://raw.githubusercontent.com/jihoo-kim/Data-Science-for-COVID-19/master/dataset/Time/Time.csv'
Korea_Time_Test = pd.read_csv(url, error_bad_lines=False)

url = 'https://raw.githubusercontent.com/jihoo-kim/Data-Science-for-COVID-19/master/dataset/Case.csv'
Korea_Case = pd.read_csv(url, error_bad_lines=False)

tmp = Korea_Time_Test

x_data = np.asarray(tmp.test)
x_data = x_data.reshape(-1)

y_data = np.asarray(tmp.confirmed)
y_data = y_data.reshape(-1)


learning_rate = 0.001
training_epochs = 1000
display_step = 100

num_of_samples = x_data.shape[0]

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

W = tf.Variable(np.random.randn(), name="Weight")
B = tf.Variable(np.random.randn(), name="Bias")

with tf.name_scope('Pred') as scope:
    pred = X + W * B * X
    tf.summary.histogram("hypothesis", pred)

with tf.name_scope('Input') as scope:
    cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*num_of_samples)
    cost_summ = tf.summary.scalar("cost", cost)

with tf.name_scope('Optimizer') as scope:
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./Model", sess.graph)

writer.add_graph(sess.graph)  # Show the graph


for epoch in range(training_epochs):
    for (x, y) in zip(x_data, y_data):   # zip에 있는 리스트들을 원소로하는 튜플을 생성하여
        summary, _ = sess.run([merged, optimizer], feed_dict={X: x, Y: y})
        writer.add_summary(summary, epoch)

    if (epoch+1) % display_step == 0:
        c = sess.run(cost, feed_dict={X: x_data, Y: y_data})
        print("Epoch:", '%d' % (epoch+1), "cost=", "{:.1f}".format(c),
              "W=", sess.run(W), "b=", sess.run(B))
#         writer.add_summary(c, epoch)

print("최적화가 완료되었습니다.")
training_cost = sess.run(cost, feed_dict={X: x_data, Y: y_data})
print("훈련이 끝난 후 비용과 모델 파라미터입니다.  cost=", training_cost,
      "W=", sess.run(W), "B=", sess.run(B), '\n')

fig = go.Figure()
fig.add_trace(go.Scatter(x=tmp.date, y=tmp.confirmed,
                         mode='markers', name='실제 Data'))
fig.add_trace(go.Scatter(x=tmp.date, y=(x_data + sess.run(W) * sess.run(B) * x_data)+x_data,
                         mode='lines',
                         name='예측 Data'))
