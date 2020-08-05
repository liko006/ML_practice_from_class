# Analyzing iris data

import urllib.request as req
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
savefile = 'd:/weekend/data/iris3.csv'
req.urlretrieve(url, savefile)
print("저장되었습니다")

csv = pd.read_csv(savefile, encoding='utf-8')

import pandas as pd
from sklearn.model_selection import train_test_split  # 원하는 portion으로 데이터를 나눠주는 모듈
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris_data = pd.read_csv('./data/iris.csv', encoding='utf-8')

label = iris_data.loc[:, 'Name']
data = iris_data.loc[:, ['SepalLength','SepalWidth','PetalLength','PetalWidth']]

# 학습전용 & 테스트전용으로 분류하기
train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, train_size=0.8, shuffle=True)

clf = svm.SVC()
clf.fit(train_data, train_label)

pre = clf.predict(test_data)
score = metrics.accuracy_score(test_label, pre)
print("예측결과: ", pre)
print("정확도: ", score)
