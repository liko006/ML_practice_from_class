# 파일명 : bmi-create.py

import random

# bmi를 계산해서 레이블을 리턴하는 함수
def calc_bmi(h, w):
    bmi = w / ((h/100)**2)
    if bmi < 18.5: return "thin"
    if bmi < 25: return "normal"
    return "fat"

# 출력파일
fp = open('./0801/bmi.csv', 'w', encoding='utf-8')
fp.write('height,weight,result\r\n')

# 랜덤함수로 무작위 데이터 생성하기 : 2만명 정보
cnt = {"thin":0, "normal":0, "fat":0}

for i in range(20000):
    h = random.randint(120, 210)
    w = random.randint(35, 120)
    result = calc_bmi(h, w)
    cnt[result] += 1
    fp.write("{0},{1},{2}\r\n".format(h, w, result))
    
fp.close()

print('OK', cnt)

# BMI 지수 테스트 (SVM 이용)
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# 키와 몸무게 데이터 읽어들이기
bmi_data = pd.read_csv('./0801/bmi.csv', encoding='utf-8')

# 칼럼을 자르고 정규화하기
label = bmi_data['result']
w = bmi_data['weight'] / 120
h = bmi_data['height'] / 210
wh = pd.concat([w,h], axis=1)

# 학습전용 테스트 전용으로 구분하기
train_data, test_data, train_label, test_label = train_test_split(wh, label, test_size=0.2, train_size=0.8, shuffle=True)

clf = svm.SVC()
clf.fit(train_data, train_label)

pre = clf.predict(test_data)
score = metrics.accuracy_score(test_label, pre)
report = metrics.classification_report(test_label, pre)

print("정확도: ", score)
print("리포트 = \n", report)

# 비만도(BMI) 그래프
import matplotlib.pyplot as plt
import pandas as pd

tbl = pd.read_csv('./0801/bmi.csv', index_col=2)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

def scatter(lbl, color):
    b = tbl.loc[lbl]
    ax.scatter(b['weight'], b['height'], c=color, label=lbl)
    
scatter('fat', 'red')
scatter('normal', 'green')
scatter('thin', 'blue')
ax.legend()
plt.savefig('bmi-test.png')
plt.show()

