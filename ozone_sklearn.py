from sklearn import linear_model
import numpy as np
import pandas as pd

df = pd.read_csv('./data/ozone.csv', sep=',')
df = df.dropna(how='any')

X = df[['Solar.R','Wind','Temp']]
y = df['Ozone']

lm = linear_model.LinearRegression()
lm.fit(X,y)
print('W의 값: {}'.format(lm.coef_))
print('intercept : {}'.format(lm.intercept_))
print()

predictions = lm.predict([[190.0, 7.4, 67.0]])
print('예측값: {}'.format(predictions))
