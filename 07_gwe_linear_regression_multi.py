from gwe_fishdata import FishData
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import numpy as np

fdata = FishData()
#print(fdata.features)
perch = fdata.getSpecies('Perch')
pr, pd = fdata.getFeatures(perch, 'Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width')
ss = StandardScaler()
ss.fit(pd)
scaledpd = ss.transform(pd)

target = pd[:,0]
train = scaledpd[:,1:]
print(train.shape, target.shape)
strain, stest, starget, stesttarget = train_test_split(train, target)
print(strain.shape, starget.shape)
print(stest.shape, stesttarget.shape)

lr = LinearRegression()
lr.fit(strain, starget)

print(lr.score(strain, starget))
print(lr.score(stest, stesttarget))
print(lr.predict(stest[:1]))

print(lr.coef_, lr.intercept_)


