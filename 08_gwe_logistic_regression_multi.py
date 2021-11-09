from gwe_fishdata import FishData
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import numpy as np

fdata = FishData()
bsdata = fdata.getSpecies('Bream', 'Smelt', 'Pike', 'Whitefish')
print(fdata.features)
bsresult, bsdata = fdata.getFeatures(bsdata, 'Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width')

print(bsresult.shape, bsdata.shape)

ss =StandardScaler()
ss.fit(bsdata)
scaled = ss.transform(bsdata)

train, test, target, testtarget = train_test_split(scaled, bsresult)

lgr = LogisticRegression()
lgr.fit(train, target)
print(lgr.coef_, lgr.intercept_)

print(lgr.score(train, target))
print(lgr.score(test, testtarget))
print(lgr.predict(test[:2]))
print(lgr.predict_proba(test[:2]))
