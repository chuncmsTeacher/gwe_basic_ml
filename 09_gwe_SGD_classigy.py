from gwe_fishdata import FishData
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import numpy as np

fdata = FishData()
print(fdata.species)
bsdata = fdata.getSpecies('Bream', 'Parkki', 'Perch', 'Pike', 'Roach', 'Smelt', 'Whitefish')
print(fdata.features)
bsresult, bsdata = fdata.getFeatures(bsdata, 'Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width')

print(bsresult.shape, bsdata.shape)

ss =StandardScaler()
ss.fit(bsdata)
scaled = ss.transform(bsdata)

train, test, target, testtarget = train_test_split(scaled, bsresult)


sc = SGDClassifier(loss='log', max_iter=100, tol=None)

trainscore = []
testscore = []
for i in range(300):
    sc.partial_fit(train, target, classes=fdata.species)
    trainscore.append(sc.score(train, target))
    testscore.append(sc.score(test, testtarget))

plt.plot(trainscore)
plt.plot(testscore)
plt.show()

'''
lgr = LogisticRegression()
lgr.fit(train, target)
print(lgr.coef_, lgr.intercept_)

print(lgr.score(train, target))
print(lgr.score(test, testtarget))
print(lgr.classes_)
print(lgr.predict(test[:2]))
print(lgr.predict_proba(test[:2]).round(3))
'''
