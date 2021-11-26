from gwe_fishdata import FishData
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import numpy as np

fdata = FishData()
print(fdata.species)
bsdata = fdata.getSpecies('Bream', 'Parkki', 'Smelt')
print(fdata.features)
bsresult, bsdata = fdata.getFeatures(bsdata, 'Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width')

print(bsresult.shape, bsdata.shape)

ss =StandardScaler()
ss.fit(bsdata)
scaled = ss.transform(bsdata)

train, test, target, testtarget = train_test_split(scaled, bsresult)


#sc = SGDClassifier(loss='log', tol=None)
sc = SGDClassifier(loss='log', alpha=0.01, max_iter=100)
#sc = SGDClassifier(loss='log', max_iter=100, tol=None)
trainscore = []
testscore = []
for i in range(300):
    sc.partial_fit(train, target, classes=fdata.species)
    trainscore.append(sc.score(train, target))
    testscore.append(sc.score(test, testtarget))

plt.plot(trainscore)
plt.plot(testscore)
plt.show()

#sc = SGDClassifier(loss='log', max_iter=100)
sc = SGDClassifier(loss='log', alpha=0.01, tol=0.1, max_iter=100)
sc.fit(train, target)
print(sc.score(train, target))
print(sc.score(test, testtarget))
