from gwe_fishdata import FishData
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import numpy as np

fdata = FishData()
bs = fdata.getSpecies('Bream', 'Smelt')
result, data = fdata.getFeatures(bs, 'Weight', 'Length2')

myfish = np.array([[160, 24]])
#myfish = np.array([[200, 24],[160,24]])

ss = StandardScaler()
ss.fit(data)

data = ss.transform(data)
myfish = ss.transform(myfish)

nc = KNeighborsClassifier()
nc.fit(data, result)
print(nc.score(data, result))
print(nc.predict(myfish))

print(nc.classes_)
dist, ind = nc.kneighbors(myfish)

