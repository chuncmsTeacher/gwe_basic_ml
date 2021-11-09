from gwe_fishdata import FishData
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

fdata = FishData()
bs = fdata.getSpecies('Bream', 'Smelt')
result, data = fdata.getFeatures(bs, 'Weight', 'Length2')

mean = np.mean(data, axis=0)
std = np.std(data, axis=0)

data = (data-mean)/std

print(mean, std)
myfish = np.array([[160, 24]])
#myfish = np.array([[200, 24],[160,24]])
myfish = (myfish-mean)/std

nc = KNeighborsClassifier()
nc.fit(data, result)
print(nc.score(data, result))
print(nc.predict(myfish))

print(nc.classes_)
dist, ind = nc.kneighbors(myfish)

