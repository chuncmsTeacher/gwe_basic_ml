from gwe_fishdata import FishData
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

fdata = FishData()
bream = fdata.getSpecies('Bream')
smelt = fdata.getSpecies('Smelt')
br, bd = fdata.getFeatures(bream, 'Weight', 'Length2')
sr, sd = fdata.getFeatures(smelt, 'Weight', 'Length2')

bs = fdata.getSpecies('Bream', 'Smelt')
result, data = fdata.getFeatures(bs, 'Weight', 'Length2')

myfish = np.array([[200, 24]])
#myfish = np.array([[200, 24],[160,24]])

nc = KNeighborsClassifier()
nc.fit(data, result)
print(nc.score(data, result))
print(nc.predict(myfish))

print(nc.classes_)
dist, ind = nc.kneighbors(myfish)

print(nc.predict_proba(myfish[:2]))

plt.scatter(bd[:,1], bd[:,0])
plt.scatter(sd[:,1], sd[:,0])
plt.scatter(myfish[:,1], myfish[:,0])
plt.scatter(data[ind,1], data[ind, 0])

plt.xlabel('Length')
plt.ylabel('Weight')
plt.title('Length & Weight')
plt.legend(['bream', 'smelt', 'Myfish', 'neighbors'])
plt.show()

