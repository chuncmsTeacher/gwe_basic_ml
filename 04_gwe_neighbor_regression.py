from gwe_fishdata import FishData
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

import numpy as np

fdata = FishData()
perch = fdata.getSpecies('Perch')
pr, pd = fdata.getFeatures(perch, 'Weight', 'Length2')
#myfish = np.array([[5,],[10,],[35,],[37,],[50,],[100,]])
myfish = np.array([[35,],[37,]])
nr = KNeighborsRegressor()
nr.fit(pd[:,1].reshape(-1,1),pd[:,0])
prediction = nr.predict(myfish)
for i in range(len(myfish)):
    print(myfish[i], '==>', prediction[i])

plt.scatter(pd[:,1], pd[:,0])
plt.scatter(myfish, prediction)
plt.show()

