from gwe_fishdata import FishData
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

fdata = FishData()
perch = fdata.getSpecies('Perch')
pr, pd = fdata.getFeatures(perch, 'Weight', 'Length2')
#myfish = np.array([[5,],[10,],[35,],[37,],[50,],[100,]])
myfish = np.array([[20], [35,],[37,]])

lr = LinearRegression()
lr.fit(pd[:,1].reshape(-1,1),pd[:,0])

prediction = lr.predict(myfish)
for i in range(len(myfish)):
    print(myfish[i], '==>', prediction[i])

print(lr.coef_, lr.intercept_)
X = range(15,45)
Y = lr.coef_*X + lr.intercept_
plt.plot(X, Y)
plt.scatter(pd[:,1], pd[:,0])
plt.scatter(myfish, prediction)
plt.show()

