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
n2 = np.column_stack((pd[:,1]**2, pd[:,1]))
lr.fit(n2,pd[:,0])

n2fish = np.column_stack((myfish**2, myfish))

prediction = lr.predict(n2fish)
for i in range(len(myfish)):
    print(myfish[i], '==>', prediction[i])

print(lr.coef_, lr.intercept_)

X = np.arange(15,45)
Y = lr.coef_[0]*(X**2) + lr.coef_[1]*X + lr.intercept_
plt.plot(X, Y)
plt.scatter(pd[:,1], pd[:,0])
plt.scatter(myfish, prediction)
plt.show()

