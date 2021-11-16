from gwe_fishdata import FishData
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

import numpy as np


d = FishData()

perch = d.getSpecies('Perch')
pr, pdata = d.getFeatures(perch, 'Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width')
weight = pdata[:,0]
data = pdata[:,1:]

ss = StandardScaler()
ss.fit(data)
scaleddata = ss.transform(data)

xt, yt, xw, yw = train_test_split(scaleddata, weight)

lg = LinearRegression()
lg.fit(xt, xw)
print(lg.score(xt, xw))
print(lg.score(yt, yw))
print(lg.coef_, lg.intercept_)

pf = PolynomialFeatures()
pf.fit(xt)
xt = pf.transform(xt)
yt = pf.transform(yt)
lg.fit(xt, xw)
print(lg.score(xt, xw))
print(lg.score(yt, yw))
print(lg.coef_, lg.intercept_)
print(pf.get_feature_names())


la = Lasso()
la.fit(xt, xw)
print(la.score(xt, xw))
print(la.score(yt, yw))
print(la.coef_, la.intercept_)

ri = Ridge()
ri.fit(xt, xw)
print(ri.score(xt, xw))
print(ri.score(yt, yw))
print(ri.coef_, la.intercept_)
