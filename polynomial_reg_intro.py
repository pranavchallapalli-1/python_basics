import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lazypredict.Supervised import LazyRegressor
import matplotlib.pyplot as plt

dataset=pd.read_csv("emp_sal.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

lin_reg=LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x, y, color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title("Linear regression model")
plt.xlabel('Position level')
plt.ylabel("Salary")
plt.show()

m=lin_reg.coef_
m
c=lin_reg.intercept_
c

linmodel_pred=lin_reg.predict([[6.5]])
linmodel_pred

## polynomial reg(non linear model)
poly_reg=PolynomialFeatures(degree=5)
x_poly=poly_reg.fit_transform(x)

poly_reg.fit(x_poly)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)

plt.scatter(x, y, color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title("Linear regression model")
plt.xlabel('Position level')
plt.ylabel("Salary")
plt.show()

poly_model_pred=lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred

### SVR model(Support vector regressor model)

svr_reg=SVR(kernel='poly',degree=4,gamma='auto',C=10.0)
svr_reg.fit(x, y)
svr_model_pred=svr_reg.predict([[6.5]])
svr_model_pred

### knn reg

knn_reg=KNeighborsRegressor(n_neighbors=4,weights='uniform')
knn_reg.fit(x,y)
knn_model_predict=knn_reg.predict([[6.5]])
knn_model_predict

### descion tree
desicion_tree_reg=DecisionTreeRegressor(criterion='absolute_error',splitter='random',)
desicion_tree_reg.fit(x,y)
desicion_tree_predict=desicion_tree_reg.predict([[6.5]])
desicion_tree_predict

### random tree
random_rf_reg=RandomForestRegressor(random_state=0,criterion='poisson')
random_rf_reg.fit(x,y)
rf_pred=random_rf_reg.predict([[6.5]])
rf_pred

###lazy predict
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=1/3,random_state=0)
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(x_train, x_test, y_train, y_test)

print(models)
