import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


df=pd.read_csv("./logit classification.csv")
x=df.iloc[:,[2,3]]
y=df.iloc[:,-1]
#splitting of dataset
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.20,random_state=0)
##scaling
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

dtree=DecisionTreeClassifier(criterion='entropy',max_depth=10,random_state=None)
dtree.fit(x_train,y_train)
y_pred=dtree.predict(x_test)
cm=confusion_matrix(y_test, y_pred)
acc=accuracy_score(y_test, y_pred)
bias=dtree.score(x_train, y_train)
print(bias)
variance=dtree.score(x_test,y_test)
print(variance)
cr=classification_report(y_test, y_pred)
print(cr)

df2=pd.read_csv("./final1.csv")
x_future=df2.iloc[:,[3,4]]
x_future=sc.fit_transform(x_future)
y_future_predict=dtree.predict(x_future)
y_future_predict
