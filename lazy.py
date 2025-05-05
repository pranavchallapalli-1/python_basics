import pandas as pd
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor

df=pd.read_csv(r"D:\Naresh_IT\coding\ml_nlp_assignment\student_info.csv")
df.isnull().sum()
df.info()
df.describe()
df2=df.fillna(df.mean())
x=df2.drop("student_marks",axis='columns')
y=df.drop("study_hours",axis="columns")
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
reg=LazyRegressor(verbose=0)
models,prediction=reg.fit(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)
print(models)