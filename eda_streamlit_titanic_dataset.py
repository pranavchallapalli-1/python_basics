import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


@st.cache_data
def load_data():
    data=pd.read_csv(r"titanic dataset.csv")
    return  data


data=load_data()
print(data.info())

st.title("EDA Dashboard about titanic dataset")
st.write("First few rows of the dataset:")
st.dataframe(data.head(10))
x=data.isnull().sum()
st.subheader("The misisng values with respect to column names:")
st.write(x)

if st.checkbox('Filling the age with its median'):
    data["Age"]=data['Age'].fillna(data["Age"].median())
if st.checkbox('Filling the emabrked with its mode'):
    data['Embarked']=data['Embarked'].fillna(data['Embarked'].mode()[0])

if st.checkbox("Drop duplicate values "):
    data.drop_duplicates()
st.subheader("Cleaned Dataset")
st.dataframe(data)

#statictical info of the data set
st.subheader("Statistical data for the given dataset")
st.write(data.describe())
#Gender Distribution
f,ax=plt.subplots()
sns.countplot(x="Sex",data=data,ax=ax)
ax.set_title("Gender distribution")
st.pyplot(f)

#Age distribution
f,ax=plt.subplots()
sns.histplot(x="Age",data=data,ax=ax)
ax.set_title("Age distribution")
st.pyplot(f)
#relation between survived and sex
st.subheader("Relation Between survied people and the gender ")
# f,ax=plt.subplots()
a=sns.catplot(x='Survived',col='Sex',data=data,height=8,aspect=1,kind="count",ax=ax)
# ax.set_title("Realtion between Survivors rate and gender")
st.pyplot(a)

# relation between Survived and pclass
st.subheader("Relation between survivers and  their chosen class")
b=sns.catplot(x="Survived",col="Pclass",height=8,aspect=1,data=data,kind="count")
st.pyplot(b)
#correlation
st.subheader("Correlation :")
correlation=data.select_dtypes(['number']).corr()
st.write(correlation)

#heatmap
st.subheader("Heatmap for the correlation")
fig,ax=plt.subplots()
sns.heatmap(data.select_dtypes(['number']).corr(),annot=True,xticklabels=False,yticklabels=False,ax=ax)
ax.set_title("HeatMap")
st.pyplot(fig)

#Feature engineering
st.subheader('Feature Engineering: Family Size')
data['FamilySize'] = data['SibSp'] + data['Parch']
fig, ax = plt.subplots()
sns.histplot(data['FamilySize'], kde=True, ax=ax)
ax.set_title('Family Size Distribution')
st.pyplot(fig)

# Conclusion Section
st.subheader('Key Insights')
insights = """
- Females have a higher survival rate than males.
- Passengers in 1st class had the highest survival rate.
- The majority of passengers are in Pclass 3.
- Younger passengers tended to survive more often.
"""
st.write(insights)
