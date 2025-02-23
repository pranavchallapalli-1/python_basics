import streamlit as st # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load data
@st.cache_data #reads data in streamlit
def load_data():
    data=pd.read_csv("./titanic_dataset.csv")
    return data

data=load_data()
#Title and description
st.title("Exploratory Data Analysis of the Titanic Dataset")
st.write("This is an EDA on the TItanic dataset.")
st.write("First few rows of the dataset:")
st.dataframe(data.head()) ##printing the dataset in app

#data cleaning section
st.subheader('Missing Values')
missing_data=data.isnull().sum()
st.write(missing_data)

if st.checkbox('Fill missing Age with median'):
    data['Age'].fillna(data['Age'].median(),inplace=True)

if st.checkbox('Fill missing values Embarked with mode'):
    data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)
if st.checkbox('Drop duplicates'):
    data.drop_duplicates(inplace=True)
    st.write(data.isnull().sum())


#EDA Section
st.subheader('Statistical Summary Of The Data')
st.write(data.describe())

#Age Distribution
st.subheader('Age Distribution')
fig,ax=plt.subplots()
sns.histplot(data['Age'],kde=True,ax=ax)
ax.set_title('Age Distribution')
st.pyplot(fig)


# Gender Distribution
st.subheader('Gender Distribution')
fig, ax = plt.subplots()
sns.countplot(x='Sex', data=data, ax=ax)
ax.set_title('Gender Distribution')
st.pyplot(fig)

#Pclass vs Survived
st.subheader('Pclass')
fig,ax=plt.subplots()
sns.countplot(x='Pclass',hue='Survived',data=data,ax=ax)
ax.set_title('Pclass vs Survived')
st.pyplot(fig)


# Feature Engineering Section
st.subheader('Feature Engineering: Family Size')
data['FamilySize'] = data['SibSp'] + data['Parch']
fig,ax = plt.subplots()
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