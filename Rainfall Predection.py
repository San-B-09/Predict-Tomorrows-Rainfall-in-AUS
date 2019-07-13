#Import libraries needed for the whole project

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#read your csv dataset file
train = pd.read_csv("E://ML//ML Projects//Rain_Prediction//weatherAUS_new.csv")     #Edit the file location to your desired location
print(train.head())         #Print first five rows of dataset to confirm the csv file is read properly


#Pre-Processing the data
print("Before Preprocessing:\n",train.isna().sum())

#Fill empty rows by mean->if data is in numeric format & mode->if data is in string format
train['MinTemp'].fillna(train['MinTemp'].mean(), inplace=True)
train['MaxTemp'].fillna(train['MaxTemp'].mean(), inplace=True)
train['Rainfall'].fillna(train['Rainfall'].mean(), inplace=True)
train['Evaporation'].fillna(train['Evaporation'].mean(), inplace=True)
train['Sunshine'].fillna(train['Sunshine'].mean(), inplace=True)
train['WindGustDir'].replace(np.NaN, train['WindGustDir'].mode()[0], inplace=True)
train['WindGustSpeed'].fillna(train['WindGustSpeed'].mean(), inplace=True)
train['WindDir9am'].fillna(train['WindDir9am'].mode()[0], inplace=True)
train['WindDir3pm'].fillna(train['WindDir3pm'].mode()[0], inplace=True)
train['WindSpeed9am'].fillna(train['WindSpeed9am'].mean(), inplace=True)
train['WindSpeed3pm'].fillna(train['WindSpeed3pm'].mean(), inplace=True)
train['Humidity9am'].fillna(train['Humidity9am'].mean(), inplace=True)
train['Humidity3pm'].fillna(train['Humidity3pm'].mean(), inplace=True)
train['Pressure9am'].fillna(train['Pressure9am'].mean(), inplace=True)
train['Pressure3pm'].fillna(train['Pressure3pm'].mean(), inplace=True)
train['Temp9am'].fillna(train['Temp9am'].mean(), inplace=True)
train['Temp3pm'].fillna(train['Temp3pm'].mean(), inplace=True)
train = train.dropna()      #drop the row which still has empty cells

print("\nAfter Preprocessing:\n",train.isna().sum())    #Make sure you don't have any empty cells
print("Data Shape = ", train.shape)         #make sure your data loss after preprocessing isn't huge

#Data Visualisation
fig = plt.figure(figsize=(15,60))
i = 1
for sibsp in train['Location'].unique():
    fig.add_subplot(20, 4, i)
    plt.title('Location : {}'.format(sibsp))
    train.RainTomorrow[train['Location'] == sibsp].value_counts().plot(kind='pie')
    i += 1

fig = plt.figure(figsize=(15,20))
i = 1
for sibsp in train['WindGustDir'].unique():
    fig.add_subplot(5, 4, i)
    plt.title('WindGustDir : {}'.format(sibsp))
    train.RainTomorrow[train['WindGustDir'] == sibsp].value_counts().plot(kind='pie')
    i += 1

#Last part of preprocessing -> converting string data to numeric as strings can't be ploted
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
train['WindGustDir'] = label_encoder.fit_transform(train['WindGustDir'])
train['Location'] = label_encoder.fit_transform(train['Location'])
train['WindDir9am'] = label_encoder.fit_transform(train['WindDir9am'])
train['WindDir3pm'] = label_encoder.fit_transform(train['WindDir3pm'])
train['RainTomorrow'] = label_encoder.fit_transform(train['RainTomorrow'])

#Dividing dataset into train and test set to verify accuracy of different methods
x = train.iloc[:, :-1].values
y = train.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
train.head()


#Apply different methods to find out method giving highest accuracy

#Linear Regression
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(x_train, y_train)
pred = regressor.predict(x_test)
print("\nLogistic Regression Accuracy Score: ", accuracy_score(y_test,pred))

#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(x_train,y_train)
pred = KNN.predict(x_test)
print("\nKNN Classifier Accuracy Score: ", accuracy_score(y_test,pred))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=300)
clf.fit(x_train,y_train)
pred=clf.predict(x_test)
print("\nRandom Forest Classifier Accuracy Score: ", accuracy_score(y_test,pred))

#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=300, max_features=2, max_depth=2, random_state=0)
gb.fit(x_train,y_train)
pred = gb.predict(x_test)

print("\nGradient Boosting Classifier Accuracy Score: ", accuracy_score(y_test,pred))