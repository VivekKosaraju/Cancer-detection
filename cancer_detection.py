# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 15:19:06 2020

@author: kosaraju vivek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#step1 : importing dataset
cancer=pd.read_csv('cancer.csv')
##viewing datasets
print("  viewing head   " )
print(cancer.head())
print("  Shape  ")
print(cancer.shape)

print("    No of null values in each coloumn")
cancer.isnull().sum()
cancer=cancer.dropna(axis=1)
cancer=cancer.iloc[:,1:]
print(cancer.shape)

cancer['diagnosis'].value_counts()
sns.countplot(cancer['diagnosis'],label='count')
print("   information of dataset     ")
print(cancer.info())

#encoding categorical data into continous data
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
cancer.iloc[:,0]=encoder.fit_transform(cancer.iloc[:,0].values)
cancer.iloc[:,0]


print("   Statistics   ")
print(cancer.describe())

#Visulizing dataset
print("   Histograms    ")
cancer.hist(bins=50,figsize=(30,20))
plt.show()
print("  Pairplot  ")
sns.pairplot(cancer.iloc[:,:5],hue='diagnosis')
plt.show()


#find correlation
print("  Correlation   ")
cor_relation=cancer.corr()
print(cor_relation['diagnosis'].sort_values(ascending=False))
plt.figure(figsize=(30,20))
print("     Heatmap    ")
sns.heatmap(cor_relation,annot=True)
plt.show()

x=cancer.iloc[:,1:].values
y=cancer.iloc[:,0].values

#step2: Data pre-processing
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values="NaN",strategy="most_frequent")
x[:,1:]=imp.fit_transform(x[:,1:])
y=imp.fit_transform(y.reshape(569,1))

#step3: Splitting data into traing and testing data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
xtrain = sc_X.fit_transform(xtrain)
xtest = sc_X.transform(xtest)

#Step3: fitting into the models

#model1: KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(xtrain,ytrain)
ypred_knn=knn.predict(xtest)

#model2: Logistic regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(xtrain,ytrain)
ypred_lr=lr.predict(xtest)

#model3: SVM
from sklearn.svm import SVC
sv=SVC()
sv.fit(xtrain,ytrain)
ypred_sv=sv.predict(xtest)

#model4: Naive bayes
from sklearn.naive_bayes import GaussianNB
naive=GaussianNB()
naive.fit(xtrain,ytrain)
ypred_naive=naive.predict(xtest)

#model5: Decision_tree
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier()
tree.fit(xtrain,ytrain)
ypred_tree=tree.predict(xtest)

#model6: Random_forest
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier()
forest.fit(xtrain,ytrain)
ypred_forest=forest.predict(xtest)

#Step4: Model evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report

print("Accuracy score for KNN : ",accuracy_score(ytest,ypred_knn))
print("Accuracy score for Logistic_Regression : ",accuracy_score(ytest,ypred_lr))
print("Accuracy score for SVM : ",accuracy_score(ytest,ypred_sv))
print("Accuracy score for naive_bayes : ",accuracy_score(ytest,ypred_naive))
print("Accuracy score for decision_tree : ",accuracy_score(ytest,ypred_tree))
print("Accuracy score for Random_forest : ",accuracy_score(ytest,ypred_forest))


print(" confusion matrix for KNN \n",confusion_matrix(ytest,ypred_knn))
print(" confusion matrix for Logistic_Regression \n",confusion_matrix(ytest,ypred_lr))
print(" confusion matrix for SVM \n",confusion_matrix(ytest,ypred_sv))
print(" confusion matrix for naive_bayes\n ",confusion_matrix(ytest,ypred_naive))
print(" confusion matrix for decision_tree \n",confusion_matrix(ytest,ypred_tree))
print(" confusion matrix for Random_forest\n ",confusion_matrix(ytest,ypred_forest))

print("classification report for KNN\n ",classification_report(ytest,ypred_knn))
print("classification report for Logstic_regression\n ",classification_report(ytest,ypred_lr))
print("classification report for SVM \n",classification_report(ytest,ypred_sv))
print("classification report for naive_bayes\n ",classification_report(ytest,ypred_naive))
print("classification report for Decision_tree\n ",classification_report(ytest,ypred_tree))
print("classification report for Random_forest \n",classification_report(ytest,ypred_forest))







