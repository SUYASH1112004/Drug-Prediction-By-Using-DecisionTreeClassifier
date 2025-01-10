# Importing the libraries Required
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
#-----------------------------------------------------------

#-----------------------------------------------------------
print ("Drug Prediction By Using Decision Tree Classifier")
#-----------------------------------------------------------

#----------------------------------------------------------
#Load the dataset and encoding the catagorical data
drug=pd.read_csv('D:\CodeTech Internship\drug200.csv')
print("Columns of dataset :\n",drug.columns)
print("First Five record of dataset :\n",drug.head())
print("Dimension Of Dataset : {}".format(drug.shape))

encoding=pd.get_dummies(drug,columns=['Sex','BP','Cholesterol'],drop_first=True)  #Encoding Features

le=LabelEncoder()
drug['Drug']=le.fit_transform(drug['Drug'])   #Encoding Labels
#----------------------------------------------------------


#----------------------------------------------------------
#Training The Data

x=encoding.drop('Drug',axis=1)  # Feature / independent variable 
y=drug['Drug']   # Label / dependent variable

x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=42)
model=tree.DecisionTreeClassifier()
model.fit(x_train,y_train)
#---------------------------------------------------------------

#Testing the data And Displaying Accuracy
print("Accuracy on training data :",model.score(x_train,y_train))
print("Accuracy on testing data set :",model.score(x_test,y_test))

predicted_value=model.predict(x_test)
print ("Accuracy :",accuracy_score(y_test,predicted_value) * 100)

print ("Confusion Matrix :\n",confusion_matrix(y_test,predicted_value))   # Displaying Confusion Matrix

#---------------------------------------------------------------
# plotting Feature Importance
plt.figure(figsize=(12,6))
n_features=x.shape[1]
plt.barh(range(n_features),model.feature_importances_,align='center')
plt.yticks(np.arange(n_features),x.columns)
plt.ylabel('Feature')
plt.xlabel('Feature Importance')
plt.ylim(-1,n_features)
plt.show()
#-----------------------------------------------------------------
