# Drug-Prediction-By-Using-DecisionTreeClassifier
I have created a machine learning model to predict the correct drug for a patient based on a given dataset using the Decision Tree Classifier.
About Dataset :-
Imagine that you are a medical researcher compiling data for a study. You have collected data about a set of patients, all of whom suffered from the same illness. During their course of treatment, each patient responded to one of 5 medications, Drug A, Drug B, Drug c, Drug x and y. The features of this dataset are Age, Sex, Blood Pressure, and the Cholesterol of the patients, and the target is the drug that each patient responded to.

Step 1: Firstly i imported the required libraries for the program used libraries like pandas for data handling, matplotlib for visualization, and scikit-learn for building the model.

Step 2: Loaded the dataset using pandas and displayed its columns, top 5 rows, and dimensions for understanding the data structure

Step 3: Encoded categorical features (Sex, BP, Cholesterol) using one-hot encoding.
Encoded the target column (Drug) using LabelEncoder.

Step 4: Separated the features (x) and target labels (y).
Split the data into training (75%) and testing (25%) sets using train_test_split

Step 5: making an object of DecisionTreeClassifier and .fit() method is used for training the model . and then displaying the score of training data and testing data .predict() is used to testing the data and then calculated the accuracy and displayed the confusion matrix.

Plotted a Feature Importance Graph to visualize which features significantly influenced the model's predictions. Age and Na_to_K were the most important features.

Training Accuracy: 100%
Testing Accuracy: 98%
