# Author Fazel Rabbi
# Data 221 Final Project
# KNN Model
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier



# Load the data and identify the target variable
Data_frame_creditcard_approval = pd.read_csv("cleaned_data.csv")

X = Data_frame_creditcard_approval.drop('TARGET', axis=1)
Y = Data_frame_creditcard_approval['TARGET']

# Encode Categorical variables
# Converts Categorical values into numerical ones
# drop_true=True avoids duplicate columns, do this so it doesnt confuse the model
X = pd.get_dummies(X, drop_first=True)

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    random_state=42,
    test_size=0.2
)

# Scale and standardize the input data so the neural network trains better
scaler = StandardScaler()

# Fit the scaler on the training set then transform it, then only transform the testing set
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Set the K values we wish to test
k_values = range(1, 21)

# Create storage for evaluation metrics
accuracy_scores = []
f1_scores = []
roc_auc_scores = []

# Build the KNN model for each value of k
for x in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=x)
    knn_model.fit(X_train, Y_train)
    knn_model_predict = knn_model.predict(X_test)
    knn_model_prob = knn_model.predict_proba(X_test)[:,1]

    # Append the results to the evaluation metrics storage
    accuracy_scores.append(accuracy_score(Y_test, knn_model_predict))
    f1_scores.append(f1_score(Y_test, knn_model_predict))
    roc_auc_scores.append(roc_auc_score(Y_test, knn_model_prob))

results_kvalues = pd.DataFrame({
    "K Values": list(k_values),
    "Accuracy": accuracy_scores,
    "F1 Scores": f1_scores,
    "ROC-AUC": roc_auc_scores
})


