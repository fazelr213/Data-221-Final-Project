# Author Fazel Rabbi
# Data 221 Final Project
# Neural Network


from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the data and identify the target variable
Data_frame_creditcard_approval = pd.read_csv("cleaned_data.csv")

X = Data_frame_creditcard_approval.drop("TARGET", axis=1)
Y = Data_frame_creditcard_approval["TARGET"]

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
