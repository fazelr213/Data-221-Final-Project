# Author Fazel Rabbi
# Data 221 Final Project
# Neural Network


from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler

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

# Build the neural network
neural_model = Sequential()

# Input Layer
neural_model.add(Input(shape=(X_train.shape[1],)))

# First Hidden Layer
# relu activation helps the model learn non-linear patterns
neural_model.add(Dense(64, activation="relu"))

# Second Hidden Layer
# relu activation helps the model learn non-linear patterns
neural_model.add(Dense(32, activation="relu"))

# Output layer
# Sigmoid is used for binary classification
neural_model.add(Dense(1, activation="sigmoid"))

# Compile the model
# adam decides how the model updates its weights to reduce error
# binary-crossentropy measures how wrong the predictions are
neural_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the neural network
# Validation split uses part of the training data to check the models performance
trained_neural_network = neural_model.fit(
    X_train,
    Y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)







