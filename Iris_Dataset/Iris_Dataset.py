# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import normalize, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = datasets.load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['Species'] = iris.target

# Rename columns for consistency
data.rename(columns={
    "sepal length (cm)": "SepalLengthCm",
    "sepal width (cm)": "SepalWidthCm",
    "petal length (cm)": "PetalLengthCm",
    "petal width (cm)": "PetalWidthCm"
}, inplace=True)

# Encode labels
le = LabelEncoder()
data['Species'] = le.fit_transform(data['Species'])

# Split features and labels
X = data.iloc[:, :-1].values
y = data['Species'].values

# Normalize features
X_normalized = normalize(X, axis=0)

# One-hot encode labels
y_encoded = to_categorical(y, num_classes=3)

# Split into train/test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_encoded, test_size=0.2, random_state=42
)

# Define the neural network model
model = Sequential([
    Dense(1000, input_dim=4, activation='relu'),
    Dense(500, activation='relu'),
    Dense(300, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=20, epochs=10, verbose=1)

# Evaluate accuracy manually
predictions = model.predict(X_test)
accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) * 100
print(f"Accuracy of the model: {accuracy:.2f}%")
