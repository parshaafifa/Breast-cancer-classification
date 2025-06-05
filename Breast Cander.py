import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# Load dataset
dataset = sklearn.datasets.load_breast_cancer()
data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
data['label'] = dataset.target

# Split features and labels
X = data.drop(['label'], axis=1)
Y = data['label']

# Split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Set random seed
tf.random.set_seed(3)

# Build model (using softmax for sparse categorical crossentropy)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='softmax')  # 2 units for classes 0 and 1
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Evaluate on test set
loss, accuracy = model.evaluate(X_test_std, Y_test)
print("Test Accuracy:", accuracy)

# Predictions
Y_pred = model.predict(X_test_std)
print("Predicted probabilities for first test sample:", Y_pred[0])
#converting the predicted probablity to class labels
Y_pred_labels=[np.argmax(i) for i in Y_pred]
print(Y_pred_labels)
input_data = [15.34,14.26,102.5,704.4,0.1073,0.2135,0.2077,0.09756,0.2521,0.07032,
              0.4388,0.7096,3.384,44.91,0.006789,0.05328,0.06446,0.02252,0.03672,0.004394,
              18.07,19.08,125.1,980.9,0.139,0.5954,0.6305,0.2393,0.4667,0.09946]

input_data_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_numpy_array.reshape(1, -1)
input_data_std = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_std)
predicted_class = np.argmax(prediction)

print("Predicted class:", predicted_class)

if predicted_class == 0:
    print('The tumor is **Malignant**')
else:
    print('The tumor is **Benign**')

