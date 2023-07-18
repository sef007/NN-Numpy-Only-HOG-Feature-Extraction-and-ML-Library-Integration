import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tqdm import tqdm

# Fetch and load the MNIST dataset
mnist = pd.read_csv('train.csv')

y = mnist['label']
X = mnist.iloc[:, 1:]

data = np.array(X, dtype='int16')
target = np.array(y, dtype='int')

# Perform HOG feature extraction
list_hog = []
print("Extracting Features\nHistogram of Oriented Gradients - ")
for feature in tqdm(data):
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
    list_hog.append(fd)
print("'HOG' Feature Extraction Complete\n")
hog_features = np.array(list_hog, dtype='float64')

# Scale the HOG features
scaler = preprocessing.StandardScaler()
hog_features_transformed = scaler.fit_transform(hog_features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(hog_features_transformed, target, random_state=0)

# Neural Network Training
_, m_train = X_train.shape
_, n_features = X_train.shape


model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(n_features,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=64, epochs=50, verbose=1)

# Final Training Accuracy
_, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
print("Final Training Accuracy: ", round(train_accuracy * 100, 2), "%")

# Final Test Accuracy
_, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Final Test Accuracy: ", round(test_accuracy * 100, 2), "%")
