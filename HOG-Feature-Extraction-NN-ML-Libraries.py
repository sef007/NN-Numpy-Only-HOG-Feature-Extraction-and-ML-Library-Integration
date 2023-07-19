import numpy as np
import pandas as pd #used for data manipulation 
from skimage.feature import hog #used for hog extraction
from sklearn import preprocessing # used for preprocessing and data splitting
from sklearn.model_selection import train_test_split 
from keras.models import Sequential #keras is used to actually build the NN model
from keras.layers import Dense
from keras.optimizers import Adam
from tqdm import tqdm #this is for progress monitoring, used for progress bar.

# Fetch and load the MNIST dataset
mnist = pd.read_csv('train.csv')

y = mnist['label']
X = mnist.iloc[:, 1:]

data = np.array(X, dtype='int16')
target = np.array(y, dtype='int')

# start to perform HOG feature extraction
list_hog = []
print("Extracting Features\nHistogram of Oriented Gradients - ")
for feature in tqdm(data):
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
    list_hog.append(fd)
print("'HOG' Feature Extraction Complete\n")
hog_features = np.array(list_hog, dtype='float64')

# Scaling the HOG features using sklearn 
scaler = preprocessing.StandardScaler()
hog_features_transformed = scaler.fit_transform(hog_features)

# splits the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(hog_features_transformed, target, random_state=0)

# Neural Network Training
_, m_train = X_train.shape
_, n_features = X_train.shape

#NN model with fully connected (Dense) layers using rectified linear activation functions.
model = Sequential() #chose a sequential neural network using Kera
model.add(Dense(128, activation='relu', input_shape=(n_features,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
# notice the last layer has ten units, this is the ten classes (0-9).

# implement the Adam Optimiser (this will adjust the learning rate adaptively), set the hyperparameters and compile the model.
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# the sparse categorical crossentropy is theh liss functionn and will calculate cross entropy loss.

# training the model 
model.fit(X_train, y_train, batch_size=64, epochs=50, verbose=1)
#batch size is the number of samples processed in one iteration. smaller = faster but low convergance, higher = slower but higher convergence.
# epochs - represents a complete pass through the entire training dataset.
# verbose simply displays the amount of information displayed to the console.

# Final Training Accuracy
_, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
print("Final Training Accuracy: ", round(train_accuracy * 100, 2), "%")

# Final Test Accuracy (done to ensure no overfitting) 
_, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Final Test Accuracy: ", round(test_accuracy * 100, 2), "%")
