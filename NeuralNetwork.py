import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()
# The dataset contains information about three types of iris flowers (setosa, versicolor, virginica
) and their features:
# Sepal length. Sepal width.Petal length.Petal width.

X = data.data#X will now contain the 4 feature values for each sample.i.e,sepal width
y = data.target#y(array) has values 0, 1, and 2, representing the three flower species.

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
#Using the same random_state guarantees that you get the same split every time.

model = Sequential()
model.add(Dense(10, input_shape=(X_train.shape[1],), activation='relu'))#represents the number of features in the training data.
model.add(Dense(8, activation='relu'))#ReLU(activation fn) sets all negative values to 0 and keeps positive values unchanged.
model.add(Dense(3, activation='softmax'))#Converts the output of the 3 neurons into probabilities.



model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#adam Automatically adjusts the learning rate for each parameter during training.
#sparse_categorical_crossentropy is used to measure the difference between the predicted output and the actual target labels.
#Metrics are used to monitor the model's performance during training and evaluation. Measures the percentage of correct predictions.
model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=1)#The fit function is used to train the model by adjusting its weights using the training data.
# the model will iterate through the entire training set 50 times.
#the model will process 5 samples at a time and then update the weights.
loss, accuracy = model.evaluate(X_test, y_test)#The evaluate method returns a tuple
print(f"Model accuracy on test data: {accuracy*100:.2f}%")

sample = np.array([X_test[0]])
prediction = model.predict(sample)
print("Prediction:", np.argmax(prediction))
print("Actual Label:", y_test[0])
