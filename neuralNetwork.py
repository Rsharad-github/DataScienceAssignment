data = load_iris()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = Sequential()
model.add(Dense(10, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=1)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model accuracy on test data: {accuracy*100:.2f}%")

sample = np.array([X_test[0]])
prediction = model.predict(sample)
print("Prediction:", np.argmax(prediction))
print("Actual Label:", y_test[0])
