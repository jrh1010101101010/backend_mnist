import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

#load data from the model
train_df = pd.read_csv("data/mnist_train.csv")
test_df = pd.read_csv("data/mnist_test.csv")

# extract the information from the model
train_images = train_df.iloc[:, 1:].values.astype('float32')
train_labels = train_df.iloc[:, 0].values.astype('int32')

test_labels = test_df.iloc[:, 0].values.astype('int32')
test_images = test_df.iloc[:, 1:].values.astype('float32')

# reshape the model to an approate size
train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
test_images = test_images.reshape(-1, 28, 28, 1) / 255.0

# define the model architeure 
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10)
])


# compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train the model
model.fit(train_images, train_labels, epochs=5)


# evaluate the accuracy of the predictions
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Predict the label of the image
predictions = model.predict(image_array)
predicted_label = np.argmax(predictions)

# Display the predicted label
print("The predicted label is:", predicted_label)

model.save('new_model.h5')


#flask