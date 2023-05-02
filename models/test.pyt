import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('cnn-mnist-model.h5')

# Load and preprocess the image
image = Image.open('testImages/1.jpeg').convert('L')  # convert to grayscale
image = image.resize((28, 28))  # resize to 28x28 pixels
image_array = np.array(image)  # convert to numpy array
image_array = image_array.astype('float32') / 255  # normalize pixel values
image_array = image_array.reshape((1, 28, 28, 1))  # reshape to match model input shape

# Make a prediction
prediction = model.predict(image_array)
predicted_class = np.argmax(prediction[0])

# Print the predicted class
print('Predicted class:', predicted_class)
