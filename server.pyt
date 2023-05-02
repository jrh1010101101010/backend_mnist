from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
from PIL import Image
import io
import base64


app = Flask(__name__)

model = tf.keras.models.load_model('model.h5')

def preprocess_image(image):

    img = image.convert('L').resize((28, 28))
   
    img_array = np.array(img)

    img_array = 255 - img_array
    img_array = np.expand_dims(img_array, axis=0)
    
 
    img_array = img_array.reshape(img_array.shape[0], 28, 28, 1)
    
 
    img_array = img_array.astype('float32') / 255.0
    
    return img_array


@app.route('/', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']
    
    
    decoded_image = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(decoded_image))
    
    img_array = preprocess_image(img)
    prediction = np.argmax(model.predict(img_array))
    
    return jsonify({'prediction': str(prediction)})


if __name__ == '__main__':
    app.run(port=5000)

