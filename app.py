from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import requests

def predicted_class(arr):
    dic = {0:'Glioma', 1:'Meningioma', 2:'Healthy', 3:'Pituitary'}
    return dic[arr.argmax()]

# Load your model
model = tf.keras.models.load_model('final_model.keras')

app = Flask(__name__)

def preprocess_image(image):
    # Resize image to (512, 512)
    image = image.resize((512, 512))
    # Convert image to numpy array
    image_array = np.array(image)
    
    # Ensure the image has 3 color channels
    if len(image_array.shape) == 2:  # Grayscale image
        image_array = np.stack([image_array]*3, axis=-1)
        
    # Normalize the image
    image_array = image_array / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict(image_array):
    # Predict using the model
    predictions = model.predict(image_array)
    return predicted_class(predictions)

def fetch_image_from_url(url):
    response = requests.get(url)
    image = Image.open(io.BytesIO(response.content))
    return image

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if image URL is provided
        image_url = request.form.get('image_url')
        if image_url:
            # Fetch and process the image from URL
            image = fetch_image_from_url(image_url)
            image_array = preprocess_image(image)
            
            # Make prediction
            result = predict(image_array)
            return render_template('result.html', result=result)
        
        # Check if file is uploaded
        file = request.files.get('file')
        if file:
            # Open and process the image
            image = Image.open(file).convert('RGB')  # Ensure the image is in RGB format
            image_array = preprocess_image(image)
            
            # Make prediction
            result = predict(image_array)
            return render_template('result.html', result=result)
    
    return render_template('upload.html')

@app.route('/static/<path:filename>')
def send_static(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    app.run(debug=True)
