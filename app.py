from flask import Flask, request, jsonify, render_template
import onnxruntime as ort
import numpy as np
import base64
from PIL import Image
import io

# Load the ONNX model
onnx_model_path = 'model.onnx'  # Replace with your ONNX model file
session = ort.InferenceSession(onnx_model_path)

app = Flask(__name__)

def preprocess_image(image_data):
    # Decode base64 image
    image = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1])))
    image = image.resize((288, 288))  # Adjust size to your model's input dimensions
    image = np.array(image) / 255.0  # Normalize the image
    # Adjust dimensions to match ONNX model input requirements
    # Add batch dimension if required (1, Height, Width, Channels)
    return np.expand_dims(image.astype(np.float32), axis=0)

@app.route('/')
def index():
    return render_template('index.html')  # Render the frontend HTML file

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']
        processed_image = preprocess_image(image_data)

        # Prepare input for ONNX model
        input_name = session.get_inputs()[0].name
        prediction = session.run(None, {input_name: processed_image})

        # Assuming the model outputs probabilities, find the class with the highest probability
        predicted_class = np.argmax(prediction[0], axis=1)[0]
        return jsonify({'prediction': int(predicted_class)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
