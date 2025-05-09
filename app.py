from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import torch.nn as nn
from torchvision import models
import requests
import time

app = Flask(__name__)
CORS(app)  # This allows cross-origin requests

# Define the class names
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# Function to download model if not present
def download_model():
    model_dir = 'model'
    model_path = os.path.join(model_dir, 'VGG16_v2-OCT_Retina_half_dataset.pt')
    
    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Check if model exists locally
    if os.path.exists(model_path):
        print("Model already exists locally")
        return model_path
    
    # Get the model URL from environment variable
    model_url = os.environ.get('MODEL_URL')
    
    if not model_url:
        raise Exception("MODEL_URL environment variable not set")
    
    print(f"Downloading model from {model_url}")
    
    # Download with progress reporting
    start_time = time.time()
    response = requests.get(model_url, stream=True)
    response.raise_for_status()
    
    # Get total file size if available
    total_size = int(response.headers.get('content-length', 0))
    
    # Download and save the file
    with open(model_path, 'wb') as f:
        if total_size == 0:
            f.write(response.content)
        else:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = int(100 * downloaded / total_size)
                        if percent % 10 == 0:
                            print(f"Download progress: {percent}%")
    
    elapsed = time.time() - start_time
    print(f"Model download completed in {elapsed:.1f} seconds")
    return model_path

# Load the model
def load_model():
    # Create the model architecture
    vgg16 = models.vgg16_bn(pretrained=False)
    
    # Modify the classifier
    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, len(class_names))])  # Add our layer with 4 outputs
    vgg16.classifier = nn.Sequential(*features)
    
    try:
        # First ensure model is downloaded
        model_path = download_model()
        
        # Load the trained weights
        vgg16.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    
    # Set the model to evaluation mode
    vgg16.eval()
    return vgg16

# Transform for input images
def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

# Initialize model and transform
model = None
transform = get_transform()

@app.route('/predict', methods=['POST'])
def predict():
    global model
    
    # Lazy load the model on first request
    if model is None:
        model = load_model()
    
    # Check if image was sent in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read and process the image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            _, preds = torch.max(outputs, 1)
            pred_class = class_names[preds.item()]
            
            # Get probabilities for all classes
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            class_probs = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
            
        return jsonify({
            'prediction': pred_class,
            'probabilities': class_probs
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/', methods=['GET'])
def home():
    return """
    <html>
        <head>
            <title>OCT Retina Classification API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>OCT Retina Classification API</h1>
            <p>This API provides retina OCT image classification into four categories:</p>
            <ul>
                <li>CNV (Choroidal Neovascularization)</li>
                <li>DME (Diabetic Macular Edema)</li>
                <li>DRUSEN</li>
                <li>NORMAL</li>
            </ul>
            <h2>Endpoints:</h2>
            <ul>
                <li><code>POST /predict</code> - Upload an image for classification</li>
                <li><code>GET /health</code> - Health check endpoint</li>
            </ul>
            <p>To use this API, send a POST request to /predict with an image file.</p>
        </body>
    </html>
    """

if __name__ == '__main__':
    # Use the PORT environment variable provided by Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)