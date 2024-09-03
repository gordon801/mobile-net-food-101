import io
import torch
from flask import Flask, jsonify, request, render_template
from PIL import Image
from main import DEVICE, DTYPE, FOOD101_CLASSES
from src.mobilenet import MyMobileNet
from src.data_utils import get_test_transform, load_classes

# Set constants
MODEL_PATH = 'checkpoint/final-model/full-50e-lr1e-4/best_model.pth.tar' # Replace with the file path to your final trained model's checkpoint for deployment
CLASS_PATH = 'data/food-101/meta/classes.txt'

app = Flask(__name__)

# Load classes and model
classes = load_classes(CLASS_PATH)

model = MyMobileNet(
    output_classes=FOOD101_CLASSES, 
    device=DEVICE, 
    checkpoint_path=MODEL_PATH
)
model.eval()

def transform_image(image_bytes):
    """Process raw image bytes and apply test image transformations for model input."""
    transform = get_test_transform()
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(image_bytes):
    """Get predicted class index and class for the image bytes."""
    x = transform_image(image_bytes)
    x = x.to(device=DEVICE, dtype=DTYPE)

    with torch.no_grad(): # Disable gradient tracking
        score = model(x)
        _, pred = score.max(1)
    return pred.item(), classes[pred.item()]

@app.route('/', methods=['GET'])
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    if request.method == 'POST':
        # Receive and read file from request
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})

if __name__ == '__main__':
    app.run(debug=True)