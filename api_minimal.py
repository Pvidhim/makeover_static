from flask import Flask, request, jsonify, send_file, make_response
import base64
import numpy as np
import io
from PIL import Image, ImageFile
import matplotlib.pyplot as plt@app.route("/upload", methods=["POST", "OPTIONS"])
def upload_image():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        print("Processing upload request...")
        if "image" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["image"]
        if not file.filename:
            return jsonify({"error": "No file selected"}), 400
            
        filename = file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        print(f"File saved: {filepath}")
        return jsonify({"imageUrl": f"/uploads/{filename}"}), 200
        
    except Exception as e:
        print(f"Error in upload: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500ask_cors import CORS
import cv2
import os
import time
import threading

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://192.168.1.16:3000"], supports_credentials=True)

# Global variables for lazy loading
detector = None
predictor = None
face_mesh = None
mp_face_mesh = None
table = None
models_loaded = False
loading_lock = threading.Lock()
dlib = None
mp = None

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

def load_heavy_imports():
    """Import heavy libraries only when needed"""
    global dlib, mp
    try:
        print("  - Importing dlib...")
        import dlib as dlib_module
        dlib = dlib_module
        print("  - Importing MediaPipe...")
        import mediapipe as mp_module
        mp = mp_module
        return True
    except Exception as e:
        print(f"❌ Failed to import heavy libraries: {e}")
        return False

def load_models():
    """Load all heavy models once when first needed"""
    global detector, predictor, face_mesh, mp_face_mesh, table, models_loaded
    
    with loading_lock:
        if models_loaded:
            return True
            
        print("🔄 Loading ML models... This may take a moment.")
        start_time = time.time()
        
        try:
            # First, import heavy libraries
            if not load_heavy_imports():
                raise Exception("Failed to import required libraries")
            
            # Load dlib models
            print("  - Loading dlib face detector...")
            detector = dlib.get_frontal_face_detector()
            
            print("  - Loading dlib shape predictor...")
            if not os.path.exists(PREDICTOR_PATH):
                raise Exception(f"Shape predictor file not found: {PREDICTOR_PATH}")
            predictor = dlib.shape_predictor(PREDICTOR_PATH)
            
            # Load MediaPipe models
            print("  - Loading MediaPipe Face Mesh...")
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Initialize face parts table
            print("  - Setting up face parts mapping...")
            table = {
                'hair': 17,
                'upper_lip': 12,
                'lower_lip': 13,
                'eye_part_left': 4,
                'eye_part_right': 5,
                'left_eyebrow': 2,
                'right_eyebrow': 3,
                'skin': 1,
            }
            
            models_loaded = True
            elapsed_time = time.time() - start_time
            print(f"✅ All models loaded successfully in {elapsed_time:.2f} seconds!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            models_loaded = False
            return False

def ensure_models_loaded():
    """Ensure models are loaded before processing"""
    if not models_loaded:
        success = load_models()
        if not success:
            raise Exception("Failed to load required models")

# Ensure that truncated files are loaded properly
ImageFile.LOAD_TRUNCATED_IMAGES = True

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": models_loaded,
        "message": "Server is running" if models_loaded else "Server is running, models will load on first request"
    }), 200

# Models status endpoint
@app.route("/models/status", methods=["GET"])
def models_status():
    return jsonify({
        "models_loaded": models_loaded,
        "available_endpoints": [
            "/lips", "/eyeliner", "/hair", "/eyeshadow", 
            "/eyebrows", "/concealer", "/blush", "/iris"
        ]
    }), 200

# Preload models endpoint
@app.route("/models/preload", methods=["POST"])
def preload_models():
    try:
        success = load_models()
        if success:
            return jsonify({"message": "Models loaded successfully"}), 200
        else:
            return jsonify({"error": "Failed to load models"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to load models: {str(e)}"}), 500

@app.route("/test", methods=["GET"])
def test_endpoint():
    return jsonify({
        "message": "API is working!",
        "models_loaded": models_loaded,
        "python_version": "3.8.10"
    }), 200

def hex_to_rgb(hex_code):
    """Convert hex color to RGB"""
    hex_code = hex_code.lstrip('#')
    return [int(hex_code[i:i+2], 16) for i in (0, 2, 4)]

def correct_image_orientation(image_pil):
    """Correct image orientation"""
    try:
        exif = image_pil._getexif()
        if exif:
            orientation_tag = 274
            orientation = exif.get(orientation_tag)
            if orientation == 3:
                image_pil = image_pil.rotate(180, expand=True)
            elif orientation == 6:
                image_pil = image_pil.rotate(270, expand=True)
            elif orientation == 8:
                image_pil = image_pil.rotate(90, expand=True)
    except Exception:
        pass
    return image_pil.convert("RGB")

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["image"]
    filename = file.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    return jsonify({"imageUrl": f"/uploads/{filename}"})

@app.route('/iris/colors', methods=['GET', 'OPTIONS'])
def get_iris_colors():
    """Get available iris colors"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        print("Fetching iris colors...")
        colors = [
            {"name": "Blue", "value": "bluegreeniris"},
            {"name": "Green", "value": "greeniris"},
            {"name": "Brown", "value": "browniris"},
            {"name": "Hazel", "value": "hazeliris"},
            {"name": "Gray", "value": "grayiris"}
        ]
        print(f"Returning {len(colors)} iris colors")
        return jsonify({'colors': colors}), 200
    except Exception as e:
        print(f"Error in get_iris_colors: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/iris', methods=['POST', 'OPTIONS'])
def apply_iris_api():
    """Apply iris color change"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        print(f"Received iris request: {type(data)}")
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        print("Processing iris request...")
        # For now, just return the original image
        response = jsonify({'image': data['image']})
        print("Iris request processed successfully")
        return response, 200
        
    except Exception as e:
        print(f"Error in iris_api: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/lips', methods=['POST', 'OPTIONS'])
def lips_api():
    """Apply lipstick"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        print(f"Received lips request: {type(data)}")
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        print("Processing lips request...")
        # For now, just return the original image
        response = jsonify({'image': data['image']})
        print("Lips request processed successfully")
        return response, 200
        
    except Exception as e:
        print(f"Error in lips_api: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/hair', methods=['POST'])
def hair_api():
    """Apply hair color"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # For now, just return the original image
        return jsonify({'image': data['image']}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/eyeliner', methods=['POST'])
def eyeliner_api():
    """Apply eyeliner"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # For now, just return the original image
        return jsonify({'image': data['image']}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/eyeshadow', methods=['POST'])
def eyeshadow_api():
    """Apply eyeshadow"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # For now, just return the original image
        return jsonify({'image': data['image']}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/eyebrows', methods=['POST'])
def eyebrows_api():
    """Apply eyebrow color"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # For now, just return the original image
        return jsonify({'image': data['image']}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/concealer', methods=['POST'])
def concealer_api():
    """Apply concealer"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # For now, just return the original image
        return jsonify({'image': data['image']}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/blush', methods=['POST'])
def blush_api():
    """Apply blush"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # For now, just return the original image
        return jsonify({'image': data['image']}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Handle OPTIONS requests for all routes
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "http://localhost:3000")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Makeover API is running!",
        "status": "healthy",
        "endpoints": ["/health", "/models/status", "/lips", "/iris", "/hair", "/upload"]
    }), 200

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print("⚠️  Models will be loaded lazily on first request")
    print("📍 Server will be available at: http://localhost:4999")
    
    try:
        app.run(debug=False, host='0.0.0.0', port=4999, threaded=True)
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        import traceback
        traceback.print_exc()
