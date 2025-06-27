from flask import Flask, request, jsonify, send_file, make_response
import base64
import numpy as np
import io
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from flask_cors import CORS
import cv2
import os
import time
import threading

# Heavy imports will be loaded lazily
dlib = None
mp = None
pillow_heif = None
ImageCms = None

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://192.168.1.16:3000"])

# Global variables for lazy loading
detector = None
predictor = None
face_mesh = None
mp_face_mesh = None
table = None
models_loaded = False
loading_lock = threading.Lock()

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

def load_heavy_libraries():
    """Load heavy libraries only when needed"""
    global dlib, mp, pillow_heif, ImageCms
    
    if dlib is None:
        try:
            print("  - Importing dlib...")
            import dlib as dlib_module
            dlib = dlib_module
        except Exception as e:
            print(f"❌ Failed to import dlib: {e}")
            return False
    
    if mp is None:
        try:
            print("  - Importing MediaPipe...")
            import mediapipe as mp_module
            mp = mp_module
        except Exception as e:
            print(f"❌ Failed to import MediaPipe: {e}")
            return False
    
    if pillow_heif is None:
        try:
            print("  - Importing pillow_heif...")
            import pillow_heif as heif_module
            from pillow_heif import register_heif_opener
            from PIL import ImageCms as cms_module
            pillow_heif = heif_module
            ImageCms = cms_module
            register_heif_opener()
        except Exception as e:
            print(f"⚠️  pillow_heif not available: {e}")
            # This is optional, so we don't fail
    
    return True

def load_models():
    """Load all heavy models once when first needed"""
    global detector, predictor, face_mesh, mp_face_mesh, table, models_loaded
    
    with loading_lock:
        if models_loaded:
            return
        
        print("🔄 Loading ML models... This may take a moment.")
        start_time = time.time()
        
        try:
            # First, load heavy libraries
            if not load_heavy_libraries():
                raise Exception("Failed to load required libraries")
            
            # Load dlib models
            print("  - Loading dlib face detector...")
            detector = dlib.get_frontal_face_detector()
            
            print("  - Loading dlib shape predictor...")
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
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            raise e

def ensure_models_loaded():
    """Ensure models are loaded before processing"""
    if not models_loaded:
        load_models()

def hex_to_rgb(hex_code):
    # Strip any leading '#' and convert to RGB tuple
    hex_code = hex_code.lstrip('#')
    return [int(hex_code[i:i+2], 16) for i in (0, 2, 4)]

# Ensure that truncated files are loaded properly
ImageFile.LOAD_TRUNCATED_IMAGES = True

def convert_heic_to_jpeg(image_data):
    try:
        if pillow_heif is None:
            load_heavy_libraries()
        
        heif_file = pillow_heif.read_heif(io.BytesIO(image_data).read())
        heif_pil_image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw"
        )
        with io.BytesIO() as output:
            heif_pil_image.save(output, format="JPEG")
            jpeg_data = output.getvalue()
            
        return jpeg_data
    except Exception as e:
        raise ValueError(f"Failed to convert HEIC/HEIF to JPEG: {e}")

def correct_image_orientation(image_pil):
    """
    Corrects image orientation and ensures consistent color profile (sRGB).
    """
    # ✅ Correct orientation
    try:
        exif = image_pil._getexif()
        if exif:
            orientation_tag = 274  # EXIF orientation tag
            orientation = exif.get(orientation_tag)

            if orientation == 3:
                image_pil = image_pil.rotate(180, expand=True)
            elif orientation == 6:
                image_pil = image_pil.rotate(270, expand=True)  # 90 degrees clockwise
            elif orientation == 8:
                image_pil = image_pil.rotate(90, expand=True)   # 90 degrees counterclockwise
    except Exception:
        pass  # Ignore if no EXIF data

    # ✅ Handle HEIF/HEIC color profile
    if pillow_heif is not None:
        # Only process if pillow_heif is available
        if "icc_profile" in image_pil.info:
            icc_profile = image_pil.info.get("icc_profile")
            if icc_profile and ImageCms is not None:
                srgb_profile = ImageCms.createProfile("sRGB")
                image_pil = ImageCms.profileToProfile(
                    image_pil,
                    io.BytesIO(icc_profile),  # Original profile
                    srgb_profile,              # Convert to sRGB
                    outputMode="RGB"
                )

    # Convert to RGB for compatibility
    image_pil = image_pil.convert("RGB")

    return image_pil

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)  # Create uploads/ folder if not exists

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
            "/eyebrows", "/concealer", "/blush"
        ]
    }), 200

# Preload models endpoint (optional)
@app.route("/models/preload", methods=["POST"])
def preload_models():
    try:
        load_models()
        return jsonify({"message": "Models loaded successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to load models: {str(e)}"}), 500

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    filename = file.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)  # Save the file in uploads/
    
    # Return the publicly accessible image URL
    return jsonify({"imageUrl": f"/uploads/{filename}"})

@app.route('/lips', methods=['POST'])
def lips_api():
    # Ensure models are loaded
    ensure_models_loaded()
    
    # Lazy import heavy modules to avoid startup delays
    from test import evaluate
    from app import matte_lips, glossy_lips, apply_lip_liner
    
    data = request.get_json()

    lip_shade = None
    lip_liner_color = None  # Initialize liner shade
    
    lipstick_type = data.get("type", "matte").lower()
    print(f"lipstick type: {lipstick_type}")

    # Extract lipstick shade
    if "shade" in data:
        lip_shade = hex_to_rgb(data['shade'].upper())  
        print(f"Shade fetched: {data['shade']} -> RGB: {lip_shade}")
    # Extract liner shade
    if "liner_shade" in data:  
        lip_liner_color = hex_to_rgb(data['liner_shade'].upper())
        print(f"Liner Shade fetched: {data['liner_shade']} -> RGB: {lip_liner_color}")
    
    data = data['image']
    
    if not data:
        return jsonify({'error': 'No image provided'}), 400

    # Decode the base64 image
    image_data = base64.b64decode(data.split(',')[1])

    try:
        image_pil = Image.open(io.BytesIO(image_data))
        image_pil = correct_image_orientation(image_pil)  # Updated function
        image_pil = image_pil.convert("RGB")
    except Exception as e:
        return jsonify({'error': 'Failed to process image format', 'details': str(e)}), 400

    ori = np.array(image_pil)
    h, w, _ = ori.shape
    image = ori.copy()

    # Resize image for processing
    max_dim = 1024
    scale_factor = min(1, max_dim / max(h, w))
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    image = cv2.resize(ori, (new_w, new_h))

    # Detect face
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_image)

    if len(faces) == 0:
        return jsonify({'error': 'No face detected in the image'}), 400

    largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
    x, y, width, height = largest_face.left(), largest_face.top(), largest_face.width(), largest_face.height()

    # Expand bounding box
    padding = int(0.2 * height)
    y_start, y_end = max(0, y - padding), min(image.shape[0], y + height + padding)
    x_start, x_end = max(0, x - padding), min(image.shape[1], x + width + padding)

    cropped_face = image[y_start:y_end, x_start:x_end]
    resized_face = cv2.resize(cropped_face, (512, 512))

    # Lip segmentation
    cp = 'cp/79999_iter.pth'
    parsing = evaluate(resized_face, cp)
    parsing = cv2.resize(parsing, cropped_face.shape[:2], interpolation=cv2.INTER_NEAREST)

    # Apply lipstick
    if lipstick_type == "matte":
        processed_face = matte_lips(cropped_face, parsing, table['upper_lip'], lip_shade)
        processed_face = matte_lips(processed_face, parsing, table['lower_lip'], lip_shade)
    elif lipstick_type == "glossy":
        processed_face = glossy_lips(cropped_face, parsing, table['upper_lip'], lip_shade)
        processed_face = glossy_lips(processed_face, parsing, table['lower_lip'], lip_shade)

    # Apply lip liner if shade is provided
    if lip_liner_color:
        processed_face = apply_lip_liner(cropped_face, parsing, table['upper_lip'], table['lower_lip'], lip_liner_color)

    # Replace processed lips back into image
    image[y_start:y_end, x_start:x_end] = processed_face

    # Resize back to original dimensions
    processed_image = cv2.resize(image, (w, h))

    # Convert processed image to base64
    processed_image_pil = Image.fromarray(processed_image)
    buffered = io.BytesIO()
    processed_image_pil.save(buffered, format="PNG")
    processed_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({'image': processed_image_str}), 200

@app.route('/eyeliner', methods=['POST'])
def apply_eyeliner_api():
    # Ensure models are loaded
    ensure_models_loaded()
    
    # Lazy import heavy modules
    from app import get_face_landmarks, get_upper_eyelid_landmarks, apply_eyeliner, resize_image
    
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({'error': 'No image provided'}), 400

    transparency = data.get("transparency", 0.85)
    style = data.get("style", "wing").lower()

    if "shade" in data:
        eyeliner_shade = hex_to_rgb(data['shade'].upper())
    else:
        eyeliner_shade = [0, 0, 0]  # Black

    try:
        image_data = data["image"].split(",")[-1]
        image = decode_base64_image(image_data)
        if image is None or image.size == 0:
            raise ValueError("Decoded image is empty or invalid")
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    processed_image = process_image(image, eyeliner_shade, transparency, style=style)

    processed_image_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    processed_image_pil.save(buffered, format="PNG")
    processed_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({'image': processed_image_str}), 200

def process_image(image, eyeliner_color=(0, 0, 0), transparency=0.85, style='wing', attempts=5):
    # Import functions locally to avoid global import issues
    from app import resize_image, get_face_landmarks, get_upper_eyelid_landmarks, apply_eyeliner
    
    image = resize_image(image)
    face_landmarks = None
    for _ in range(attempts):
        face_landmarks = get_face_landmarks(image)
        if face_landmarks:
            break
        cv2.waitKey(100)
    if face_landmarks is None:
        print("No face detected after multiple attempts.")
        return image

    left_eye, right_eye = get_upper_eyelid_landmarks(image, face_landmarks)

    return apply_eyeliner(image, left_eye, right_eye, eyeliner_color, transparency, style=style)

def decode_base64_image(image_data):
    """Decodes base64 image and corrects orientation."""
    try:
        image_bytes = base64.b64decode(image_data)
        image_pil = Image.open(io.BytesIO(image_bytes))

        # ✅ Correct orientation before converting to NumPy
        image_pil = correct_image_orientation(image_pil)

        # Convert PIL image to OpenCV format
        image = np.array(image_pil)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if image is None or image.size == 0:
            raise ValueError("Decoded image is None or empty")

        return image

    except Exception as e:
        raise ValueError("Invalid image data or format")

# Hair
@app.route('/hair', methods=['POST'])
def hair_colour():
    # Ensure models are loaded
    ensure_models_loaded()
    
    # Lazy import heavy modules
    from test import evaluate
    from app import hair
    
    data = request.get_json()

    hair_shade = None

    # Extract hair shade
    if "shade" in data:
        hair_shade = hex_to_rgb(data['shade'].upper())
        print(f"Shade fetched: {data['shade']} -> RGB: {hair_shade}")

    data_img = data.get('image')
    if not data_img:
        return jsonify({'error': 'No image provided'}), 400

    # Decode the base64 image
    image_data = base64.b64decode(data_img.split(',')[1])
    try:
        image_pil = Image.open(io.BytesIO(image_data))
        image_pil = correct_image_orientation(image_pil)
        image_pil = image_pil.convert("RGB")
    except Exception as e:
        return jsonify({'error': 'Failed to process image format', 'details': str(e)}), 400

    ori = np.array(image_pil)
    h, w, _ = ori.shape
    image = ori.copy()

    # Resize image for processing
    image = cv2.resize(image, (1024, 1024))

    cp = 'cp/79999_iter.pth'
    parsing = evaluate(image, cp)
    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

    # Apply hair color if shade is provided
    if hair_shade is not None:
        image = hair(image, parsing, part=table['hair'], color=hair_shade)
    else:
        image = hair(image, parsing, part=table['hair'], color=[92, 21, 20])  # Default color

    # Resize back to original dimensions
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert processed image back to base64
    processed_image = Image.fromarray(image)
    buffered = io.BytesIO()
    processed_image.save(buffered, format="PNG")
    processed_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({'image': processed_image_str}), 200

@app.route('/eyeshadow', methods=['POST'])
def apply_eyeshadow_api():
    # Ensure models are loaded
    ensure_models_loaded()
    
    # Lazy import heavy modules
    from app import create_eyeshadow_mask, apply_eyeshadow
    
    data = request.get_json()
    
    eyeshadow_color = None
    if "shade" in data:
        eyeshadow_color = hex_to_rgb(data['shade'].upper())  # Convert HEX to RGB
        print(f"Shade fetched: {data['shade']} -> RGB: {eyeshadow_color}")
    
    data = data['image']
    if not data:
        return jsonify({'error': 'No image provided'}), 400

    # Decode the base64 image
    image_data = base64.b64decode(data.split(',')[1])
    
    try:
        image_pil = Image.open(io.BytesIO(image_data))
        image_pil = correct_image_orientation(image_pil)
        image_pil_format = image_pil.format
        print(f"Image format detected: {image_pil_format}")

        image_pil = image_pil.convert("RGB")
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': 'Failed to process image format', 'details': str(e)}), 400
    
    ori = np.array(image_pil)
    h, w, _ = ori.shape  # Store original dimensions
    print(f"Original Image Dimensions: {w}x{h}")
    
    image = ori.copy()
    max_dim = 1024
    scale_factor = min(1, max_dim / max(h, w))
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    print(f"Resized Image Dimensions: {new_w}x{new_h}")
    
    image = cv2.resize(ori, (new_w, new_h))
    
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        print("Face detected, applying eyeshadow...")
        for face_landmarks in results.multi_face_landmarks:
            mask_left, mask_right = create_eyeshadow_mask(image, face_landmarks)
            image = apply_eyeshadow(image, mask_left, mask_right, color=eyeshadow_color)
            print("Eyeshadow applied successfully.")
    else:
        print("No face detected in the image.")
    
    processed_image = cv2.resize(image, (w, h))  # Resize back to original dimensions
    
    processed_image_pil = Image.fromarray(processed_image)
    buffered = io.BytesIO()
    processed_image_pil.save(buffered, format="PNG")
    processed_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    print("Image processing complete. Returning response.")
    return jsonify({'image': processed_image_str}), 200

@app.route('/eyebrows', methods=['POST'])
def apply_eyebrow_api():
    # Ensure models are loaded
    ensure_models_loaded()
    
    # Lazy import heavy modules
    from test import evaluate
    from app import apply_eyebrow_shade
    
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        image_pil = Image.open(io.BytesIO(image_data))
        image_pil = correct_image_orientation(image_pil)
        image_pil = image_pil.convert("RGB")

    except Exception as e:
        return jsonify({'error': 'Failed to process image', 'details': str(e)}), 400

    ori = np.array(image_pil)
    h, w, _ = ori.shape
    image = ori.copy()

    # Resize image for processing
    max_dim = 1024
    scale_factor = min(1, max_dim / max(h, w))
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    image = cv2.resize(ori, (new_w, new_h))

    # Convert image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_image)

    if len(faces) == 0:
        return jsonify({'error': 'No face detected in the image'}), 400

    # Get the largest detected face
    largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
    x, y, width, height = largest_face.left(), largest_face.top(), largest_face.width(), largest_face.height()

    # Expand the bounding box slightly
    padding = int(0.2 * height)
    y_start = max(0, y - padding)
    y_end = min(image.shape[0], y + height + padding)
    x_start = max(0, x - padding)
    x_end = min(image.shape[1], x + width + padding)

    cropped_face = image[y_start:y_end, x_start:x_end]

    # Resize cropped face
    resized_face = cv2.resize(cropped_face, (512, 512))

    # Load segmentation model and get eyebrow parsing
    cp = 'cp/79999_iter.pth'
    parsing = evaluate(resized_face, cp)
    parsing = cv2.resize(parsing, cropped_face.shape[:2], interpolation=cv2.INTER_NEAREST)

    processed_face = cropped_face.copy()

    # Apply eyebrow shading if shade is provided
    if "shade" in data:
        eyebrow_shade = hex_to_rgb(data['shade'].upper())  
        print(eyebrow_shade)
        processed_face = apply_eyebrow_shade(processed_face, parsing, table['left_eyebrow'], table['right_eyebrow'], color=eyebrow_shade)

    # Replace processed region back into original image
    image[y_start:y_end, x_start:x_end] = processed_face

    # Resize back to original dimensions
    processed_image = cv2.resize(image, (w, h))

    # Convert processed image back to base64
    processed_image_pil = Image.fromarray(processed_image)
    buffered = io.BytesIO()
    processed_image_pil.save(buffered, format="PNG")
    processed_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({'image': processed_image_str}), 200

@app.route('/concealer', methods=['POST'])
def apply_concealer_api():
    # Ensure models are loaded
    ensure_models_loaded()
    
    # Lazy import heavy modules
    from app import create_concealer_mask, apply_concealer
    
    data = request.get_json()
    
    concealer_color = None
    if "shade" in data:
        concealer_color = hex_to_rgb(data['shade'].upper())  # Convert HEX to RGB
        print(f"Shade fetched: {data['shade']} -> RGB: {concealer_color}")
    
    data = data['image']
    if not data:
        return jsonify({'error': 'No image provided'}), 400

    # Decode the base64 image
    image_data = base64.b64decode(data.split(',')[1])
    
    try:
        image_pil = Image.open(io.BytesIO(image_data))
        image_pil = correct_image_orientation(image_pil)
        image_pil_format = image_pil.format
        print(f"Image format detected: {image_pil_format}")

        image_pil = image_pil.convert("RGB")
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': 'Failed to process image format', 'details': str(e)}), 400
    
    ori = np.array(image_pil)
    h, w, _ = ori.shape  # Store original dimensions
    print(f"Original Image Dimensions: {w}x{h}")
    
    image = ori.copy()
    max_dim = 1024
    scale_factor = min(1, max_dim / max(h, w))
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    print(f"Resized Image Dimensions: {new_w}x{new_h}")
    
    image = cv2.resize(ori, (new_w, new_h))
    
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        print("Face detected, applying concealer...")
        for face_landmarks in results.multi_face_landmarks:
            mask_concealer = create_concealer_mask(image, face_landmarks)
            image = apply_concealer(image, mask_concealer, concealer_color, intensity=0.6)
    else:
        print("No face detected in the image.")
    
    processed_image = cv2.resize(image, (w, h))  # Resize back to original dimensions
    
    processed_image_pil = Image.fromarray(processed_image)
    buffered = io.BytesIO()
    processed_image_pil.save(buffered, format="PNG")
    processed_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    print("Image processing complete. Returning response.")
    return jsonify({'image': processed_image_str}), 200

@app.route('/blush', methods=['POST'])
def apply_blush_api():
    # Ensure models are loaded
    ensure_models_loaded()
    
    # Lazy import heavy modules
    from app import get_cheek_masks, apply_blush
    
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({'error': 'No image provided'}), 400

    image_data = data["image"].split(",")[-1]
    blush_hex = data.get("shade", "#FFA07A")

    # 🟨 Debug: Print received hex shade
    print(f"[DEBUG] Received hex shade: {blush_hex}")

    try:
        ori = decode_base64_image(image_data)
        if ori is None or ori.size == 0:
            return jsonify({'error': 'Invalid image'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    h, w = ori.shape[:2]

    # === Resize image for MediaPipe face mesh detection ===
    max_dim = 1024
    scale_factor = min(1, max_dim / max(h, w))
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    image = cv2.resize(ori, (new_w, new_h))

    # Convert resized image to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return jsonify({'error': 'No face detected'}), 400

    blush_color = hex_to_rgb(blush_hex)
    print(f"[DEBUG] Converted RGB shade: {blush_color}")

    for i, face_landmarks in enumerate(results.multi_face_landmarks):
        print(f"[DEBUG] Processing face {i+1} with {len(face_landmarks.landmark)} landmarks.")
        # If needed, scale landmarks back to original size
        def scale_point(point, scale_x, scale_y):
            return int(point.x * new_w / scale_x), int(point.y * new_h / scale_y)

        # Get cheek masks on resized image
        mask_left, mask_right = get_cheek_masks(image, face_landmarks)

        # Resize masks to match original image size
        mask_left = cv2.resize(mask_left, (w, h))
        mask_right = cv2.resize(mask_right, (w, h))

        # Apply blush on original image with upscaled masks
        ori = apply_blush(ori, mask_left, mask_right, color=blush_color, intensity=1.7)

    # Convert processed image to base64 PNG
    processed_image_pil = Image.fromarray(cv2.cvtColor(ori, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    processed_image_pil.save(buffered, format="PNG")
    processed_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({'image': processed_image_str}), 200

# Iris Color
@app.route('/iris/colors', methods=['GET'])
def get_iris_colors():
    """Get available iris colors"""
    try:
        from app import get_available_iris_colors
        colors = get_available_iris_colors()
        return jsonify({'colors': colors}), 200
    except Exception as e:
        print(f"Error getting iris colors: {e}")
        return jsonify({'error': 'Failed to get iris colors'}), 500

@app.route('/iris', methods=['POST'])
def apply_iris_api():
    # Lazy import heavy modules
    from app import change_iris_color_advanced
    
    data = request.get_json()
    
    iris_color_name = data.get('iris_color', 'bluegreeniris')
    opacity = data.get('opacity', 0.75)
    
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # Decode the base64 image
    image_data = base64.b64decode(data['image'].split(',')[1])
    
    try:
        image_pil = Image.open(io.BytesIO(image_data))
        image_pil = correct_image_orientation(image_pil)
        image_pil = image_pil.convert("RGB")
    except Exception as e:
        return jsonify({'error': 'Failed to process image format', 'details': str(e)}), 400
    
    ori = np.array(image_pil)
    h, w, _ = ori.shape
    image = ori.copy()

    # Apply iris color change
    try:
        image = change_iris_color_advanced(image, iris_color_name, opacity)
        print(f"Applied iris color: {iris_color_name} with opacity: {opacity}")
    except Exception as e:
        print(f"Error applying iris color: {e}")
        return jsonify({'error': f'Failed to apply iris color: {str(e)}'}), 500
    
    # Convert processed image back to base64
    processed_image_pil = Image.fromarray(image)
    buffered = io.BytesIO()
    processed_image_pil.save(buffered, format="PNG")
    processed_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return jsonify({'image': processed_image_str}), 200

if __name__ == '__main__':
    print("🚀 Starting Makeover API Server...")
    print("📍 Server will be available at: http://localhost:4999")
    print("💡 Models will load automatically when first needed")
    print("🔗 CORS enabled for: http://localhost:3000")
    
    try:
        # Start the Flask server without any global initialization
        app.run(host='0.0.0.0', port=4999, debug=True, threaded=True)
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()