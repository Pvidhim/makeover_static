from flask import Flask, request, jsonify, send_file, make_response
import base64
import numpy as np
import io
from PIL import Image, ImageFile, ExifTags
import matplotlib.pyplot as plt
from flask_cors import CORS
import cv2
import dlib
from test import evaluate
from app import apply_eyeliner, apply_lip_liner, matte_lips, glossy_lips,hair, get_cheek_masks, apply_blush,  create_eyeshadow_mask,apply_eyeshadow,get_upper_eyelid_landmarks,get_face_landmarks, resize_image
import mediapipe as mp
import os
import pillow_heif
import logging

app = Flask(__name__)
CORS(app, origins=["http://192.168.1.21:3000"])

PREDICTOR_PATH = r"C:\Users\DEEKSHITA\OneDrive\Desktop\Visuareal\Makeover-main\shape_predictor_68_face_landmarks.dat"

# Initialize face and landmark detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def hex_to_rgb(hex_code):
    # Strip any leading '#' and convert to RGB tuple
    hex_code = hex_code.lstrip('#')
    return [int(hex_code[i:i+2], 16) for i in (0, 2, 4)]

# Ensure that truncated files are loaded properly
ImageFile.LOAD_TRUNCATED_IMAGES = True

def convert_heic_to_jpeg(image_data):
    try:
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

def convert_heic_to_jpeg_and_back(image_array):
    try:
        # Convert the image array back to a PIL Image
        image_pil = Image.fromarray(image_array)
        
        # Save the PIL Image to HEIC format using pillow-heif
        with io.BytesIO() as output:

            # Configure HEIC settings if necessary
            pillow_heif.register_avif()  # Register AVIF/HEIC codecs
            pillow_heif.register_heif_opener()  # Ensure HEIF codecs are active
            
            # Save the image in HEIC format
            pillow_heif.write_heif(image_pil, output, format="HEIC", quality=90)
            
            heic_data = output.getvalue()  # Get HEIC-encoded bytes
            
        return heic_data
    
    except Exception as e:
        raise ValueError(f"Failed to convert image back to HEIC: {e}")


def correct_image_orientation(image_pil):
        
    exif = image_pil._getexif()
    if exif:
        orientation_tag = 274  # Orientation tag in EXIF data
        orientation = exif.get(orientation_tag)
            
        if orientation == 3:
            image_pil = image_pil.rotate(180, expand=True)
        elif orientation == 6:
            image_pil = image_pil.rotate(270, expand=True)  # Rotate 90 degrees clockwise
        elif orientation == 8:
            image_pil = image_pil.rotate(90, expand=True)  # Rotate 90 degrees counterclockwise
   
    return image_pil


@app.route('/lips', methods=['POST'])
def upload_image():
    data = request.get_json()

    lip_shade = None
    lip_liner_color = None  # Initialize liner shade
    
    # Extract lipstick shade
    if "shade" in data:
        lip_shade = hex_to_rgb(data['shade'].upper())  

    # Extract liner shade
    if "liner_shade" in data:  
        lip_liner_color = hex_to_rgb(data['liner_shade'].upper())

    lipstick_type = data.get("type", "matte").lower()
    data = data['image']
    
    if not data:
        return jsonify({'error': 'No image provided'}), 400

    # Decode the base64 image
    image_data = base64.b64decode(data.split(',')[1])
    
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
    """API endpoint for applying eyeliner."""
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({'error': 'No image provided'}), 400

    
    # Extract color and transparency values
    transparency = data.get("transparency", 0.85)  # Adjusted default transparency

    if "shade" in data:
        eyeliner_shade = hex_to_rgb(data['shade'].upper())  # Get RGB format for shade
        print(eyeliner_shade)
    else:
        eyeliner_shade = [0, 0, 0]  # Default to black if no shade provided
    
    try:
        # Decode base64 image
        image_data = data["image"].split(",")[-1]  # Remove data URI prefix if present
        image = decode_base64_image(image_data)
        if image is None or image.size == 0:
            raise ValueError("Decoded image is empty or invalid")
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # Process image
    processed_image = process_image(image, eyeliner_shade, transparency)

    # Convert processed image to base64
    processed_image_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    processed_image_pil.save(buffered, format="PNG")
    processed_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({'image': processed_image_str}), 200

def process_image(image, eyeliner_color=(0, 0, 0), transparency=0.85, attempts=5):
    """Processes the image by applying eyeliner using face landmarks."""
    image = resize_image(image)  # Resize for consistency
    
    # Try multiple attempts to get face landmarks
    face_landmarks = None
    for _ in range(attempts):
        face_landmarks = get_face_landmarks(image)
        if face_landmarks:
            break
        cv2.waitKey(100)  # Small delay between attempts
    
    if face_landmarks is None:
        print("No face detected after multiple attempts.")
        return image

    left_eye, right_eye = get_upper_eyelid_landmarks(image, face_landmarks)
    image_with_eyeliner = apply_eyeliner(image, left_eye, right_eye, eyeliner_color, transparency)

    return image_with_eyeliner

def decode_base64_image(image_data):
    try:
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError("Decoded image is None or empty")
        return image
    except Exception as e:
        raise ValueError("Invalid image data or format")


# Hair
@app.route('/hair', methods=['POST'])
def hair_colour():
    # Get the base64 encoded image from the request
    data = request.json.get('image')
    if not data:
        return jsonify({'error': 'No image provided'}), 400

    # Decode the base64 image
    image_data = base64.b64decode(data.split(',')[1])  # Ignore the data:image/jpeg;base64 part
    image = np.array(Image.open(io.BytesIO(image_data)))

    # Process the image
    cp = 'cp/79999_iter.pth'
    ori = image.copy()
    h, w, _ = ori.shape
    image = cv2.resize(image, (1024, 1024))

    parsing = evaluate(image,cp)
    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

    image = hair(image, parsing, part=table['hair'], color=button['color'])

    # Convert processed image back to base64
    processed_image = Image.fromarray(image)
    buffered = io.BytesIO()
    processed_image.save(buffered, format="PNG")
    processed_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({'image': processed_image_str}), 200

#Blush
@app.route('/blush', methods=['POST'])
def apply_blush_api():
    # Get the base64 encoded image from the request
    data = request.json.get('image')
    if not data:
        return jsonify({'error': 'No image provided'}), 400

    # Decode the base64 image
    image_data = base64.b64decode(data.split(',')[1])  # Ignore the data:image/jpeg;base64 part
    image = np.array(Image.open(io.BytesIO(image_data)))

    # Convert image to RGB for processing with MediaPipe
    results = face_mesh.process(image)
    print('result')

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mask_left, mask_right = get_cheek_masks(image, face_landmarks)
            blush_color = (255, 105, 180)  # Example blush color (pink)
            image = apply_blush(image, mask_left, mask_right, color=button['color'])

    # Convert processed image back to base64
    processed_image = Image.fromarray(image)
    buffered = io.BytesIO()
    processed_image.save(buffered, format="PNG")
    processed_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({'image': processed_image_str}), 200

# def decode_base64_image(image_data):
#     image_data = base64.b64decode(image_data.split(',')[1])  # Decode base64
#     try:
#         image_pil = Image.open(io.BytesIO(image_data))
#         image_pil = correct_image_orientation(image_pil)  # Fix orientation
#         image_pil_format = image_pil.format

#         if image_pil_format in ["HEIC", "HEIF"]:
#             image_data = convert_heic_to_jpeg(image_data)  # Convert HEIC to JPEG
#             image_pil = Image.open(io.BytesIO(image_data))

#         image_pil = image_pil.convert("RGB")  # Ensure RGB format
#         image = np.array(image_pil)
#         return image

#     except Exception as e:
#         raise ValueError(f"Failed to decode and process image: {str(e)}")



@app.route('/eyeshadow', methods=['POST'])
def apply_eyeshadow_api():
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
    else:
        print("No face detected in the image.")
    
    processed_image = cv2.resize(image, (w, h))  # Resize back to original dimensions
    
    processed_image_pil = Image.fromarray(processed_image)
    buffered = io.BytesIO()
    processed_image_pil.save(buffered, format="PNG")
    processed_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    print("Image processing complete. Returning response.")
    return jsonify({'image': processed_image_str}), 200


if __name__ == '__main__':
    table = {
        'hair': 17,
        'upper_lip': 12,
        'lower_lip': 13,
        'eye_part_left' :4 ,
        'eye_part_right' : 5 ,
        'skin': 1,
        'eyes_u':3  
    }
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    button = {'color': [92, 21, 20]}  # Example color
    app.run(host='0.0.0.0', port=4999, debug=True)