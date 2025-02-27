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
from app import matte_lips, glossy_lips,hair, get_cheek_masks, apply_blush, draw_eyeliner, get_eye_interpolated_points, create_eyeshadow_mask,apply_eyeshadow
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
    
    lip_shade = None;
    # Handle the lip color shade
    if "shade" in data:
        lip_shade = hex_to_rgb(data['shade'].upper())  # Get RGB format for shade
        print(lip_shade)
      # Default shade 
    
    lipstick_type = data.get("type", "matte").lower()

    data = data['image']
    if not data:
        return jsonify({'error': 'No image provided'}), 400

    # Decode the base64 image
    image_data = base64.b64decode(data.split(',')[1])  # Ignore the data:image/jpeg;base64 part

    try:

        image_pil = Image.open(io.BytesIO(image_data))
        image_pil = correct_image_orientation(image_pil)
        image_pil_format = image_pil.format
       
        if image_pil_format in ["HEIC", "HEIF"]:
            original_format = "HEIC"
            image_data = convert_heic_to_jpeg(image_data)
            image_pil = Image.open(io.BytesIO(image_data))

        else:
            original_format = None  

        image_pil = image_pil.convert("RGB")  # Ensure RGB format

    except Exception as e:
        return jsonify({'error': 'Failed to process image format', 'details': str(e)}), 400

    ori = np.array(image_pil)  # Keep the original image for resizing later
    h, w, _ = ori.shape  # Store original dimensions
    image = ori.copy()

    max_dim = 1024
    scale_factor = min(1, max_dim / max(h, w))
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    image = cv2.resize(ori, (new_w, new_h))

    # Process the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_image)

    if len(faces) == 0:
        return jsonify({'error': 'No face detected in the image'}), 400

    # Get the bounding box of the largest face detected
    largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
    x, y, width, height = largest_face.left(), largest_face.top(), largest_face.width(), largest_face.height()

    # Expand the bounding box slightly to include the lips region better
    padding = int(0.2 * height)
    y_start = max(0, y - padding)
    y_end = min(image.shape[0], y + height + padding)
    x_start = max(0, x - padding)
    x_end = min(image.shape[1], x + width + padding)

    cropped_face = image[y_start:y_end, x_start:x_end]

    # Resize cropped face to a fixed size for processing
    resized_face = cv2.resize(cropped_face, (512, 512))

    # Evaluate segmentation
    cp = 'cp/79999_iter.pth'
    parsing = evaluate(resized_face, cp)
    parsing = cv2.resize(parsing, cropped_face.shape[:2], interpolation=cv2.INTER_NEAREST)


    # Apply lipstick based on type
    if lipstick_type == "matte":
        processed_face = matte_lips(cropped_face, parsing, part=table['upper_lip'], color=lip_shade)
        processed_face = matte_lips(processed_face, parsing, part=table['lower_lip'], color=lip_shade)
    elif lipstick_type == "glossy":
        processed_face = glossy_lips(cropped_face, parsing,  part=table['upper_lip'], color=lip_shade)
        processed_face = glossy_lips(processed_face, parsing,  part=table['lower_lip'], color=lip_shade)

    # Replace the processed region back into the original image
    image[y_start:y_end, x_start:x_end] = processed_face

    # Resize back to original dimensions
    processed_image = cv2.resize(image, (w, h))  # Resize to match original dimensions

    # Convert processed image back to base64
    if original_format == "HEIC":
        processed_image_data = convert_heic_to_jpeg_and_back(processed_image)  # Convert to HEIC format
        processed_image_str = base64.b64encode(processed_image_data).decode("utf-8")
    else:
        # Convert processed image to PNG if it's not from HEIC
        processed_image_pil = Image.fromarray(processed_image)
        buffered = io.BytesIO()
        processed_image_pil.save(buffered, format="PNG")
        processed_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({'image': processed_image_str}), 200



# Eyeliner 

@app.route('/eyeliner', methods=['POST'])
def apply_eyeliner_api():
    data = request.get_json()
    
    # Extract eyeliner parameters
    color_hex = data.get('color', '#000000')
    thickness = data.get('thickness', 6)
    transparency = data.get('transparency', 0.8)
    
    # Validate image input
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    
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
    
    # Resize image for processing
    max_dim = 1024
    scale_factor = min(1, max_dim / max(h, w))
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    image = cv2.resize(ori, (new_w, new_h))
    
    # Detect face landmarks
    results = face_mesh.process(image)
    if not results.multi_face_landmarks:
        return jsonify({'error': 'No face landmarks detected'}), 400
    
    color = tuple(int(color_hex.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))[::-1]
    
    for face_landmarks in results.multi_face_landmarks:
        interp_pts = get_eye_interpolated_points(image, face_landmarks)
        if interp_pts:
            image = draw_eyeliner(image, interp_pts, color=color, thickness=thickness, transparency=transparency)
    
    # Resize back to original dimensions
    processed_image = cv2.resize(image, (w, h))
    
    # Convert back to base64
    processed_image_pil = Image.fromarray(processed_image)
    buffered = io.BytesIO()
    processed_image_pil.save(buffered, format="PNG")
    processed_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return jsonify({'image': processed_image_str}), 200




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

def decode_base64_image(image_data):
    image_data = base64.b64decode(image_data.split(',')[1])  # Decode base64
    try:
        image_pil = Image.open(io.BytesIO(image_data))
        image_pil = correct_image_orientation(image_pil)  # Fix orientation
        image_pil_format = image_pil.format

        if image_pil_format in ["HEIC", "HEIF"]:
            image_data = convert_heic_to_jpeg(image_data)  # Convert HEIC to JPEG
            image_pil = Image.open(io.BytesIO(image_data))

        image_pil = image_pil.convert("RGB")  # Ensure RGB format
        image = np.array(image_pil)
        return image

    except Exception as e:
        raise ValueError(f"Failed to decode and process image: {str(e)}")


@app.route('/eyeshadow', methods=['POST'])
def apply_eyeshadow_api():
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({'error': 'No image provided'}), 400

    image_data = data["image"]
    shade_hex = data.get("shade", "#800080")  # Default: Purple (#800080)

    try:
        image = decode_base64_image(image_data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # Convert HEX to RGB
    eyeshadow_color = hex_to_rgb(shade_hex)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mask_left, mask_right = create_eyeshadow_mask(image, face_landmarks)
            image = apply_eyeshadow(image, mask_left, mask_right, color=eyeshadow_color)  # Apply color

    # Convert processed image to base64
    processed_image_pil = Image.fromarray(image)
    buffered = io.BytesIO()
    processed_image_pil.save(buffered, format="PNG")
    processed_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

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