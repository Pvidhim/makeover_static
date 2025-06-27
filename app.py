import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageColor,ImageEnhance
from skimage.filters import gaussian
from test import evaluate
from flask import jsonify
import mediapipe as mp
from scipy.interpolate import splprep, splev
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



mp_face_mesh = mp.solutions.face_mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


#matte lipstick
def matte_lips(image, parsing, part=12, color=[]):
    """
    Apply a natural matte lipstick effect while preserving lip texture.

    Args:
    image (np.array): Original image in BGR format.
    parsing (np.array): Segmented face parsing map.
    part (int): Face part index for lips.
    color (tuple): BGR color tuple for lip tint.
    highlight_width (int): Width of the highlight area in pixels.
    alpha (float): Blending factor (0-1), where 1 is full overlay.

    Returns:
    np.array: Image with matte lips applied.
    """
    # Resize parsing map to match image dimensions
    parsing = cv2.resize(parsing, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    lips_mask = (parsing == part).astype(np.uint8)
    
    # Soften lip mask edges for a natural transition
    lips_mask = cv2.GaussianBlur(lips_mask * 255, (7, 7), 3) / 255.0
    
    # Create a base lipstick layer with smooth blending
    lipstick_layer = np.full_like(image, color, dtype=np.uint8)
    base_lipstick = cv2.addWeighted(image, 0.5, lipstick_layer, 0.5, 0)
    
    # Apply the base lipstick to the image
    result_image = image.copy()
    for c in range(3):
        result_image[..., c] = (lips_mask * base_lipstick[..., c] + (1 - lips_mask) * image[..., c]).astype(np.uint8)
    
    return np.clip(result_image, 0, 255).astype(np.uint8)


#glossy lips
def glossy_lips(image, parsing, part=12, color=[0, 0, 255], gloss_intensity=2.0, transparency=0.45, lipstick_intensity=0.35, highlight_boost=3.0):
    """
    Apply a realistic glossy lipstick effect, ensuring gloss appears only on naturally glossy areas.
    """
    # Resize parsing mask to match image dimensions
    parsing = cv2.resize(parsing, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create a mask for lower lips only
    lips_mask = (parsing == part).astype(np.uint8)
    # lips_mask[:lips_mask.shape[0] // 2, :] = 0  # Remove upper lips
    lips_mask = cv2.GaussianBlur(lips_mask * 255, (7, 7), 3) / 255.0  # Soft edges

    # Convert image to grayscale to detect natural gloss
    gray_lips = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect natural **bright** areas where gloss should be applied
    highlight_mask = cv2.adaptiveThreshold(gray_lips, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -8)
    highlight_mask = cv2.GaussianBlur(highlight_mask, (5, 5), 2) / 255.0  # Smoothen highlights

    # Only keep **strongest** highlights (top 5% brightest areas)
    gloss_threshold = np.percentile(gray_lips[lips_mask > 0], 95)  # Top 5% brightest areas
    refined_gloss_mask = ((gray_lips > gloss_threshold) * lips_mask).astype(np.uint8)
    # refined_gloss_mask = cv2.GaussianBlur(refined_gloss_mask * 255, (5, 5), 2) / 255.0
    refined_gloss_mask = np.clip(refined_gloss_mask * highlight_boost, 0, 1)  # Boost gloss

    # Apply **stronger lipstick color**
    color_layer = np.full_like(image, color, dtype=np.uint8)
    colored_lips = cv2.addWeighted(image, 1 - lipstick_intensity, color_layer, lipstick_intensity, 0)

    # Blend lips naturally
    result_image = np.zeros_like(image)
    for c in range(3):
        result_image[..., c] = lips_mask * colored_lips[..., c] + (1 - lips_mask) * image[..., c]

    # Create **gloss layer ONLY for the real glossy areas**
    gloss_layer = np.full_like(image, [255, 255, 255], dtype=np.uint8)
    gloss_layer = (refined_gloss_mask[..., None] * gloss_layer * gloss_intensity).astype(np.uint8)

    # Blend gloss into the lips
    gloss_overlay = cv2.addWeighted(result_image.astype(np.float32), 1 - transparency, gloss_layer.astype(np.float32), transparency, 0).astype(np.uint8)

    # Apply **precise gloss effect only where the natural shine exists**
    for c in range(3):
        result_image[..., c] = (refined_gloss_mask * gloss_overlay[..., c] + (1 - refined_gloss_mask) * result_image[..., c]).astype(np.uint8)

    return np.clip(result_image, 0, 255).astype(np.uint8)


#lipliner
def apply_lip_liner(image, parsing, upper_lip, lower_lip, color=(120, 0, 60), liner_thickness=2, blur_intensity=3, liner_intensity=3):
    """
    Apply a precise and natural lip liner along the vermilion border while:
    - Avoiding the inner lip area.
    - Ensuring the liner is only on the outer edges.
    - Blending smoothly for a natural effect.
    """

    # Resize parsing mask if needed
    if parsing.shape[:2] != image.shape[:2]:
        parsing = cv2.resize(parsing, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create a binary mask for lips
    lips_mask = ((parsing == upper_lip) | (parsing == lower_lip)).astype(np.uint8) * 255

    # Detect the outer lip contour
    contours, _ = cv2.findContours(lips_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    liner_mask = np.zeros_like(image, dtype=np.uint8)

    if contours:
        # Draw the lip liner along the vermilion border
        cv2.drawContours(liner_mask, contours, -1, color, thickness=liner_thickness)

    # Convert to grayscale and blur for smooth blending
    liner_mask_gray = cv2.cvtColor(liner_mask, cv2.COLOR_BGR2GRAY)
    blurred_liner = cv2.GaussianBlur(liner_mask_gray, (blur_intensity * 2 + 1, blur_intensity * 2 + 1), blur_intensity)

    # Adjust liner intensity
    alpha = (blurred_liner.astype(np.float32) / 255.0) * liner_intensity
    alpha = np.clip(alpha, 0, 1)
    alpha = np.expand_dims(alpha, axis=-1)  # Match image channels

    # Blend the lip liner with the original image
    output = (image * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)

    return output


# Eyeliner 
 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True)

def resize_image(image, target_width=1000, min_width=500):
    height, width = image.shape[:2]
    if width > target_width:
        scale = target_width / width
    elif width < min_width:
        scale = min_width / width
    else:
        return image
    new_size = (int(width * scale), int(height * scale))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    return resized_image

def get_face_landmarks(image, attempts=1):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = None
    for _ in range(attempts):
        results = face_mesh.process(rgb_image)
        if results.multi_face_landmarks:
            break
    return results.multi_face_landmarks[0] if results.multi_face_landmarks else None

def get_upper_eyelid_landmarks(image, face_landmarks):
    height, width, _ = image.shape
    upper_left_eyelid = [173, 157, 158, 159, 160, 161, 246, 33, 130]
    upper_right_eyelid = [398, 384, 385, 386, 387, 388, 466, 263, 359]
    left_eye_width = abs(face_landmarks.landmark[33].x - face_landmarks.landmark[130].x) * width
    right_eye_width = abs(face_landmarks.landmark[263].x - face_landmarks.landmark[359].x) * width
    lift_factor = int(min(left_eye_width, right_eye_width) * 0.4)
    right_eye_points = [(int(face_landmarks.landmark[idx].x * width), 
                         int(face_landmarks.landmark[idx].y * height) - lift_factor) 
                        for idx in upper_right_eyelid]
    left_eye_points = [(int(face_landmarks.landmark[idx].x * width), 
                        int(face_landmarks.landmark[idx].y * height) - lift_factor)
                       for idx in upper_left_eyelid]
    return left_eye_points, right_eye_points


def draw_smooth_curve(image, points, color, min_thickness, max_thickness):
    points = np.array(points, dtype=np.float32)
    if len(points) < 3:
        return image
    tck, u = splprep(points.T, s=0.001)
    u_fine = np.linspace(0, 1, num=400)
    x_new, y_new = splev(u_fine, tck)
    overlay = image.copy()

    def smooth_thickness(t):
        return min_thickness + (max_thickness - min_thickness) * (1 - np.cos(np.pi * t)) / 2

    num_points = len(x_new)
    for i in range(1, num_points):
        t = i / num_points
        thickness = int(smooth_thickness(t))
        cv2.line(overlay, (int(x_new[i-1]), int(y_new[i-1])), (int(x_new[i]), int(y_new[i])), color, thickness, cv2.LINE_AA)
    return overlay

def draw_wing(image_with_eyeliner, eye_points, direction, shade):
    if not eye_points or len(eye_points) < 2:
        print("Error: eye_points list is empty or too short!")
        return
    eye_end = eye_points[-1]
    if eye_end is None:
        print("Error: eye_end is None!")
        return
    eye_width = abs(eye_points[0][0] - eye_points[-1][0])
    wing_length = max(5, min(int(eye_width * 0.25), 24))
    wing_height = max(3, min(int(eye_width * 0.10), 10))
    base_thickness = max(2, min(int(eye_width * 0.10), 7))
    num_segments = 5
    shade_bgr = (shade[2], shade[1], shade[0])

    for i in range(num_segments):
        ratio = (i + 1) / num_segments
        start_point = (eye_end[0] + int(direction * ratio * wing_length * 0.4),
                       eye_end[1] - int(ratio * wing_height * 0.4))
        end_point = (eye_end[0] + int(direction * ratio * wing_length),
                     eye_end[1] - int(ratio * wing_height))
        thickness = max(1, int(base_thickness * (1 - ratio)))
        cv2.line(image_with_eyeliner, start_point, end_point, shade_bgr, thickness, cv2.LINE_AA)

# -------- MODIFICATIONS FOR DIFFERENT STYLES -------- #

def modify_points_thin(eye_points):
    """
    Removes the outermost point from the eye to eliminate the wing effect.
    Returns a cleaner eyeliner shape with no extension.
    """
    if len(eye_points) < 2:
        return eye_points  # Not enough points to trim

    # Remove the last point (outer edge) to prevent wing
    modified = eye_points[:-1]

    return modified

def modify_points_thin_wing(eye_points):
    """
    Removes the outermost point from the eye to eliminate the wing effect.
    Returns a cleaner eyeliner shape with no extension.
    """
    if len(eye_points) < 2:
        return eye_points  # Not enough points to trim

    # Remove the last point (outer edge) to prevent wing
    modified = eye_points[:-1]

    return modified

def modify_points_normal(eye_points):
    """
    Removes the outermost point from the eye to eliminate the wing effect.
    Returns a cleaner eyeliner shape with no extension.
    """
    if len(eye_points) < 2:
        return eye_points  # Not enough points to trim

    # Remove the last point (outer edge) to prevent wing
    modified = eye_points[:-1]

    return modified
# ------------- MAIN APPLY FUNCTION ------------- #

def apply_eyeliner(image, left_eye_points, right_eye_points, color, transparency=1.65, style='wing'):
    if left_eye_points is None or right_eye_points is None:
        return image

    if style == 'normal':
        left_eye_points = modify_points_normal(left_eye_points)
        right_eye_points = modify_points_normal(right_eye_points)
    elif style == 'thin':
        left_eye_points = modify_points_thin(left_eye_points)
        right_eye_points = modify_points_thin(right_eye_points)
    elif style == 'thin+wing':
        left_eye_points = modify_points_thin_wing(left_eye_points)
        right_eye_points = modify_points_thin_wing(right_eye_points)

    image_with_eyeliner = image.copy()
    color_bgr = (color[2], color[1], color[0])
    eye_width = abs(left_eye_points[-1][0] - left_eye_points[0][0])
    thickness = max(2, min(int(eye_width * 0.14), 8))
    
    if style == 'thin' or style == 'thin+wing':
        # Draw each curve separately
        image_with_eyeliner = draw_smooth_curve(image_with_eyeliner, left_eye_points, color_bgr, 2, 2)
        image_with_eyeliner = draw_smooth_curve(image_with_eyeliner, right_eye_points, color_bgr, 2, 2)
    
    
    else:
        image_with_eyeliner = draw_smooth_curve(image_with_eyeliner, left_eye_points, color_bgr, 2, thickness)
        image_with_eyeliner = draw_smooth_curve(image_with_eyeliner, right_eye_points, color_bgr, 2, thickness)

    if style == 'wing' or style == 'thin+wing':
        draw_wing(image_with_eyeliner, left_eye_points, -1, color)
        draw_wing(image_with_eyeliner, right_eye_points, 1, color)

    blurred = cv2.GaussianBlur(image_with_eyeliner, (3, 3), 0.5)
    return cv2.addWeighted(image, 1 - transparency, blurred, transparency, 0)




#Eyeshadow 

def get_eyelash_region(landmarks, eyelash_indices, w, h):
    """
    Define an eyelash region that should not be covered by eyeshadow.
    """
    try:
        # Extract eyelash points
        eyelash_points = np.array(
            [(int(landmarks.landmark[idx].x * w), int(landmarks.landmark[idx].y * h)) for idx in eyelash_indices],
            dtype=np.int32,
        )
    except AttributeError:
        eyelash_points = np.array(
            [(landmarks.part(idx).x, landmarks.part(idx).y) for idx in eyelash_indices],
            dtype=np.int32,
        )

    return np.array(eyelash_points, dtype=np.int32)

def apply_gradient_mask(mask):
    """
    Creates a gradient effect on the mask, fading out gradually.
    """
    h, w = mask.shape
    gradient = np.linspace(1, 0, int(h * 0.4)).reshape(-1, 1)  # Vertical gradient fade
    full_gradient = np.zeros_like(mask, dtype=np.float32)
    
    y_indices = np.where(mask > 0)[0]  # Get y-coordinates of mask area
    if y_indices.size > 0:
        min_y, max_y = y_indices.min(), y_indices.max()
        fade_region = full_gradient[min_y:max_y, :]
        
        fade_length = min(fade_region.shape[0], gradient.shape[0])
        fade_region[:fade_length] = gradient[:fade_length]  # Apply gradient

        full_gradient[min_y:max_y, :] = fade_region

    return (mask * full_gradient).astype(np.uint8)


def create_eyeshadow_mask(image, landmarks):

    """
    Create a mask for eyeshadow that stays on the eyelid region, avoiding the eyelashes.
    """
    h, w, _ = image.shape  
    mask_left = np.zeros((h, w), dtype=np.uint8)
    mask_right = np.zeros((h, w), dtype=np.uint8)

    # Define landmark indices for the eyelids and eyelashes
    left_upper_eyelid_indices = [33, 246, 161, 160, 159, 158, 157, 173]
    right_upper_eyelid_indices = [398, 384, 385, 386, 387, 388, 466,263,359]
    left_upper_eyelash_indices = [163, 144, 145, 153, 154, 155]
    right_upper_eyelash_indices = [390, 373, 374, 380, 381, 382]

    left_eye_region = get_eyeshadow_region(landmarks, left_upper_eyelid_indices, w, h, extension_factor=0.025)
    right_eye_region = get_eyeshadow_region(landmarks, right_upper_eyelid_indices, w, h, extension_factor=0.025)


    left_eyelash_region = get_eyelash_region(landmarks, left_upper_eyelash_indices, w, h)
    right_eyelash_region = get_eyelash_region(landmarks, right_upper_eyelash_indices, w, h)

    # Fill the regions into masks
    cv2.fillPoly(mask_left, [left_eye_region], 255)
    cv2.fillPoly(mask_left, [left_eyelash_region], 0)  # Exclude the eyelash region

    cv2.fillPoly(mask_right, [right_eye_region], 255)
    cv2.fillPoly(mask_right, [right_eyelash_region], 0)  # Exclude the eyelash region
    # Apply gradient effect for smooth blending
    mask_left = apply_gradient_mask(mask_left)
    mask_right = apply_gradient_mask(mask_right)
    return mask_left, mask_right

def apply_eyeshadow(image, mask_left, mask_right, color, intensity=0.95):
    """
    Apply a natural-looking eyeshadow effect with enhanced blending at the outer edges.
    """
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[mask_left > 0] = color
    overlay[mask_right > 0] = color

    # Increased Gaussian blur for a more natural diffusion
    blurred_overlay = cv2.GaussianBlur(overlay, (35, 35), 20)

    # Reduce intensity slightly to prevent an artificial look
    blended_image = cv2.addWeighted(image, 1, blurred_overlay, intensity, 0)

    # Combine masks for better edge control
    combined_mask = cv2.bitwise_or(mask_left, mask_right)

    # Improve blending by applying **directional blurring**
    feathered_mask = cv2.GaussianBlur(combined_mask, (55, 55), 30).astype(np.float32) / 255.0

    # Apply directional feathering for outer-edge soft blending
    for c in range(3):
        blended_image[..., c] = (
            feathered_mask * blended_image[..., c] + (1 - feathered_mask) * image[..., c]
        )

    return blended_image

def get_eyeshadow_region(landmarks, eyelid_indices, w, h ,extension_factor, corner_extension=0.03):
    """
    Define an eyeshadow region with better coverage at the outer edges.
    Increase upward extension, especially for HEIC images.
    """
    try:
        eyelid_points = np.array(
            [(int(landmarks.landmark[idx].x * w), int(landmarks.landmark[idx].y * h)) for idx in eyelid_indices],
            dtype=np.int32,
        )
    except AttributeError:
        eyelid_points = np.array(
            [(landmarks.part(idx).x, landmarks.part(idx).y) for idx in eyelid_indices],
            dtype=np.int32,
        )

    # Extend upwards for better coverage
    extended_points = [(x, int(y - extension_factor * h)) for x, y in eyelid_points]

    # Extend outer corner blending
    leftmost_x = max(eyelid_points[0][0] - int(corner_extension * w), 0)
    rightmost_x = min(eyelid_points[-1][0] + int(corner_extension * w), w - 1)

    leftmost = (leftmost_x, eyelid_points[0][1])
    rightmost = (rightmost_x, eyelid_points[-1][1])

    # Close contour
    region = np.vstack([[leftmost], eyelid_points, [rightmost], extended_points[::-1]])

    return np.array(region, dtype=np.int32)


#Eyebrows
def apply_eyebrow_shade(image, parsing, left_brow, right_brow, color=(50, 30, 20), shade_intensity=0.4):
    """
    Enhances eyebrow hairs naturally while preserving hair texture.
    Applies only to eyebrows (no extra shading outside the area).
    Uses adaptive edge detection for precise hair detection.
    Soft-light blending for a natural, smooth look.
    """

    # Ensure parsing mask size matches the image
    if parsing.shape[:2] != image.shape[:2]:
        parsing = cv2.resize(parsing, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Extract eyebrow mask (left + right brows)
    brows_mask = np.where((parsing == left_brow) | (parsing == right_brow), 255, 0).astype(np.uint8)

    if np.sum(brows_mask) == 0:
        print("Eyebrow mask is empty. Check segmentation.")
        return image  # Return original image if no eyebrows detected

    # Convert image to grayscale for hair detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyebrow_region = cv2.bitwise_and(gray, gray, mask=brows_mask)

    # **Step 1: Adaptive Hair Strand Detection (Fine Edge Detection)**
    sigma = 0.33  # Auto-thresholding for edge detection
    median_val = np.median(eyebrow_region)
    lower = int(max(0, (1.0 - sigma) * median_val))
    upper = int(min(255, (1.0 + sigma) * median_val))
    edges = cv2.Canny(eyebrow_region, lower, upper)  # Adaptive edge detection

    # **Step 2: Remove Extra Noise (Keeps Only Eyebrow Hair)**
    edges = cv2.bitwise_and(edges, brows_mask)  # Ensure it stays within eyebrow region

    # **Step 3: Strengthen Eyebrow Hairs (Morphological Processing)**
    kernel = np.ones((2, 2), np.uint8)
    eyebrow_hair_mask = cv2.dilate(edges, kernel, iterations=1)  # Strengthen fine hairs
    eyebrow_hair_mask = cv2.bitwise_and(eyebrow_hair_mask, brows_mask)  # Remove unwanted noise
    eyebrow_hair_mask = cv2.GaussianBlur(eyebrow_hair_mask, (3, 3), 1)  # Smooth out edges

    # **Step 4: Create a Natural Color Overlay**
    brow_shade = np.full_like(image, color, dtype=np.uint8)
    brow_shade = cv2.bitwise_and(brow_shade, brow_shade, mask=eyebrow_hair_mask)

    # **Step 5: Generate a Soft Alpha Mask for Natural Blending**
    alpha = (eyebrow_hair_mask.astype(np.float32) / 255.0) * shade_intensity
    alpha = cv2.GaussianBlur(alpha, (7, 7), 3)  # Feather for soft blending
    alpha = np.clip(alpha, 0, 1)
    alpha = cv2.merge([alpha, alpha, alpha])  # Convert to 3-channel alpha mask

    # **Step 6: Apply Soft Light Blending for a Natural Effect**
    output = image.astype(np.float32) * (1 - alpha) + brow_shade.astype(np.float32) * alpha

    return output.astype(np.uint8)


#Concealer 
def create_concealer_mask(image, face_landmarks, shift_down=20):
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Under-eye landmark points from MediaPipe model
    LEFT_UNDER_EYE = [341, 256, 252, 253, 254, 339, 255, 448, 348, 349, 350, 453, 446, 261]  
    RIGHT_UNDER_EYE = [226, 25, 110, 24, 23, 22, 26, 112, 128, 121, 120, 119, 229, 228, 31]

    # Convert landmarks to pixel coordinates and shift downward
    left_eye_points = np.array([(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h) + shift_down) for i in LEFT_UNDER_EYE])
    right_eye_points = np.array([(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h) + shift_down) for i in RIGHT_UNDER_EYE])

    # Fill the shifted under-eye regions
    cv2.fillPoly(mask, [cv2.convexHull(left_eye_points)], 255)
    cv2.fillPoly(mask, [cv2.convexHull(right_eye_points)], 255)

    # Apply Gaussian blur for natural blending
    mask = cv2.GaussianBlur(mask, (15, 15), 10)

    return mask


def apply_concealer(image, mask, shade, intensity=0.6):
    """
    Applies a realistic concealer effect using the specified shade with preserved skin texture.

    :param image: Input BGR image (NumPy array)
    :param mask: Binary mask where concealer should be applied
    :param shade: BGR tuple representing the concealer color
    :param intensity: Blend intensity (0.0 - subtle, 1.0 - full coverage)
    :return: Image with concealer applied
    """
    # Convert image to float32 for smoother blending
    image = image.astype(np.float32)

    # ✅ Use the provided shade color directly
    concealer_layer = np.full_like(image, shade, dtype=np.float32)

    # Apply stronger Gaussian blur for softer edges
    blurred_mask = cv2.GaussianBlur(mask, (45, 45), sigmaX=30, sigmaY=30)

    # Normalize mask to range [0,1] and apply adaptive opacity
    alpha = (blurred_mask.astype(float) / 255.0) * intensity

    # Blend image and concealer
    blended = (image * (1 - alpha[..., None]) + concealer_layer * alpha[..., None])

    # ✅ Preserve skin texture using bilateral filtering
    texture_preserved = cv2.bilateralFilter(blended.astype(np.uint8), d=5, sigmaColor=35, sigmaSpace=350)

    # ✅ Add slight sharpness to avoid over-smoothing
    sharpened = cv2.addWeighted(texture_preserved, 1.2, image.astype(np.uint8), -0.2, 0)

    # ✅ Blend with original image slightly for a natural look
    final = cv2.addWeighted(sharpened, 0.85, image.astype(np.uint8), 0.15, 0)

    return final

#contour

def get_highlight_contour_masks(image, landmarks):
    h, w, _ = image.shape

    contour_points_left = []
    contour_points_right = []
    chin_points_right = []

    # Draw landmarks and collect points for the contour
    for index, landmark in enumerate(landmarks.landmark):
        x = int(landmark.x * w)
        y = int(landmark.y * h)

        # Add specific points to the contour list
        if index in [21, 67, 69, 71]:  # Adjust indices based on your needs
            contour_points_left.append([x, y])
        
    
    contour_points_left = np.array([contour_points_left], dtype=np.int32)

    for index, landmark in enumerate(landmarks.landmark):
        x = int(landmark.x * w)
        y = int(landmark.y * h)

        # Add specific points to the contour list
        if index in [338, 284, 337, 298]:  # Adjust indices based on your needs
            contour_points_right.append([x, y])
        
    contour_points_right = np.array([contour_points_right], dtype=np.int32)


    # Chin contour points
    chin_contour_points_left = np.array([
        [landmarks.landmark[132].x * w, landmarks.landmark[132].y * h],  # Start of jawline left
        [landmarks.landmark[58].x * w, landmarks.landmark[58].y * h],    # Mid-jawline left
        [landmarks.landmark[138].x * w, landmarks.landmark[138].y * h],  # Below the ear left
        [landmarks.landmark[136].x * w, landmarks.landmark[136].y * h],  # Near chin left
        [landmarks.landmark[210].x * w, landmarks.landmark[210].y * h],  # Chin bottom
    ], dtype=np.int32).reshape((-1, 1, 2))
    


    for index, landmark in enumerate(landmarks.landmark):
        x = int(landmark.x * w)
        y = int(landmark.y * h)

        # Add specific points to the contour list
        if index in [411,433,397,364,430]:  # Adjust indices based on your needs
            chin_points_right.append([x, y])
        
    chin_contour_points_right = np.array([chin_points_right], dtype=np.int32)


    mask_forehead_left = np.zeros((h, w), dtype=np.uint8)
    mask_forehead_right = np.zeros((h, w), dtype=np.uint8)
    mask_chin_left = np.zeros((h, w), dtype=np.uint8)
    mask_chin_right = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(mask_forehead_left, [contour_points_left], 255)
    cv2.fillPoly(mask_forehead_right, [contour_points_right], 255)
    cv2.fillPoly(mask_chin_left, [chin_contour_points_left], 255)
    cv2.fillPoly(mask_chin_right, [chin_contour_points_right], 255)

    return mask_forehead_left, mask_forehead_right, mask_chin_left, mask_chin_right

# Use this function in your processing flow to apply contouring to the specified areas
     
def brighten_color(color, factor=1.5):
    # Convert RGB color to brighter version by scaling its components up to the max value 255
    return tuple(min(int(c * factor), 255) for c in color)

def apply_contour(image, mask_left, mask_right, mask_chin_left, mask_chin_right, color=(105, 105, 105), intensity=0.4, brightness_factor=1.5):
    # Brighten the color applied to the mask
    brightened_color = brighten_color(color, brightness_factor)

    # Create an overlay for the contour
    overlay = np.zeros_like(image, dtype=np.uint8)
    
    # Apply brightened color to the masks
    overlay[mask_left == 255] = brightened_color
    overlay[mask_right == 255] = brightened_color
    overlay[mask_chin_left == 255] = brightened_color
    overlay[mask_chin_right == 255] = brightened_color
    
    # Apply Gaussian Blur to create a gradient effect
    blurred_overlay = cv2.GaussianBlur(overlay, (171, 171), 100)
    
    # Blend the blurred overlay with the original image
    blended_image = cv2.addWeighted(image, 1, blurred_overlay, intensity + 0.3, 0)
    
    return blended_image


def interpolate_color(value, start_color, end_color):
    """
    Interpolates between two colors based on the value (0 to 100).
    """
    return tuple(
        int(start + (end - start) * (value / 100.0))
        for start, end in zip(start_color, end_color)
    )

def apply_skin_tone(image, parsing, part, color, alpha=0.3):
    """
    Apply the skin tone adjustment to the specified regions with a natural effect.
    blended
    Args:
    image (np.array): Original image.
    parsing (np.array): Segmented face parsing map.
    part (int): Face part index for skin tone adjustment.
    color (tuple): RGB color for skin tone.
    alpha (float): Transparency factor for blending, where 0 is fully transparent and 1 is fully opaque.
    """
    # Create a mask for the skin areas
    mask = (parsing == part).astype(np.uint8)
    
    # Apply the color to the skin area with alpha blending
    result = image.copy()
    for c in range(3):  # Loop over color channels
        result[:, :, c] = np.where(mask, image[:, :, c] * (1 - alpha) + color[c] * alpha, image[:, :, c])

    return result


def create_precise_iris_mask(image, landmarks, iris_indices):
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Get iris points
    iris_points = np.array([(int(landmarks.landmark[idx].x * w), int(landmarks.landmark[idx].y * h)) for idx in iris_indices])

    # Compute the convex hull of the iris points
    hull = cv2.convexHull(iris_points)

    # Draw the convex hull on the mask
    cv2.fillConvexPoly(mask, hull, 255)
    
    return mask


def change_iris_color(image, mask, color):
    b, g, r = color
    iris_color = np.zeros_like(image)
    iris_color[:, :, 0] = b
    iris_color[:, :, 1] = g
    iris_color[:, :, 2] = r

    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    mask = mask.astype(bool)
    blended_image = image.copy()
    blended_image[mask] = cv2.addWeighted(image[mask], 0.5, iris_color[mask], 0.5, 0)
    return blended_image


def apply_iris_color_change(image, face_mesh, color):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_iris_indices = [474, 475, 476, 477]
            right_iris_indices = [469, 470, 471, 472]

            left_iris_mask = create_precise_iris_mask(image, face_landmarks, left_iris_indices)
            right_iris_mask = create_precise_iris_mask(image, face_landmarks, right_iris_indices)

            iris_mask = cv2.bitwise_or(left_iris_mask, right_iris_mask)
            return change_iris_color(image, iris_mask, color)
    return image


def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, mode='reflect')

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)



def get_face_mask(image, landmarks):
    """
    Generate a face contour mask using key landmarks to prevent blush from spilling onto hair.
    """
    h, w, _ = image.shape

    # Define points around the jawline and forehead (face oval area)
    # Use landmark indices that cover the outer face contour
    contour_indices = [
        10,  # Top center forehead
        338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
        378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
        162, 21, 54, 103, 67, 109
    ]

    points = []
    for idx in contour_indices:
        lm = landmarks.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        points.append([x, y])

    # Convert points to numpy array for fillPoly
    points = np.array(points, dtype=np.int32)

    # Create mask and fill polygon
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)

    # Optional: Smooth the edges
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    return mask

def get_cheek_masks(image, landmarks):
    h, w, _ = image.shape

    try:
        left_landmark = landmarks.landmark[205]
        right_landmark = landmarks.landmark[425]

        left_center = (int(left_landmark.x * w), int(left_landmark.y * h))
        right_center = (int(right_landmark.x * w), int(right_landmark.y * h))

        # Refine placement: Shift up slightly to match apple of cheeks
        vertical_shift = int(h * 0.03)
        left_center = (left_center[0], left_center[1] - vertical_shift)
        right_center = (right_center[0], right_center[1] - vertical_shift)

        face_width = abs(landmarks.landmark[234].x - landmarks.landmark[454].x) * w
        face_height = abs(landmarks.landmark[10].y - landmarks.landmark[152].y) * h
        radius_x = int(min(face_width, face_height) * 0.12)
        radius_y = int(radius_x * 0.55)  # Make it horizontally oval
    except IndexError:
        print("Error: Landmark points missing!")
        return None, None

    # Create blank masks
    mask_left = np.zeros((h, w), dtype=np.uint8)
    mask_right = np.zeros((h, w), dtype=np.uint8)

    # Draw elliptical blush areas
    cv2.ellipse(mask_left, left_center, (radius_x, radius_y), 0, 0, 360, 255, -1)
    cv2.ellipse(mask_right, right_center, (radius_x, radius_y), 0, 0, 360, 255, -1)

    # Apply Gaussian blur
    blur_strength = 75 if w > 400 else 50
    mask_left = cv2.GaussianBlur(mask_left, (151, 151), blur_strength)
    mask_right = cv2.GaussianBlur(mask_right, (151, 151), blur_strength)

    face_mask = get_face_mask(image, landmarks)
    mask_left = cv2.bitwise_and(mask_left, face_mask)
    mask_right = cv2.bitwise_and(mask_right, face_mask)

    return mask_left, mask_right

def apply_blush(image, mask_left, mask_right, color=(255, 100, 150), intensity=0.5):
    # Convert image to float32 for smooth blending
    image = image.astype(np.float32) / 255.0

    # Combine and normalize masks
    combined_mask = cv2.add(mask_left, mask_right).astype(np.float32) / 255.0
    combined_mask = cv2.GaussianBlur(combined_mask, (101, 101), 50)

    # Normalize and clamp alpha
    alpha = np.clip(combined_mask * intensity, 0, 0.8)  # Lower cap for realism

    # Prepare blush color in BGR with softened saturation
    color_bgr = np.array([color[2], color[1], color[0]]) / 255.0
    color_bgr = np.clip(color_bgr *  1.0, 0, 1)  # Softer tone

    # Create color layer
    blush_color_layer = np.ones_like(image) * color_bgr

    # Blend with original image using alpha
    blush_result = (1 - alpha[..., None]) * image + alpha[..., None] * blush_color_layer

    # Apply slight gamma correction to preserve depth
    blush_result = np.clip(blush_result, 0, 1)
    blush_result = np.power(blush_result, 1 / 1.05)  # subtle gamma correction

    return (blush_result * 255).astype(np.uint8)                    



def exclude_face_from_mask(image, mask):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return mask  # No face detected

    face_points = []
    for lm in results.multi_face_landmarks[0].landmark:
        x = int(lm.x * image.shape[1])
        y = int(lm.y * image.shape[0])
        face_points.append([x, y])

    face_mask = np.zeros_like(mask)
    cv2.fillConvexPoly(face_mask, np.array(face_points, dtype=np.int32), 255)

    # Invert face mask: 255 for non-face
    inv_face_mask = cv2.bitwise_not(face_mask)

    # Remove face from hair mask
    final_mask = cv2.bitwise_and(mask, mask, mask=inv_face_mask)
    return final_mask

def hair(image, parsing, part=17, color=[50, 20, 220]):
    """
    Applies a natural-looking hair color effect preserving hair texture.

    Parameters:
    - image: input BGR image (uint8)
    - parsing: hair segmentation mask
    - part: label index for hair
    - color: desired BGR hair color
    """
    # Binary mask for hair region
    mask = (parsing == part).astype(np.uint8) * 255

    # Clean and blur mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    blurred_mask = cv2.GaussianBlur(mask, (25, 25), 10)
    alpha = blurred_mask.astype(np.float32) / 255.0

    # Convert to float for blending
    image_float = image.astype(np.float32) / 255.0

    # Create solid color layer
    target_color = np.array(color, dtype=np.float32) / 255.0
    color_layer = np.ones_like(image_float) * target_color

    # Blend using soft light-like technique
    blended = image_float * (1 - 0.5 * alpha[..., None]) + color_layer * (0.5 * alpha[..., None])

    # Optional: Slight enhancement to contrast
    result = cv2.addWeighted(image_float, 0.5, blended, 0.5, 0)

    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


# Iris Color Functions
import os

# MediaPipe iris landmarks
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYELID = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYELID = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

def load_iris_overlay(color_name, pupil_radius_fraction=0.22, overall_opacity=0.75):
    """
    Load iris overlay image from iriscolors folder and prepare it for blending
    """
    try:
        # Load the iris overlay image
        iris_path = f"iriscolors/{color_name.lower()}.png"
        if not os.path.exists(iris_path):
            raise FileNotFoundError(f"Iris color file not found: {iris_path}")
            
        overlay = Image.open(iris_path).convert("RGBA")
        overlay_np = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGBA2BGRA)
        
        # Create alpha channel with pupil hole
        alpha = cv2.GaussianBlur(overlay_np[:, :, 3], (9, 9), 0)
        h, w = alpha.shape
        pupil_r = int(min(h, w) * pupil_radius_fraction)
        cv2.circle(alpha, (w//2, h//2), pupil_r, 0, -1)
        
        # Apply overall opacity
        overlay_np[:, :, 3] = (alpha * overall_opacity).astype(np.uint8)
        
        return overlay_np
    except Exception as e:
        print(f"Error loading iris overlay: {e}")
        return None

def apply_overlay_on_overlap(image, overlay_rgba, iris_pts, eyelid_pts):
    """
    Apply iris overlay only where it overlaps with the eyelid region
    """
    if overlay_rgba is None:
        return image
        
    # Calculate iris center and radius
    cx = int(np.mean([p[0] for p in iris_pts]))
    cy = int(np.mean([p[1] for p in iris_pts]))
    r = int(np.mean([np.linalg.norm(np.array(p) - [cx, cy]) for p in iris_pts]))
    
    # Create iris mask
    iris_mask = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(iris_mask, (cx, cy), r, 255, -1)
    
    # Create eyelid mask
    eyelid_mask = np.zeros_like(iris_mask)
    cv2.fillPoly(eyelid_mask, [np.array(eyelid_pts, np.int32)], 255)
    
    # Get overlap area
    overlap_mask = cv2.bitwise_and(iris_mask, eyelid_mask)
    
    # Resize overlay to match iris size
    ol = cv2.resize(overlay_rgba, (2*r, 2*r), interpolation=cv2.INTER_AREA)
    oh, ow = ol.shape[:2]
    tx, ty = cx - ow//2, cy - oh//2
    
    # Apply overlay pixel by pixel in overlap area
    for y in range(oh):
        for x in range(ow):
            yy, xx = ty + y, tx + x
            if (0 <= yy < image.shape[0] and 0 <= xx < image.shape[1] and 
                overlap_mask[yy, xx] > 0):
                alpha = ol[y, x, 3] / 255.0
                if alpha > 0:
                    # Blend the overlay with the original image
                    for c in range(3):
                        image[yy, xx, c] = (
                            (1 - alpha) * image[yy, xx, c] + 
                            alpha * ol[y, x, c]
                        ).astype(np.uint8)
    
    return image

def change_iris_color_advanced(image, iris_color_name, opacity=0.75):
    """
    Apply iris color change using MediaPipe face mesh detection
    """
    # MediaPipe landmark indices
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    LEFT_EYELID = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
    RIGHT_EYELID = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
    
    # Load iris overlay
    overlay_rgba = load_iris_overlay(iris_color_name, overall_opacity=opacity)
    if overlay_rgba is None:
        print(f"Failed to load iris overlay for {iris_color_name}")
        return image
    
    h, w = image.shape[:2]
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = face_mesh.process(rgb_image)
    
    if not results.multi_face_landmarks:
        print("No face detected for iris color change")
        return image
    
    # Apply iris color to each detected face
    for face_landmarks in results.multi_face_landmarks:
        # Get eyelid coordinates
        left_eyelid_pts = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYELID]
        right_eyelid_pts = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYELID]
        
        # Get iris coordinates  
        left_iris_pts = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_IRIS]
        right_iris_pts = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_IRIS]
        
        # Apply overlay to both eyes
        image = apply_overlay_on_overlap(image, overlay_rgba.copy(), left_iris_pts, left_eyelid_pts)
        image = apply_overlay_on_overlap(image, overlay_rgba.copy(), right_iris_pts, right_eyelid_pts)
    
    return image

def get_available_iris_colors():
    """Get list of available iris colors from iriscolors folder"""
    iris_folder = "iriscolors"
    if not os.path.exists(iris_folder):
        return []
    
    colors = []
    for file in os.listdir(iris_folder):
        if file.lower().endswith('.png'):
            color_name = file.replace('.png', '')
            colors.append(color_name)
    
    return sorted(colors)

    # change_skin_tone_checkbox = st.sidebar.checkbox('Change Skin Tone (beta function)')

    # if change_skin_tone_checkbox:
    #     # Select the color and alpha value for the skin tone adjustment
    #     skin_color_hex = st.sidebar.color_picker('Pick the Skin Tone Color', '#8D5524')
    #     alpha = st.sidebar.slider('Adjust Skin Tone Transparency', 0.0, 0.50, 0.5)

    #     # Convert hex to RGB
    #     skin_color_rgb = tuple(int(skin_color_hex[i:i+2], 16) for i in (1, 3, 5))

    #     # Apply the skin tone adjustment
    #     image = apply_skin_tone(image, parsing, table['skin'], skin_color_rgb, alpha)
    #     image = apply_skin_tone(image, parsing,10, skin_color_rgb, alpha)
    #     image = apply_skin_tone(image, parsing,14, skin_color_rgb, alpha)
        



    # Apply contour if the checkbox is checked
    # if apply_contour_checkbox:
    #     contour_color = st.sidebar.color_picker('Pick the Contour Color', '#6B4226')  # Darker shade for contour
    #     contour_color = ImageColor.getcolor(contour_color, "RGB")
    #     if results.multi_face_landmarks:
    #         for face_landmarks in results.multi_face_landmarks:
    #             mask_forehead_left, mask_forehead_right, mask_chin_left, mask_chin_right = get_highlight_contour_masks(image, face_landmarks)
    #             image = apply_contour(image, mask_forehead_left, mask_forehead_right, mask_chin_left, mask_chin_right, color=contour_color)


    image = cv2.resize(image, (w, h))
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Original Image')
        st.image(new_image, use_column_width=True)

    with col2:
        st.subheader('Output Image')
        st.image(image, use_column_width=True)
