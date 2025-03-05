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


def visualize_landmarks(image, landmarks, font_scale=0.5, color=(0, 255, 0), thickness=1):
    """
    Visualize landmarks on the given image by annotating them with their index numbers.
    
    Parameters:
    - image: The input image as a numpy array.
    - landmarks: The landmarks detected by the face mesh model.
    - font_scale: Font scale for the numbers.
    - color: Color of the text for the numbers.
    - thickness: Thickness of the text.
    
    Returns:
    - The image with landmarks visualized as numbers.
    """
   
    h, w, _ = image.shape
    for idx, landmark in enumerate(landmarks.landmark):
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        # Put the index number at each landmark position
        cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    return image


def get_concealer_masks(image, landmarks):
    h, w, _ = image.shape

    # Define concealer mask points for the left and right eyes
    contour_points_left = []
    contour_points_right = []

    # Draw landmarks and collect points for the contour
    for index, landmark in enumerate(landmarks.landmark):
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        # Add specific points to the contour list
        if index in [31, 111, 117, 118, 119, 120, 121, 232,128]:  # Adjust indices based on your needs
            contour_points_left.append([x, y])

    # Ensure the points are in the correct format for cv2.fillPoly
    contour_points_left = np.array([contour_points_left], dtype=np.int32)
    
    for index, landmark in enumerate(landmarks.landmark):
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        # Add specific points to the contour list
        if index in [453,350,349,348,347,346,261,340]:  # Adjust indices based on your needs
            contour_points_right.append([x, y])

    # Ensure the points are in the correct format for cv2.fillPoly
    contour_points_right = np.array([contour_points_right], dtype=np.int32)

    mask_left = np.zeros((h, w), dtype=np.uint8)
    mask_right = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(mask_left, contour_points_left, 255)
    cv2.fillPoly(mask_right, contour_points_right, 255)

    return mask_left, mask_right

def soft_light_blend(base, overlay, intensity=0.1, scale=1.0):
    """Apply a soft light blend to simulate makeup blending with adjustable scale."""
    # Scale the intensity based on the scaling factor
    scaled_intensity = intensity * scale
    
    # Convert color to float and scale from 0 to 1
    base = base.astype(np.float32) / 255
    overlay = overlay.astype(np.float32) / 255

    # Soft light blending equation modified with scaled intensity
    blend = base * (1 - scaled_intensity) + (2 * base * overlay + base * base * (1 - 2 * overlay)) * scaled_intensity
    blend = np.clip(blend, 0, 1)  # Ensure the values are within the valid range

    # Convert back to 8-bit colors
    result = (blend * 255).astype(np.uint8)
    return result

def apply_concealer(image, mask_left, mask_right, color=(255, 224, 189), intensity=0.1):
    """Apply concealer directly using given masks, with specified color and intensity."""
    # Create an overlay with the concealer color
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[mask_left == 255] = color
    overlay[mask_right == 255] = color

    # Apply Gaussian blur for a smoother transition
    blur_size = (31,31)  # Large blur size for very soft edges
    blurred_overlay = cv2.GaussianBlur(overlay, blur_size,50)

    # Blend the blurred overlay with the original image using the given intensity
    blended_image = cv2.addWeighted(image, 1, blurred_overlay, intensity + 0.2, 0)

    return blended_image

def override_cheek_mask():
    ''' 
additional method to get blush mask
def get_cheek_masks(image, landmarks):
    h, w, _ = image.shape
    
    left_cheek_points = [234, 93, 132, 58, 4, 164, 166]
    right_cheek_points = [234, 93, 132, 58, 4, 164]

    left_cheek = []
    right_cheek = []

    for idx in left_cheek_points:
        if idx < len(landmarks.landmark):
            left_cheek.append([landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h])
        else:
            # If any coordinate is missing, return empty masks
            return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)

    for idx in right_cheek_points:
        if idx < len(landmarks.landmark):
            right_cheek.append([w - (landmarks.landmark[idx].x * w), landmarks.landmark[idx].y * h])
        else:
            # If any coordinate is missing, return empty masks
            return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)

    left_cheek = np.array(left_cheek, dtype=np.int32)
    right_cheek = np.array(right_cheek, dtype=np.int32)

    mask_left = np.zeros((h, w), dtype=np.uint8)
    mask_right = np.zeros((h, w), dtype=np.uint8)

    if len(left_cheek) > 0:
        cv2.fillPoly(mask_left, [left_cheek], 255)
    
    if len(right_cheek) > 0:
        cv2.fillPoly(mask_right, [right_cheek], 255)
    
    return mask_left, mask_right
'''
    pass
def get_cheek_masks(image, landmarks):
    h, w, _ = image.shape

    try:
        left_cheek = np.array([
            [landmarks.landmark[234].x * w, landmarks.landmark[234].y * h],  # Near the ear
            [landmarks.landmark[93].x * w, landmarks.landmark[93].y * h],    # Cheekbone
            [landmarks.landmark[132].x * w, landmarks.landmark[132].y * h],  # Jawline
            [landmarks.landmark[58].x * w, landmarks.landmark[58].y * h],    # Lower cheek
            [landmarks.landmark[4].x * w, landmarks.landmark[4].y * h],      # Nose
            [landmarks.landmark[164].x * w, landmarks.landmark[164].y * h],  # Upper cheek
            [landmarks.landmark[0].x * w, landmarks.landmark[0].y * h]       # Mid-cheek
        ], dtype=np.int32)
    except IndexError:
        left_cheek = None

    try:
        right_cheek = np.array([
            [landmarks.landmark[454].x * w, landmarks.landmark[454].y * h],
            [landmarks.landmark[323].x * w, landmarks.landmark[323].y * h],
            [landmarks.landmark[361].x * w, landmarks.landmark[361].y * h],
            [landmarks.landmark[288].x * w, landmarks.landmark[288].y * h],
            [landmarks.landmark[14].x * w, landmarks.landmark[14].y * h],
            [landmarks.landmark[397].x * w, landmarks.landmark[397].y * h],
            [landmarks.landmark[399].x * w, landmarks.landmark[399].y * h]
        ], dtype=np.int32)
    except IndexError:
        right_cheek = None

    mask_left = np.zeros((h, w), dtype=np.uint8)
    mask_right = np.zeros((h, w), dtype=np.uint8)

    if left_cheek is not None:
        cv2.fillPoly(mask_left, [left_cheek], 255)
    if right_cheek is not None:
        cv2.fillPoly(mask_right, [right_cheek], 255)

    return mask_left, mask_right

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


def apply_blush(image, mask_left, mask_right, color=(255, 20, 147), intensity=0.4):
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[mask_left == 255] = color
    overlay[mask_right == 255] = color
    
    blurred_overlay = cv2.GaussianBlur(overlay, (171, 171), 100)
    blended_image = cv2.addWeighted(image, 1, blurred_overlay, intensity, 0)
    
    pil_image = Image.fromarray(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_image)
    brightened_image = enhancer.enhance(1)
    
    return cv2.cvtColor(np.array(brightened_image), cv2.COLOR_RGB2BGR)

def color_transfer(source, mask, target_mean, target_std):
    # Ensure source is float and extract masked areas
    source = source.astype(np.float32)
    masked_source = source[mask > 0]

    # Calculate mean and std of the source within the mask
    original_mean = np.mean(masked_source, axis=0)
    original_std = np.std(masked_source, axis=0)

    # Normalize source data based on its statistics
    normalized_source = (masked_source - original_mean) / (original_std + 1e-6)  # Avoid division by zero

    # Apply target mean and std
    transformed_source = normalized_source * target_std + target_mean

    # Place transformed data back into the source image in Lab color space
    new_source = source.copy()
    new_source[mask > 0] = transformed_source

    return new_source


def create_overlay(image, color, alpha=1.0):
    """
    Create an overlay image with the specified color and alpha transparency.
    """
    overlay = np.full_like(image, color, dtype=np.uint8)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def interpolate_color(value, start_color, end_color):
    """
    Interpolate between two colors based on the value (0 to 100).
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


def hair(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r = color
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    # Convert target and image to HSV
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create mask for the specified part
    mask = (parsing == part)

    # Ensure the mask is correctly shaped for the image
    if mask.shape[:2] != image_hsv.shape[:2]:
        raise ValueError("Mask shape does not match image shape.")

    # Apply color transformation where the mask applies
    # Modify hue and saturation gently
    image_hsv[..., 0][mask] = (image_hsv[..., 0][mask] * 0.5 + tar_hsv[..., 0][mask] * 0.5).astype(np.uint8)
    image_hsv[..., 1][mask] = (image_hsv[..., 1][mask] * 0.7 + tar_hsv[..., 1][mask] * 0.3).astype(np.uint8)

    # Soften the value channel by adding noise and lightening
    noise = np.random.normal(loc=0, scale=5, size=image_hsv[..., 2][mask].shape)
    image_hsv[..., 2][mask] = np.clip(image_hsv[..., 2][mask] + noise, 0, 255)

    # Convert back to BGR
    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    return changed

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
    gloss_threshold = np.percentile(gray_lips[lips_mask > 0], 93)  # Top 5% brightest areas
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

# Apply eyebrow color
def apply_eyebrow_color(image, parsing, color, alpha=0.05):
    # Extract B, G, R values from the desired color
    b, g, r = color
    # Create a mask for the eyebrow region
    mask = (parsing == 2) | (parsing == 3)  # Assuming 6 and 7 are the eyebrow parts
    mask = mask.astype(np.uint8) * 255

    # Extract the eyebrow region from the image
    eyebrow_region = cv2.bitwise_and(image, image, mask=mask)

    # Convert the eyebrow region to float for manipulation
    eyebrow_region = eyebrow_region.astype(np.float32)

    # Calculate the target color transformation
    target_color = np.zeros_like(eyebrow_region)
    target_color[:, :, 0] = b
    target_color[:, :, 1] = g
    target_color[:, :, 2] = r

    # Normalize the pixel values
    target_color = target_color.astype(np.float32)
    eyebrow_region[mask > 0] = (eyebrow_region[mask > 0] / 255) * (target_color[mask > 0] / 255)

    # Convert the eyebrow region back to uint8
    eyebrow_region = np.clip(eyebrow_region * 255, 0, 255).astype(np.uint8)

    # Blend the changed eyebrow region back with the original image
    blended_image = cv2.addWeighted(image, 1 - alpha, eyebrow_region, alpha + 0.1, 0)

    return blended_image


#     return final_image
def resize_image(image, target_width=1000, min_width=500):
    """ 
    Resizes image while maintaining aspect ratio, ensuring the face remains recognizable. 
    Uses an adaptive sharpening filter to enhance details after resizing.
    """
    height, width = image.shape[:2]

    # Avoid extreme downscaling to preserve facial details
    if width > target_width:
        scale = target_width / width
    elif width < min_width:
        scale = min_width / width  # Scale up if image is too small (ensures clarity)
    else:
        return image  # Return original if within an optimal size

    new_size = (int(width * scale), int(height * scale))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

    return resized_image

def get_face_landmarks(image, attempts=5):
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
    u_fine = np.linspace(0, 1, num=400)  # Increase resolution
    x_new, y_new = splev(u_fine, tck)

    overlay = image.copy()

    def smooth_thickness(t):
        return min_thickness + (max_thickness - min_thickness) * (1 - np.cos(np.pi * t)) / 2

    num_points = len(x_new)
    for i in range(1, num_points):
        t = i / num_points
        thickness = int(smooth_thickness(t))
        cv2.line(overlay, (int(x_new[i-1]), int(y_new[i-1])), (int(x_new[i]), int(y_new[i])),
                 color, thickness, cv2.LINE_AA)

    return overlay


def draw_wing(image_with_eyeliner, eye_points, direction, shade):
    """Draws a tapered eyeliner wing with the given shade color."""
    if not eye_points or len(eye_points) < 2:
        print("Error: eye_points list is empty or too short!")
        return

    eye_end = eye_points[-1]
    if eye_end is None:
        print("Error: eye_end is None!")
        return

    eye_width = abs(eye_points[0][0] - eye_points[-1][0])

    # Increased wing length for a sharper effect
    wing_length = max(5, min(int(eye_width * 0.25), 24))  # Max length = 24px
    wing_height = max(3, min(int(eye_width * 0.10), 10))   # Max height = 10px
    base_thickness = max(2, min(int(eye_width * 0.10), 7))  # Max base thickness = 7px

    print(f"DEBUG: thickness={base_thickness}, wing_length={wing_length}, wing_height={wing_height}, eye_end={eye_end}")

    num_segments = 5  # More segments for a smoother taper
    shade_bgr = (shade[2], shade[1], shade[0])  # Convert shade to BGR

    for i in range(num_segments):
        ratio = (i + 1) / num_segments  # Tapering factor
        start_point = (eye_end[0] + int(direction * ratio * wing_length * 0.4),
                       eye_end[1] - int(ratio * wing_height * 0.4))
        end_point = (eye_end[0] + int(direction * ratio * wing_length),
                     eye_end[1] - int(ratio * wing_height))

        # Thickness decreases gradually towards the tip
        thickness = max(1, int(base_thickness * (1 - ratio)))

        # Use the provided shade instead of black
        cv2.line(image_with_eyeliner, start_point, end_point, shade_bgr, thickness, cv2.LINE_AA)


def apply_eyeliner(image, left_eye_points, right_eye_points, color, transparency=0.85):
    if left_eye_points is None or right_eye_points is None:
        return image
    
    image_with_eyeliner = image.copy()
    color_bgr = (color[2], color[1], color[0])

    eye_width = abs(left_eye_points[-1][0] - left_eye_points[0][0])
    thickness = max(2, min(int(eye_width * 0.14), 8))

    image_with_eyeliner = draw_smooth_curve(image_with_eyeliner, left_eye_points, color_bgr, 2, thickness)
    image_with_eyeliner = draw_smooth_curve(image_with_eyeliner, right_eye_points, color_bgr, 2, thickness)
    
    draw_wing(image_with_eyeliner, left_eye_points, -1, color)
    draw_wing(image_with_eyeliner, right_eye_points, 1, color)
    
    blurred = cv2.GaussianBlur(image_with_eyeliner, (3, 3), 0.5)  # Slight blur for soft edges
    return cv2.addWeighted(image, 1 - transparency, blurred, transparency, 0)

def smooth_image(image):
    smoothed_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    return smoothed_image


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

# def apply_gradient_mask(mask):
#     """Apply a radial gradient to soften the outer edges of the eyeshadow mask."""
#     h, w = mask.shape
#     y, x = np.indices((h, w))
#     mask_indices = np.where(mask == 255)
#     if len(mask_indices[0]) == 0:
#         return mask  # No mask detected
#     # Find the center of the eyeshadow region
#     center_x, center_y = np.mean(mask_indices[1]), np.mean(mask_indices[0])
#     # Compute distance from the center
#     distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
#     max_distance = np.max(distance[mask == 255])
#     # Create a fading gradient effect
#     gradient = np.clip(1 - (distance / max_distance), 0, 1) * 255
#     return (mask * (gradient / 255)).astype(np.uint8)


def apply_gradient_mask(mask):
    """
    Creates a gradient effect on the mask, fading out gradually.
    """
    h, w = mask.shape
    gradient = np.linspace(1, 0, int(h * 0.3)).reshape(-1, 1)  # Vertical gradient fade
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

    left_eye_region = get_eyeshadow_region(landmarks, left_upper_eyelid_indices, w, h, extension_factor=0.03)
    right_eye_region = get_eyeshadow_region(landmarks, right_upper_eyelid_indices, w, h, extension_factor=0.03)


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

def apply_eyeshadow(image, mask_left, mask_right, color, intensity=0.85):
    """
    Apply a natural-looking eyeshadow effect with enhanced blending at the outer edges.
    """
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[mask_left > 0] = color
    overlay[mask_right > 0] = color

    # Increased Gaussian blur for a more natural diffusion
    blurred_overlay = cv2.GaussianBlur(overlay, (25, 25), 15)

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

def get_eyeshadow_region(landmarks, eyelid_indices, w, h ,extension_factor=0.05, corner_extension=0.03):
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




if __name__ == "__main__":
    try:
        DEMO_IMAGE = 'imgs/116.jpg'

        st.title('Make Over')

        visuareal_logo = Image.open('./imgs/Asset.png')
        st.sidebar.image(visuareal_logo, use_column_width=True)

        st.sidebar.title('How to get your virtual Glam')

        table = {
            'hair': 17,
            'upper_lip': 12,
            'lower_lip': 13,
            'eye_part_left' :4 ,
            'eye_part_right' : 5 ,
            'skin': 1,
            'eyes_u':3  
        }

        img_file_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", 'png'])

        if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))
            demo_image = img_file_buffer
        else:
            demo_image = DEMO_IMAGE
            image = np.array(Image.open(demo_image))

        new_image = image.copy()

        cp = 'cp/79999_iter.pth'
        ori = image.copy()
        h, w, _ = ori.shape
        image = cv2.resize(image, (1024, 1024))

        parsing = evaluate(demo_image, cp)
        parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)
    except:
        print('issue')

    # Get hair and lip colors from user input
    # import streamlit as st
    # from PIL import ImageColor

    # # Define color options
    # color_options = {
    #     'Red': '#FF0000',
    #     'Purple': '#800080',
    #     'Pink': '#FFC0CB'
    # }

    # # Check if hair_color is already stored in session_state
    # if 'hair_color' not in st.session_state:
    #     st.session_state.hair_color = None

    # # Add radio buttons for color selection
    # selected_color_name = st.sidebar.radio(
    #     'Pick the Hair Color',
    #     ('Red', 'Purple', 'Pink')
    # )

    # # Get the hex value of the selected color
    # selected_color = color_options[selected_color_name]

    # # Store the selected color in session_state
    # st.session_state.hair_color = ImageColor.getcolor(selected_color, "RGB")

    # # Use the stored hair color
    # hair_color = st.session_state.hair_color

    # # Apply hair and lip color modifications
    # image = hair(image, parsing, part=table['hair'], color=hair_color)


    # color_options = {
    #     'Red': '#FF0000',
    #     'Purple': '#800080',
    #     'Pink': '#52374a'
    # }

    # # Check if eyebrow_color is already stored in session_state
    # if 'eyebrow_color' not in st.session_state:
    #     st.session_state.eyebrow_color = None

    # # Add radio buttons for color selection
    # selected_color_name = st.sidebar.radio(
    #     'Pick the Eyebrow Color',
    #     ('Red', 'Purple', 'Pink')
    # )

    # # Get the hex value of the selected color
    # selected_color = color_options[selected_color_name]

    # # Store the selected color in session_state
    # st.session_state.eyebrow_color = ImageColor.getcolor(selected_color, "RGB")

    # # Use the stored eyebrow color
    # eyebrow_color = st.session_state.eyebrow_color

    def ensure_numpy_array(image):
        if not isinstance(image, np.ndarray):
            return np.array(image)
        return image


        # Load the base image (this part should load your base image correctly)

        # Define the buttons and their corresponding image paths and colors
        buttons = [
            {'label': 'lips Purple', 'image_path': 'colour.png', 'color': (138, 32, 189)},
            {'label': 'lips Red', 'image_path': 'colour2.png', 'color': (135, 25, 16)},
            {'label': 'Pink', 'image_path': 'colour3.png', 'color': (64, 41, 26)},
        ]

        # Loop through the buttons to create dynamic columns and handle button clicks
        for i, button in enumerate(buttons):
            # Load and resize the image
            image_path = button['image_path']
            button_image = Image.open(image_path)
            button_image = button_image.resize((50, 50))
            
            # Save the resized image
            resized_image_path = f'resized_image_{i}.png'
            button_image.save(resized_image_path)

            # Create two columns in the sidebar
            col1, col2 = st.sidebar.columns([1, 3])

            with col1:
                button_clicked = st.button(button['label'])

            with col2:
                st.image(button_image, use_column_width=False)

            # Main area processing
            if button_clicked:
                # Ensure image is a NumPy array
                image = ensure_numpy_array(image)
                # Process the image (replace this with your actual processing logic)
                image = lips(image, parsing, part=table['upper_lip'], color=button['color'])
                image = lips(image, parsing, part=table['lower_lip'], color=button['color'])


    # image = apply_eyebrow_color(image, parsing, eyebrow_color)


    # Define color options for iris
    # color_options = {
    #     'Blue': '#1d5161',
    #     'Green': '#1f4d22',
    #     'Brown': '#73411d'
    # }

    # Check if eye_color is already stored in session_state
    # if 'eye_color' not in st.session_state:
    #     st.session_state.eye_color = None

    # # Add radio buttons for color selection
    # selected_color_name = st.sidebar.radio(
    #     'Pick the Iris Color',
    #     ('Blue', 'Green', 'Brown')
    # )

    # # Get the hex value of the selected color
    # selected_color = color_options[selected_color_name]

    # # Store the selected color in session_state
    # st.session_state.eye_color = ImageColor.getcolor(selected_color, "RGB")

    # # Use the stored eye color
    # eye_color = st.session_state.eye_color

    # st.write(f"Selected iris color: {eye_color}")

    # # Add a checkbox for applying the iris color change
    # apply_iris_color_checkbox = st.sidebar.checkbox('Apply Iris Color Change')
    # if apply_iris_color_checkbox:
    #     image = apply_iris_color_change(image, face_mesh, eye_color)

    # #Checkbox to enable skin tone change

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
        


    # Define color options for blush
    color_options = {
        'Light Pink': '#5c1514',
        'Rose': '#FF007F',
        'Peach': '#FFDAB9'
    }

    # Check if blush_color is already stored in session_state
    if 'blush_color' not in st.session_state:
        st.session_state.blush_color = None

    # Add radio buttons for color selection
    selected_color_name = st.sidebar.radio(
        'Pick the Blush Color',
        ('Light Pink', 'Rose', 'Peach')
    )

    # Get the hex value of the selected color
    selected_color = color_options[selected_color_name]

    # Store the selected color in session_state
    st.session_state.blush_color = ImageColor.getcolor(selected_color, "RGB")

    # Use the stored blush color
    blush_color = st.session_state.blush_color

    st.write(f"Selected blush color: {blush_color}")

    # Add a checkbox for applying the blush
    apply_blush_checkbox = st.sidebar.checkbox('Apply Blush')
    if apply_blush_checkbox:
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mask_left, mask_right = get_cheek_masks(image, face_landmarks)
                image = apply_blush(image, mask_left, mask_right, color=blush_color)

    # # Contour application
    # apply_contour_checkbox = st.sidebar.checkbox('Apply Contour')
    # results = face_mesh.process(image)

    # Apply contour if the checkbox is checked
    # if apply_contour_checkbox:
    #     contour_color = st.sidebar.color_picker('Pick the Contour Color', '#6B4226')  # Darker shade for contour
    #     contour_color = ImageColor.getcolor(contour_color, "RGB")
    #     if results.multi_face_landmarks:
    #         for face_landmarks in results.multi_face_landmarks:
    #             mask_forehead_left, mask_forehead_right, mask_chin_left, mask_chin_right = get_highlight_contour_masks(image, face_landmarks)
    #             image = apply_contour(image, mask_forehead_left, mask_forehead_right, mask_chin_left, mask_chin_right, color=contour_color)

    # apply_concealer_checkbox = st.sidebar.checkbox('Apply Concealer (beta)')
    # if apply_concealer_checkbox:
    #    concealer_color = st.sidebar.color_picker('Pick the Concealer Color', '#FFE0BD')  # Default concealer color
    #    concealer_color = ImageColor.getcolor(concealer_color, "RGB")
        
    #    results = face_mesh.process(image)
        
    #    if results.multi_face_landmarks:
    #        for face_landmarks in results.multi_face_landmarks:
    #            mask_left, mask_right = get_concealer_masks(image, face_landmarks)
    #            image = apply_concealer(image, mask_left, mask_right, color=concealer_color)

    # results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # if results.multi_face_landmarks:
    #    for face_landmarks in results.multi_face_landmarks:
    #        image_with_landmarks = visualize_landmarks(image.copy(), face_landmarks)

    #    st.image(image_with_landmarks, caption="Image with Landmarks", use_column_width=True)

    #----------eyeliner---------
    # Checkbox to trigger the color picker and function calls
    # if st.sidebar.checkbox('Apply Eyeliner'):
    #     eyeliner_color = st.sidebar.color_picker('Pick an eyeliner color', '#ffffff')  # Default to black
    #     results = face_mesh.process(image)

    #     if results.multi_face_landmarks:
    #         for face_landmarks in results.multi_face_landmarks:
    #             image = draw_eyeliner(image, face_landmarks, eyeliner_color)



    # apply_eyeshadow_checkbox = st.sidebar.checkbox('Apply Eyeshadow')
    # if apply_eyeshadow_checkbox:
    #     eyeshadow_color = st.sidebar.color_picker('Pick the Eyeshadow Color', '#6A0DAD')  # Default to a deep purple
    #     eyeshadow_color_rgb = ImageColor.getcolor(eyeshadow_color, "RGB")

    #     # Process the image to find landmarks
    #     results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #     if results.multi_face_landmarks:
    #         for face_landmarks in results.multi_face_landmarks:
    #             mask_left, mask_right = create_eyeshadow_mask(image, face_landmarks)
    #             # Apply eyeshadow effect using the masks
    #             image = apply_eyeshadow(image, mask_left, mask_right, color=eyeshadow_color_rgb)

    image = cv2.resize(image, (w, h))
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Original Image')
        st.image(new_image, use_column_width=True)

    with col2:
        st.subheader('Output Image')
        st.image(image, use_column_width=True)
