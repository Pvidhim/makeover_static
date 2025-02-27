# import cv2
# import dlib
# import numpy as np
# import matplotlib.pyplot as plt
# import random
# import faceBlendCommon as fbc

# # Set up matplotlib
# plt.rcParams['figure.figsize'] = (8.0, 8.0)
# plt.rcParams['image.cmap'] = 'gray'
# plt.rcParams['image.interpolation'] = 'bilinear'

# # Landmark model location
# PREDICTOR_PATH = r"C:\Users\DEEKSHITA\OneDrive\Desktop\Visuareal\Makeover-main\shape_predictor_68_face_landmarks.dat"

# # Initialize face and landmark detectors
# faceDetector = dlib.get_frontal_face_detector()
# landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

# # Load the uploaded image
# IMAGE_PATH = "/mnt/data/image.png"
# im = cv2.imread(IMAGE_PATH)
# imDlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# # Display the original image
# plt.imshow(imDlib)
# plt.axis('off')
# plt.title("Original Image")
# plt.show()

# # Define lipstick colors
# lipstick_colors = {
#     "vamptastic_plum": (97, 45, 130),
#     "red_dahlia": (51, 30, 136),
#     "flamenco_red": (42, 31, 192),
#     "cherry_red": (63, 45, 222),
#     "caramel_nude": (120, 131, 201),
#     "mango_tango": (103, 92, 223),
#     "neon_red": (79, 32, 223),
#     "electric_orchid": (139, 64, 243),
#     "forbidden_fuchsia": (105, 39, 184),
#     "sweet_marsala": (93, 67, 164),
# }

# def getLipsMask(size, lips):
#     """Creates a refined mask for the lips."""
#     hullIndex = cv2.convexHull(np.array(lips), returnPoints=False)
#     hullInt = [lips[hIndex[0]] for hIndex in hullIndex]
#     mask = np.zeros((size[0], size[1], 3), dtype=np.uint8)
#     cv2.fillConvexPoly(mask, np.int32(hullInt), (255, 255, 255))
#     return mask

# def refine_mask(mask):
#     """Refines the mask to smooth edges and improve realism."""
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     mask = cv2.GaussianBlur(mask, (15, 15), 0)
#     return mask



# def apply_color_to_mask(mask, selected_color=None):
#     """Applies lipstick color to the mask."""
#     if selected_color is None:
#         color_name, color = random.choice(list(lipstick_colors.items()))
#     else:
#         color_name, color = selected_color, lipstick_colors[selected_color]
#     print(f"[INFO] Lipstick Color: {color_name}")
#     b, g, r = cv2.split(mask)
#     b = np.where(b > 0, color[0], 0).astype('uint8')
#     g = np.where(g > 0, color[1], 0).astype('uint8')
#     r = np.where(r > 0, color[2], 0).astype('uint8')
#     return cv2.merge((b, g, r)), color_name

# def alphaBlend(alpha, foreground, background):
#     """Blends the mask with the image using alpha blending."""
#     fore = cv2.multiply(alpha, foreground, scale=1 / 255.0)
#     alphaPrime = 255 - alpha
#     back = cv2.multiply(alphaPrime, background, scale=1 / 255.0)
#     return cv2.add(fore, back)

# # Process the image
# try:
#     # Step 1: Get facial landmarks
#     points = fbc.getLandmarks(faceDetector, landmarkDetector, imDlib)

#     # Debugging: Draw landmarks
#     for point in points:
#         cv2.circle(imDlib, point, 2, (0, 255, 0), -1)
#     plt.imshow(imDlib)
#     plt.axis('off')
#     plt.title("Landmark Detection")
#     plt.show()

#     # Step 2: Isolate lips and create masks
#     lips = [points[x] for x in range(48, 68)]
#     mouth = [points[x] for x in range(60, 68)]

#     mask = getLipsMask(im.shape, lips)
#     mouth_mask = getLipsMask(im.shape, mouth)
#     mouth_mask = cv2.bitwise_not(mouth_mask)

#     # Combine masks and refine
#     mask = cv2.bitwise_and(mask, mask, mask=mouth_mask[:, :, 0])
#     mask = refine_mask(mask)

#     # Debugging: Show the lip mask
#     plt.imshow(mask[:, :, 0], cmap='gray')
#     plt.axis('off')
#     plt.title("Lip Mask")
#     plt.show()

#     # Step 3: Apply lipstick color
#     color_mask, color_name = apply_color_to_mask(mask)


#     # Debugging: Display colored lip mask
#     plt.imshow(cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.title(f"Colored Lip Mask with Gloss ({color_name})")
#     plt.show()

#     # Step 5: Blend color mask with the image
#     final = alphaBlend(mask, color_mask, im)

#     # Debugging: Show the final output
#     plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.title("Final Image with Lipstick")
#     plt.show()

#     # Save the final output
#     cv2.imwrite("final_lipstick_image.jpg", final)

# except Exception as e:
#     print(f"Error: {e}")
