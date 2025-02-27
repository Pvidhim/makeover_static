import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, static_image_mode=False, min_detection_confidence=0.5)

def draw_landmarks_with_numbers(image, landmarks):
    """Draw facial landmarks and their indices on the image."""
    h, w, _ = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    for idx, landmark in enumerate(landmarks):
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # Green dot
        cv2.putText(image, str(idx), (x + 5, y + 5), font, font_scale, (255, 255, 255), thickness)  # White number

    return image

def main():
    # Capture video from the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to fetch frame from camera")
                break

            # Convert the BGR image to RGB before processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the RGB frame to find facial landmarks
            results = face_mesh.process(rgb_frame)

            # If landmarks are detected, draw them on the frame
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    frame = draw_landmarks_with_numbers(frame, face_landmarks.landmark)

            # Display the frame
            cv2.imshow('FaceMesh Landmarks with Numbers', frame)

            # Exit loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()

if __name__ == "__main__":
    main()