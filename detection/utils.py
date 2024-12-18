import os
import cv2
from ultralytics import YOLO
import numpy as np

# Initialize YOLO model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'yolo_app', 'yolo', 'license_plate_detector.pt')
# model_path = os.path.join('yolov8n.pt')
model = YOLO(model_path)

def detect_objects(image):
    # Perform object detection
    results = model(image)
    
    # Extract bounding boxes
    detected_objects = []
    for box in results[0].boxes.xyxy:  # Access bounding boxes
        detected_objects.append(box.tolist())
    
    return detected_objects

import cv2

def compare_images(reference_image_path, livestream_frame):
    """
    Compare a reference image with a livestream frame using ORB feature matching.

    Args:
        reference_image_path (str): Path to the reference image.
        livestream_frame (np.ndarray): The current frame from the livestream.

    Returns:
        int: The number of matching features between the reference image and the livestream frame.
    """
    # Convert to grayscale for comparison
    ref_img = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    frame_gray = cv2.cvtColor(livestream_frame, cv2.COLOR_BGR2GRAY)

    # Validate images
    if ref_img is None:
        raise FileNotFoundError(f"Reference image not found at {reference_image_path}.")
    if frame_gray is None:
        raise ValueError("Invalid livestream frame provided.")

    # Use ORB for feature detection and descriptor extraction
    orb = cv2.ORB_create(nfeatures=500)  # Limit features to 500 for performance
    kp1, des1 = orb.detectAndCompute(ref_img, None)
    kp2, des2 = orb.detectAndCompute(frame_gray, None)

    # Handle cases where descriptors are None
    if des1 is None or des2 is None:
        return 0  # No matches if features are not detected in one or both images

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Optional: Filter out matches by a maximum distance threshold (e.g., good matches)
    good_matches = [m for m in matches if m.distance < 75]  # Adjust threshold as needed

    # Return the number of good matches
    return len(good_matches)
