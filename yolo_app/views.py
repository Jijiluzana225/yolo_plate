from django.http import JsonResponse
from ultralytics import YOLO  # Correct import for YOLOv8
import cv2
import numpy as np
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

# # Load YOLOv8 model (you can use a pre-trained model like 'yolov8n.pt' or any custom-trained model)
# model = YOLO('yolov8n.pt')  # Ensure you use the correct model path

def detect_objects(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        file_name = default_storage.save(uploaded_file.name, ContentFile(uploaded_file.read()))
        image_path = default_storage.path(file_name)
        
        # Read image with OpenCV
        img = cv2.imread(image_path)

        # Perform object detection
        results = model(img)  # Detect objects

        # Extract the detection results
        detections = []
        for box in results.xywh[0]:  # xywh contains [x, y, width, height, confidence, class_id]
            x, y, w, h, confidence, class_id = box
            detections.append({
                'class_id': int(class_id),
                'confidence': float(confidence),
                'bounding_box': [x, y, w, h]
            })

        return JsonResponse({'detections': detections})

    return JsonResponse({'error': 'No image uploaded or incorrect request.'})

import pytesseract
from django.http import StreamingHttpResponse
from ultralytics import YOLO
import cv2
import numpy as np

# Path to the YOLOv8 license plate model (update with your model path)
model = YOLO('yolo_app\yolo\license_plate_detector.pt')

# Initialize the camera capture
cap = cv2.VideoCapture(0)

# License plate class ID (adjust based on your model's class ID)
LICENSE_PLATE_CLASS_ID = 0  # Update with the correct class ID for license plates

# Define pytesseract executable path (if needed, adjust for your system)
# For Windows, use: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Example for Windows

def extract_plate_number(frame, x1, y1, x2, y2):
    # Crop the license plate from the frame using the bounding box coordinates
    cropped_plate = frame[int(y1):int(y2), int(x1):int(x2)]
    
    # Convert the cropped image to grayscale (optional, but can improve OCR accuracy)
    gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to improve text visibility (optional, experiment with different methods)
    _, thresh_plate = cv2.threshold(gray_plate, 150, 255, cv2.THRESH_BINARY)
    
    # Use Tesseract OCR to extract text from the cropped image
    plate_text = pytesseract.image_to_string(thresh_plate, config='--psm 8')  # PSM 8 is for single word text

    return plate_text.strip()


# Updated functions
def enhance_image_with_super_resolution(cropped_plate):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel('yolo_app\yolo\FSRCNN_x4.pb')  # Ensure this model file is in your project
    sr.setModel('fsrcnn', 4)
    return sr.upsample(cropped_plate)

import easyocr

reader = easyocr.Reader(['en'])
def extract_plate_number(frame, x1, y1, x2, y2):
    cropped_plate = frame[int(y1):int(y2), int(x1):int(x2)]
    result = reader.readtext(cropped_plate, detail=0)
    return result[0] if result else ''


from django.http import JsonResponse
import cv2
from ultralytics import YOLO
import numpy as np
from .models import CarInformation

# Initialize the YOLO model
model = YOLO('yolo_app/yolo/license_plate_detector.pt')

# Function to extract the plate number
def extract_plate_number(frame, x1, y1, x2, y2):
    cropped_plate = frame[int(y1):int(y2), int(x1):int(x2)]
    # Use OCR to detect the plate number (You can use easyOCR or pytesseract)
    plate_number = pytesseract.image_to_string(cropped_plate, config='--psm 8').strip()
    return plate_number

from django.http import JsonResponse
import cv2
from ultralytics import YOLO
import numpy as np
from .models import CarInformation
import pytesseract

# Initialize the YOLO model
model = YOLO('yolo_app/yolo/license_plate_detector.pt')

# Function to extract the plate number
# def extract_plate_number(frame, x1, y1, x2, y2):
#     cropped_plate = frame[int(y1):int(y2), int(x1):int(x2)]
#     # Use OCR to detect the plate number (You can use easyOCR or pytesseract)
#     plate_number = pytesseract.image_to_string(cropped_plate, config='--psm 8').strip()
#     return plate_number

import re

# Function to extract the plate number and remove special characters
def extract_plate_number(frame, x1, y1, x2, y2):
    cropped_plate = frame[int(y1):int(y2), int(x1):int(x2)]
    # Use OCR to detect the plate number (You can use easyOCR or pytesseract)
    plate_number = pytesseract.image_to_string(cropped_plate, config='--psm 8').strip()

    # Remove special characters (non-alphanumeric) from the detected plate number
    plate_number_cleaned = re.sub(r'[^a-zA-Z0-9]', '', plate_number)

    return plate_number_cleaned



def generate_frames():
    cap = cv2.VideoCapture(0)
    match_message = ""  # Initialize the match/no match message

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results[0].boxes

        for box in detections:
            class_id = int(box.cls[0].cpu().numpy())
            if class_id == 0:  # License plate class ID
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                plate_number = extract_plate_number(frame, x1, y1, x2, y2)

                # Convert the extracted plate number to lowercase for case-insensitive matching
                plate_number_lower = plate_number.lower()

                # Query the CarInformation model to check if the plate number exists (case-insensitive)
                try:
                    car_info = CarInformation.objects.get(plate_number__iexact=plate_number_lower)
                    # If a match is found, display the car details and color the message yellow
                    message = f"Plate Number: {car_info.plate_number}\n" \
                              f"Owner: {car_info.car_owner}\n" \
                              f"Address: {car_info.address}\n" \
                              f"Remarks: {car_info.remarks}"
                    color = (0, 255, 255)  # Yellow (BGR)
                    match_message = f"Match Found for Plate: {plate_number}"  # Set match message
                except CarInformation.DoesNotExist:
                    # If no match is found, display "NO MATCH" and color it red
                    message = "NO MATCH"
                    color = (0, 0, 255)  # Red (BGR)
                    match_message = f"No Match Found for Plate: {plate_number}"  # Set no match message

                # Draw a rectangle around the detected plate and add the message
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Split the message into lines and display each on a new line
                y_offset = int(y1) - 10  # Start just above the bounding box
                for line in message.split('\n'):
                    cv2.putText(frame, line, (int(x1), y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    y_offset += 30  # Add some space between the lines

        # Draw the overall match/no match message at the bottom of the frame
        cv2.putText(frame, match_message, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Encode the frame as JPEG and yield it to the HTTP response
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')



def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')





from django.shortcuts import render

def stream_page(request):
    return render(request, 'video_stream.html')

# from django.shortcuts import render
# from django.http import JsonResponse
# from django.views.decorators.http import require_http_methods
# from django.views.decorators.csrf import csrf_exempt
# import json
# import base64
# import numpy as np
# import cv2
# from ultralytics import YOLO
# import pytesseract
# import io

# # Load YOLO model
# model = YOLO('yolo_app/yolo/license_plate_detector.pt')  # Path to your YOLO model

# # CSRF Exempt and only POST method allowed for /yolo/process/
# @csrf_exempt  # Remove this in production, only for testing
# @require_http_methods(["POST"])
# def process_frame(request):
#     if request.method == 'POST':
#         try:
#             # Parse the request body
#             body = request.body.decode('utf-8')
#             data = json.loads(body)

#             # Check for image data in the body
#             if 'image' not in data:
#                 return JsonResponse({'error': 'No image data provided'}, status=400)

#             # Get the base64 encoded image data
#             image_data = data['image']

#             # Decode the base64 image
#             img_data = base64.b64decode(image_data.split(',')[1])  # Remove the data URL prefix
#             np_img = np.frombuffer(img_data, dtype=np.uint8)
#             frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

#             # Perform YOLO detection
#             results = model(frame)
#             detections = []
#             for box in results[0].boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                 class_id = int(box.cls[0].cpu().numpy())
#                 confidence = float(box.conf[0].cpu().numpy())
#                 detections.append({
#                     'class_id': class_id,
#                     'confidence': confidence,
#                     'bounding_box': [x1, y1, x2, y2]
#                 })

#             # Send the results back
#             return JsonResponse({'status': 'success', 'detections': detections})

#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)

#     return JsonResponse({'error': 'Invalid request method'}, status=405)

# # Stream page that renders the HTML page    
# def stream_page(request):
#     return render(request, 'video_stream.html')

import cv2
import numpy as np
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from .models import CarInformation
from django.core.files.uploadedfile import InMemoryUploadedFile

@require_http_methods(["POST"])
def compare_images(request):
    # Check if an image file was uploaded
    live_image = request.FILES.get('live_image')
    if not live_image:
        return JsonResponse({'error': 'No image uploaded'}, status=400)

    # Convert uploaded image to a NumPy array
    image_bytes = np.frombuffer(live_image.read(), np.uint8)
    uploaded_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    # Iterate through all CarInformation entries
    for car in CarInformation.objects.all():
        if car.picture:
            stored_image_path = car.picture.path
            stored_image = cv2.imread(stored_image_path)

            # Compare images using SSIM (Structural Similarity Index)
            if stored_image is not None:
                stored_image_gray = cv2.cvtColor(stored_image, cv2.COLOR_BGR2GRAY)
                uploaded_image_gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)

                # Resize images to the same dimensions if necessary
                resized_stored_image = cv2.resize(stored_image_gray, (uploaded_image_gray.shape[1], uploaded_image_gray.shape[0]))

                # Compute similarity score
                similarity = cv2.matchTemplate(uploaded_image_gray, resized_stored_image, cv2.TM_CCOEFF_NORMED).max()

                if similarity > 0.8:  # Adjust threshold as needed
                    return JsonResponse({
                        'status': 'Match Found',
                        'car_details': {
                            'plate_number': car.plate_number,
                            'car_owner': car.car_owner,
                            'address': car.address,
                            'classification': car.classification.name,
                            'remarks': car.remarks,
                        }
                    })

    # No matches found
    return JsonResponse({'status': 'No Match'})



def calculate_image_similarity(image1, image2):
    """
    Compare two images using histogram similarity.
    Returns a similarity score between 0 and 1.
    """
    # Convert both images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate histograms
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

    # Normalize histograms
    hist1 = cv2.normalize(hist1, hist1)
    hist2 = cv2.normalize(hist2, hist2)

    # Calculate the correlation similarity
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity

