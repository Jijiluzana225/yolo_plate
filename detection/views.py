from django.http import StreamingHttpResponse
from django.shortcuts import render
from .models import ReferenceImage
from .utils import detect_objects, compare_images
import cv2

def video_feed(request):
    reference_image = ReferenceImage.objects.first()  # Get the first reference image

    if not reference_image:
        return StreamingHttpResponse("No reference image found.", status=404)

    reference_path = reference_image.image.path

    def generate_frames():
        cap = cv2.VideoCapture(0)  # Use webcam
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Perform object detection and comparison
            detected_objects = detect_objects(frame)
            comparison_score = compare_images(reference_path, frame)

            # Check if comparison score exceeds the threshold for match
            match_found = comparison_score > 125  # Match if score > 110

            # Draw detection boxes for detected objects
            for box in detected_objects:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Overlay comparison score on the frame
            cv2.putText(frame, f"Score: {comparison_score}" , (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # If a match is found and score exceeds 110, overlay the reference image name
            if match_found:
                cv2.putText(frame, f"Matched: {reference_image.name}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Encode and yield the frame to display
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return StreamingHttpResponse(generate_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
