import cv2
import numpy as np
import json
from channels.generic.websocket import WebsocketConsumer
from asgiref.sync import async_to_sync

class VideoConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        # Decode the received frame from the WebSocket
        frame_data = json.loads(text_data)
        frame = np.array(frame_data['frame'])

        # Process the frame (perform object detection or other processing)
        results = model(frame)  # Use your YOLOv8 model or other detection logic
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            confidence = box.conf[0].cpu().numpy()
            detections.append({'class_id': class_id, 'confidence': confidence, 'bounding_box': [x1, y1, x2, y2]})

        # Send the results back to the client
        self.send(text_data=json.dumps({'detections': detections}))
