import cv2
import face_recognition
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import face_recognition

class FaceRecognitionSystem:
    def __init__(self, similarity_threshold=0.6, tracker_overlap_threshold=0.4):
        self.unique_faces = []
        self.face_trackers = {}
        self.current_face_id = 0
        self.similarity_threshold = similarity_threshold
        self.tracker_overlap_threshold = tracker_overlap_threshold
        
        # Load YOLO model for person detection
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.yolo_model.classes = [0]  # Only detect persons (class 0 in COCO dataset)
    
    def detect_persons(self, frame):
        """Detect persons in the frame using YOLO."""
        results = self.yolo_model(frame)
        detections = results.xyxy[0].cpu().numpy()
        
        person_boxes = []
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if cls == 0:  # Class 0 is 'person'
                person_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        
        return person_boxes
    
    def is_unique(self, embedding, threshold=None):
        """Check if a face embedding is unique compared to stored embeddings."""
        if threshold is None:
            threshold = self.similarity_threshold
            
        embeddings_list = [f['embedding'] for f in self.unique_faces]
        
        if len(embeddings_list) == 0:
            return True  # If no embeddings exist, it's the first unique face
        
        distances = euclidean_distances([embedding], embeddings_list)
        
        return np.min(distances) > threshold
    
    def process_frame(self, frame):
        """Process a video frame to detect faces and extract embeddings."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        return face_encodings, face_locations
    
    def create_face_tracker(self, frame, location):
        """Initialize a new face tracker for a given face location."""
        tracker = cv2.TrackerCSRT_create()
        top, right, bottom, left = location
        bbox = (left, top, right - left, bottom - top)
        tracker.init(frame, bbox)
        return tracker

    def track_faces(self, frame, draw=True):
        """Track faces over time and optionally draw bounding boxes."""
        updated_trackers = {}
        for face_id, tracker in self.face_trackers.items():
            success, bbox = tracker.update(frame)
            if success:
                updated_trackers[face_id] = tracker
                
                if draw:
                    (left, top, width, height) = [int(v) for v in bbox]
                    cv2.rectangle(frame, (left, top), (left + width, top + height), (255, 0, 0), 2)
                    cv2.putText(frame, f"ID {face_id}", (left, top - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        self.face_trackers = updated_trackers
        return frame
    
    def bbox_overlap(self, bbox1, bbox2):
        """Calculate overlap between two bounding boxes (IOU)."""
        left1, top1, width1, height1 = bbox1
        left2, top2, width2, height2 = bbox2
        
        # Calculate overlap area
        x_overlap = max(0, min(left1 + width1, left2 + width2) - max(left1, left2))
        y_overlap = max(0, min(top1 + height1, top2 + height2) - max(top1, top2))
        overlap_area = x_overlap * y_overlap
        
        bbox1_area = width1 * height1
        bbox2_area = width2 * height2
        
        # Calculate IOU
        iou = overlap_area / float(bbox1_area + bbox2_area - overlap_area)
        return iou
    
    def process_video(self, video_path, skip_frames=10, display=True, verbose=True):
        """Process a video and count unique faces."""
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        unique_faces_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit loop if video ends
            
            frame_count += 1
            
            # Detect persons using YOLO
            person_boxes = self.detect_persons(frame)
            
            for person_box in person_boxes:
                x1, y1, x2, y2 = person_box
                person_frame = frame[y1:y2, x1:x2]  # Crop the person region
                
                # Update trackers and draw bounding boxes
                frame = self.track_faces(frame, draw=display)
                
                if frame_count % skip_frames == 0:
                    face_encodings, face_locations = self.process_frame(person_frame)
                    
                    for encoding, location in zip(face_encodings, face_locations):
                        # Adjust face location to the original frame coordinates
                        adjusted_location = (
                            location[0] + y1,
                            location[1] + x1,
                            location[2] + y1,
                            location[3] + x1
                        )
                        
                        face_bbox = (
                            adjusted_location[3],
                            adjusted_location[0],
                            adjusted_location[1] - adjusted_location[3],
                            adjusted_location[2] - adjusted_location[0]
                        )
                        
                        is_new_face = True
                        for face_id, tracker in self.face_trackers.items():
                            success, tracked_bbox = tracker.update(frame)
                            if success and self.bbox_overlap(face_bbox, tracked_bbox) > self.tracker_overlap_threshold:
                                is_new_face = False
                                break
                        
                        if is_new_face and self.is_unique(encoding):
                            face_tracker = self.create_face_tracker(frame, adjusted_location)
                            self.face_trackers[self.current_face_id] = face_tracker
                            self.unique_faces.append({'id': self.current_face_id, 'embedding': encoding})
                            self.current_face_id += 1
                            unique_faces_count += 1
            
            if verbose and frame_count % 50 == 0:
                print(f"Processed {frame_count} frames, Unique faces so far: {unique_faces_count}")
            
            if display:
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        return unique_faces_count

