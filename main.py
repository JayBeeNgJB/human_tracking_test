from deepface import DeepFace
from ultralytics import YOLO
import cv2
import os
import json
from datetime import datetime
import numpy as np

class VisitorTracker:
    def __init__(self, database_folder="visitor_database", log_file="visitor_log.json"):
        self.database_folder = database_folder
        self.log_file = log_file
        self.visitors = []
        self.current_visitor = None
        self.visitor_count = 0
        self.distance_threshold = 0.5  
        
        # Initialize YOLOv8 model
        print("Loading YOLOv8 model...")
        self.yolo_model = YOLO('yolov8n.pt')  # Use nano model for speed
        self.person_class_id = 0  # Person class in COCO dataset
        
        # Track IDs to Visitor IDs mapping
        self.track_to_visitor = {}  # {yolo_track_id: visitor_id}
        self.track_last_processed = {}  # {yolo_track_id: frame_number}
        self.active_tracks = set()  # Currently visible track IDs
        
        if not os.path.exists(self.database_folder):
            os.makedirs(self.database_folder)
            print(f"Created visitor database folder: {self.database_folder}")
        
        self.load_visitor_log()
    
    def extract_face_region(self, frame):
        """Extract face region from frame using OpenCV"""
        try:
            # Use OpenCV's face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Get the largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face
                
                # Add padding
                padding = int(w * 0.2)
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2*padding)
                h = min(frame.shape[0] - y, h + 2*padding)
                
                face_region = frame[y:y+h, x:x+w]
                return face_region, True
            
            return frame, False
            
        except Exception as e:
            print(f"Face extraction error: {e}")
            return frame, False
    
    def align_and_save_face(self, frame, save_path):
        """Extract, align, and save face"""
        try:
            # First try to extract just the face region
            face_frame, face_found = self.extract_face_region(frame)
            
            if not face_found:
                print("  [DEBUG] No face found with OpenCV, using full frame")
                cv2.imwrite(save_path, frame)
                return False
            
            print("  [DEBUG] Face detected with OpenCV")
            
            # Save the face region
            temp_path = "temp_align.jpg"
            cv2.imwrite(temp_path, face_frame)
            
            # Try DeepFace alignment
            try:
                face_objs = DeepFace.extract_faces(
                    img_path=temp_path,
                    enforce_detection=False,
                    detector_backend='opencv',
                    align=True
                )
                
                if face_objs and len(face_objs) > 0:
                    aligned_face = face_objs[0]['face']
                    aligned_face = (aligned_face * 255).astype('uint8')
                    cv2.imwrite(save_path, aligned_face)
                    print("  [DEBUG] Face aligned with DeepFace")
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    return True
            except:
                pass
            
            # Fallback: use OpenCV detected face
            cv2.imwrite(save_path, face_frame)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return True
            
        except Exception as e:
            print(f"  [DEBUG] Alignment failed: {e}, using original")
            cv2.imwrite(save_path, frame)
            return False
    
    def load_visitor_log(self):
        """Load previous visitor records"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                data = json.load(f)
                self.visitor_count = data.get('total_visitors', 0)
                self.visitors = data.get('visitors', [])
                print(f"Loaded {self.visitor_count} previous visitors from log")
        else:
            print("No previous visitor log found. Starting fresh.")
    
    def save_visitor_log(self):
        """Save visitor records to file"""
        data = {
            'total_visitors': self.visitor_count,
            'visitors': self.visitors
        }
        with open(self.log_file, 'w') as f:
            json.dump(data, indent=4, fp=f)
    
    def is_known_visitor(self, current_frame_path):
        """Check if this person has visited before"""
        if self.visitor_count == 0:
            print("  [DEBUG] No previous visitors")
            return False, None
        
        print(f"  [DEBUG] Checking against {self.visitor_count} previous visitors")
        
        for i in range(1, self.visitor_count + 1):
            visitor_image = os.path.join(self.database_folder, f"visitor_{i}.jpg")
            
            if not os.path.exists(visitor_image):
                continue
            
            try:
                result = DeepFace.verify(
                    img1_path=visitor_image,
                    img2_path=current_frame_path,
                    enforce_detection=False,
                    model_name="VGG-Face"
                )
                
                print(f"    Visitor {i}: distance={result['distance']:.4f}, verified={result['verified']}")
                
                if result['verified'] and result['distance'] < self.distance_threshold:
                    print(f"  [DEBUG] Matched to Visitor #{i}!")
                    return True, i
                    
            except Exception as e:
                print(f"    Visitor {i}: Error - {e}")
                continue
        
        print("  [DEBUG] No match found - new visitor")
        return False, None
    
    def add_new_visitor(self, frame):
        """Add a new unique visitor"""
        self.visitor_count += 1
        visitor_id = self.visitor_count
        
        visitor_image_path = os.path.join(self.database_folder, f"visitor_{visitor_id}.jpg")
        self.align_and_save_face(frame, visitor_image_path)
        
        visitor_info = {
            'id': visitor_id,
            'first_seen': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'image_path': visitor_image_path,
            'visit_count': 1
        }
        self.visitors.append(visitor_info)
        self.save_visitor_log()
        
        print(f"✓ New visitor #{visitor_id} registered!")
        return visitor_id
    
    def update_visitor_timestamp(self, visitor_id):
        """Update the last seen time for a returning visitor"""
        for visitor in self.visitors:
            if visitor['id'] == visitor_id:
                visitor['last_seen'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                visitor['visit_count'] = visitor.get('visit_count', 1) + 1
                self.save_visitor_log()
                print(f"✓ Updated Visitor #{visitor_id} - Visit count: {visitor['visit_count']}")
                break
    
    def process_person_detection(self, frame, track_id, bbox, frame_count):
        """Process detected person with face recognition"""
        print(f"\n[Track {track_id}] Processing...")
        
        # Check if already processed recently
        if track_id in self.track_to_visitor:
            visitor_id = self.track_to_visitor[track_id]
            print(f"[Track {track_id}] Already mapped to Visitor #{visitor_id}")
            return visitor_id, f"Visitor #{visitor_id}", (0, 255, 0)
        
        # Check if we processed this track too recently (wait at least 30 frames)
        if track_id in self.track_last_processed:
            if frame_count - self.track_last_processed[track_id] < 30:
                print(f"[Track {track_id}] Processed too recently, skipping")
                return None, "Processing...", (255, 255, 0)
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Expand bounding box to get more context (especially upper body for face)
        height = y2 - y1
        width = x2 - x1
        
        # Focus on upper portion where face typically is
        y2_new = int(y1 + height * 0.6)  # Take top 60% of person detection
        
        # Add horizontal padding
        padding_x = int(width * 0.1)
        x1 = max(0, x1 - padding_x)
        x2 = min(frame.shape[1], x2 + padding_x)
        y1 = max(0, y1)
        y2 = min(frame.shape[0], y2_new)
        
        # Extract person region (upper body focus)
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            print(f"[Track {track_id}] Invalid crop size")
            return None, "Invalid crop", (255, 255, 0)
        
        print(f"[Track {track_id}] Crop size: {person_crop.shape}")
        
        try:
            # Save temporary frame for face recognition
            temp_path = f"temp_track_{track_id}.jpg"
            
            # Extract and save face
            face_detected = self.align_and_save_face(person_crop, temp_path)
            
            if not face_detected:
                print(f"[Track {track_id}] No face detected in crop")
                self.track_last_processed[track_id] = frame_count
                return None, "No face detected", (255, 165, 0)
            
            # Check against existing visitors
            is_known, visitor_id = self.is_known_visitor(temp_path)
            
            if is_known:
                # Map this track to known visitor
                self.track_to_visitor[track_id] = visitor_id
                self.track_last_processed[track_id] = frame_count
                self.update_visitor_timestamp(visitor_id)
                status = f"Visitor #{visitor_id}"
                color = (0, 255, 0)
                print(f"[Track {track_id}] ✓ Recognized as Visitor #{visitor_id}")
            else:
                # Register new visitor
                new_visitor_id = self.add_new_visitor(person_crop)
                self.track_to_visitor[track_id] = new_visitor_id
                self.track_last_processed[track_id] = frame_count
                visitor_id = new_visitor_id
                status = f"NEW Visitor #{new_visitor_id}!"
                color = (255, 0, 255)
                print(f"[Track {track_id}] ✓ Registered as NEW Visitor #{new_visitor_id}")
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return visitor_id, status, color
            
        except Exception as e:
            print(f"[Track {track_id}] Error: {e}")
            self.track_last_processed[track_id] = frame_count
            return None, "Error processing", (255, 0, 0)

def track_visitors():
    tracker = VisitorTracker()
    
    cap = cv2.VideoCapture(0)
    
    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n" + "="*60)
    print("Starting YOLOv8 + DeepFace Visitor Tracking System")
    print("="*60)
    print(f"Current unique visitors: {tracker.visitor_count}")
    print(f"Detection threshold: {tracker.distance_threshold}")
    print("Press 'q' to quit | 'r' to reset | 'd' to toggle debug")
    print("="*60 + "\n")
    
    frame_count = 0
    process_interval = 20  # Process face recognition every 20 frames (~0.66 sec)
    detected_persons = {}  # Store detection info per track
    debug_mode = True  # Show debug info
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Run YOLO tracking on every frame
        results = tracker.yolo_model.track(
            frame,
            persist=True,
            classes=[tracker.person_class_id],  # Only detect persons
            conf=0.5,  # Confidence threshold
            verbose=False
        )
        
        current_frame_tracks = set()
        
        # Process detections
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            track_ids = boxes.id.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            
            for track_id, conf, bbox in zip(track_ids, confidences, xyxy):
                current_frame_tracks.add(track_id)
                
                # Process face recognition periodically OR if it's a new track
                should_process = (frame_count % process_interval == 0) or (track_id not in detected_persons)
                
                if should_process:
                    visitor_id, status, color = tracker.process_person_detection(
                        frame, track_id, bbox, frame_count
                    )
                    detected_persons[track_id] = {
                        'visitor_id': visitor_id,
                        'status': status,
                        'color': color,
                        'bbox': bbox,
                        'conf': conf
                    }
                
                # Draw bounding box and info
                if track_id in detected_persons:
                    info = detected_persons[track_id]
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), info['color'], 2)
                    
                    # Draw track ID and status
                    label = f"Track {track_id}: {info['status']}"
                    
                    # Background for text
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                                (x1 + text_width + 10, y1), info['color'], -1)
                    
                    # Text
                    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Show confidence
                    conf_text = f"{conf:.2f}"
                    cv2.putText(frame, conf_text, (x1, y2 + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Clean up tracks that are no longer visible
        tracker.active_tracks = current_frame_tracks
        removed_tracks = set(detected_persons.keys()) - current_frame_tracks
        for track_id in removed_tracks:
            if track_id in detected_persons:
                print(f"\n[Track {track_id}] Left frame")
                del detected_persons[track_id]
        
        # Display statistics UI
        ui_height = 140 if debug_mode else 120
        cv2.rectangle(frame, (10, 10), (630, ui_height), (0, 0, 0), -1)
        
        status_text = f"Active Persons: {len(current_frame_tracks)}"
        cv2.putText(frame, status_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        count_text = f"Total Unique Visitors: {tracker.visitor_count}"
        cv2.putText(frame, count_text, (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if debug_mode:
            mapped_text = f"Mapped Tracks: {len(tracker.track_to_visitor)}"
            cv2.putText(frame, mapped_text, (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        cv2.putText(frame, "Q:quit | R:reset | D:debug", (20, ui_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        frame_count += 1
        cv2.imshow('YOLOv8 + DeepFace Visitor Tracking', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            response = input("\nReset visitor count? (yes/no): ")
            if response.lower() == 'yes':
                tracker.visitor_count = 0
                tracker.visitors = []
                tracker.track_to_visitor = {}
                tracker.track_last_processed = {}
                tracker.save_visitor_log()
                detected_persons = {}
                print("✓ Visitor count reset!")
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Cleanup temp files
    for temp_file in ['temp_frame.jpg', 'temp_align.jpg']:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    # Clean up any remaining track temp files
    for f in os.listdir('.'):
        if f.startswith('temp_track_') and f.endswith('.jpg'):
            os.remove(f)
    
    print(f"\n{'='*60}")
    print(f"Session Summary:")
    print(f"Total Unique Visitors: {tracker.visitor_count}")
    print(f"{'='*60}")

if __name__ == "__main__":
    track_visitors()