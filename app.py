import cv2
import torch
import numpy as np
from PIL import Image
import time
from collections import defaultdict, deque

# Try to import required modules
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Transformers not found. Please install it using:")
    print("pip install transformers")
    TRANSFORMERS_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("YOLOv8 not found. Please install it using:")
    print("pip install ultralytics")
    YOLO_AVAILABLE = False

class PersonActivityTrackerTransformers:
    def __init__(self):
        # Check if required modules are available
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers module not found. Please install it first.")
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO module not found. Please install ultralytics first.")
            
        # Setup device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load YOLOv8 model for person detection
        print("Loading YOLOv8 model...")
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            print("YOLOv8 model loaded successfully!")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            raise
        
        # Load BLIP model for image captioning and classification
        print("Loading BLIP model for activity recognition...")
        try:
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
            print("BLIP model loaded successfully!")
        except Exception as e:
            print(f"Error loading BLIP model: {e}")
            raise
        
        # Alternative: Use zero-shot image classification pipeline
        try:
            print("Loading zero-shot classification pipeline...")
            self.classifier = pipeline(
                "zero-shot-image-classification",
                model="openai/clip-vit-base-patch32",
                device=0 if torch.cuda.is_available() else -1
            )
            self.use_classifier = True
            print("Zero-shot classifier loaded successfully!")
        except Exception as e:
            print(f"Zero-shot classifier not available: {e}")
            self.use_classifier = False
        
        # Activity labels
        self.activities = [
            "person sitting on chair",
            "person standing upright", 
            "person walking",
            "person sleeping in bed",
            "person talking on phone",
            "person running fast",
            "person lying down",
            "person raising hand up",
            "person eating food",
            "person reading book"
        ]
        
        # Person tracking
        self.person_tracks = {}
        self.track_id_counter = 0
        self.activity_history = defaultdict(lambda: deque(maxlen=8))
        
        # Colors for different activities
        self.activity_colors = {
            "sitting": (0, 255, 0),      # Green
            "standing": (255, 0, 0),     # Blue
            "walking": (0, 165, 255),    # Orange
            "sleeping": (128, 0, 128),   # Purple
            "talking": (255, 255, 0),    # Cyan
            "running": (0, 0, 255),      # Red
            "lying": (147, 20, 255),     # Pink
            "raising": (0, 255, 255),    # Yellow
            "eating": (255, 128, 0),     # Light Blue
            "reading": (128, 255, 128),  # Light Green
            "unknown": (255, 255, 255)   # White
        }
    
    def detect_persons(self, frame):
        """Detect persons using YOLOv8"""
        results = self.yolo_model(frame, classes=[0])  # Class 0 is 'person'
        persons = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    conf = box.conf.cpu().numpy()[0]
                    if conf > 0.5:  # Confidence threshold
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
                        persons.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf
                        })
        
        return persons
    
    def classify_activity_with_blip(self, person_crop):
        """Classify person activity using BLIP image captioning"""
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
            
            # Generate caption
            inputs = self.blip_processor(pil_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50, num_beams=5)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            # Analyze caption for activities
            caption_lower = caption.lower()
            detected_activity = "unknown"
            confidence = 0.5
            
            activity_keywords = {
                "sitting": ["sitting", "seated", "chair", "bench"],
                "standing": ["standing", "upright", "pose"],
                "walking": ["walking", "moving", "step"],
                "sleeping": ["sleeping", "lying", "bed", "rest"],
                "talking": ["talking", "phone", "speaking"],
                "running": ["running", "jogging", "fast"],
                "lying": ["lying down", "horizontal", "floor"],
                "raising": ["raising", "hand up", "waving"],
                "eating": ["eating", "food", "meal"],
                "reading": ["reading", "book", "paper"]
            }
            
            for activity, keywords in activity_keywords.items():
                for keyword in keywords:
                    if keyword in caption_lower:
                        detected_activity = activity
                        confidence = 0.8
                        break
                if detected_activity != "unknown":
                    break
            
            return detected_activity, confidence, caption
            
        except Exception as e:
            print(f"Error in BLIP classification: {e}")
            return "unknown", 0.0, ""
    
    def classify_activity_with_clip(self, person_crop):
        """Classify person activity using zero-shot classification"""
        try:
            if not self.use_classifier:
                return "unknown", 0.0
                
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
            
            # Classify with zero-shot classifier
            results = self.classifier(pil_image, self.activities)
            
            # Get best result
            best_result = results[0]
            activity_full = best_result['label']
            confidence = best_result['score']
            
            # Extract main activity word
            activity = "unknown"
            for key in self.activity_colors.keys():
                if key in activity_full.lower():
                    activity = key
                    break
            
            return activity, confidence
            
        except Exception as e:
            print(f"Error in zero-shot classification: {e}")
            return "unknown", 0.0
    
    def classify_activity(self, person_crop):
        """Main activity classification method"""
        if self.use_classifier:
            return self.classify_activity_with_clip(person_crop)
        else:
            activity, confidence, caption = self.classify_activity_with_blip(person_crop)
            return activity, confidence
    
    def simple_tracker(self, current_persons, threshold=50):
        """Simple person tracker based on bounding box overlap"""
        tracked_persons = []
        
        for current_person in current_persons:
            x1, y1, x2, y2 = current_person['bbox']
            current_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            best_match_id = None
            min_distance = float('inf')
            
            # Find closest previous person
            for prev_id, prev_person in self.person_tracks.items():
                if 'center' in prev_person:
                    prev_center = prev_person['center']
                    distance = np.sqrt((current_center[0] - prev_center[0])**2 + 
                                     (current_center[1] - prev_center[1])**2)
                    
                    if distance < min_distance and distance < threshold:
                        min_distance = distance
                        best_match_id = prev_id
            
            # Assign ID
            if best_match_id is not None:
                track_id = best_match_id
            else:
                track_id = self.track_id_counter
                self.track_id_counter += 1
            
            # Update tracking info
            self.person_tracks[track_id] = {
                'bbox': current_person['bbox'],
                'center': current_center,
                'last_seen': time.time()
            }
            
            tracked_persons.append({
                'id': track_id,
                'bbox': current_person['bbox'],
                'confidence': current_person['confidence']
            })
        
        # Clean up old tracks
        current_time = time.time()
        self.person_tracks = {
            k: v for k, v in self.person_tracks.items() 
            if current_time - v['last_seen'] < 3.0  # Remove tracks older than 3 seconds
        }
        
        return tracked_persons
    
    def smooth_activity(self, person_id, activity, confidence):
        """Smooth activity prediction using history"""
        if confidence > 0.4:  # Only add reasonably confident predictions
            self.activity_history[person_id].append(activity)
        
        if len(self.activity_history[person_id]) == 0:
            return "unknown"
        
        # Get most common activity in recent history
        activity_counts = {}
        for act in self.activity_history[person_id]:
            activity_counts[act] = activity_counts.get(act, 0) + 1
        
        return max(activity_counts, key=activity_counts.get)
    
    def draw_results(self, frame, tracked_persons):
        """Draw bounding boxes and activity labels"""
        for person in tracked_persons:
            person_id = person['id']
            x1, y1, x2, y2 = person['bbox']
            
            # Extract person crop (with padding)
            h, w = frame.shape[:2]
            x1_crop = max(0, x1-10)
            y1_crop = max(0, y1-10)
            x2_crop = min(w, x2+10)
            y2_crop = min(h, y2+10)
            
            person_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
            
            if person_crop.size > 0 and person_crop.shape[0] > 30 and person_crop.shape[1] > 30:
                # Classify activity
                activity, confidence = self.classify_activity(person_crop)
                
                # Smooth the activity prediction
                smoothed_activity = self.smooth_activity(person_id, activity, confidence)
                
                # Get color for activity
                color = self.activity_colors.get(smoothed_activity, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw labels
                label = f"ID:{person_id} {smoothed_activity} ({confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background for text
                cv2.rectangle(frame, (x1, y1-30), (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 0, 0), 2)
        
        return frame
    
    def run(self):
        """Main loop to run the activity tracking system"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting Person Activity Tracking System with Transformers...")
        print("Activities detected: sitting, standing, walking, sleeping, talking, running, lying, raising hand, eating, reading")
        print("Press 'q' to quit")
        
        fps_counter = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Detect persons
            persons = self.detect_persons(frame)
            
            # Track persons
            tracked_persons = self.simple_tracker(persons)
            
            # Draw results
            frame = self.draw_results(frame, tracked_persons)
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                start_time = end_time
                print(f"FPS: {fps:.2f}")
            
            # Draw info on frame
            cv2.putText(frame, f"Persons: {len(tracked_persons)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            model_info = "CLIP" if self.use_classifier else "BLIP"
            cv2.putText(frame, f"Model: {model_info}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Person Activity Tracking - Transformers', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run the activity tracker"""
    try:
        # Check if all required modules are available
        if not TRANSFORMERS_AVAILABLE:
            print("Error: Transformers module is not installed.")
            print("Please install Transformers using:")
            print("pip install transformers")
            return
            
        if not YOLO_AVAILABLE:
            print("Error: YOLOv8 module is not installed.")
            print("Please install it using: pip install ultralytics")
            return
        
        # Initialize the tracker
        tracker = PersonActivityTrackerTransformers()
        
        # Run the tracking system
        tracker.run()
        
    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        print("\nInstallation Instructions:")
        print("1. Install PyTorch:")
        print("   pip install torch torchvision torchaudio")
        print("2. Install Transformers:")
        print("   pip install transformers")
        print("3. Install YOLOv8:")
        print("   pip install ultralytics")
        print("4. Install other dependencies:")
        print("   pip install opencv-python pillow numpy")

if __name__ == "__main__":
    main()