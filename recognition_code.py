import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle

# Load encodings
print("[INFO] Loading encodings...")
try:
    with open("encodings.pickle", "rb") as f:
        data = pickle.loads(f.read())
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]
    print(f"[INFO] Loaded {len(known_face_encodings)} face encodings")
except FileNotFoundError:
    print("[ERROR] encodings.pickle not found!")
    print("Please run train_model.py first")
    exit(1)

# Initialize camera
picam2 = Picamera2()
# Reduce resolution for better FPS
picam2.configure(picam2.create_preview_configuration(
    main={"format": 'XRGB8888', "size": (640, 480)}))  # Was 1920x1080!
picam2.start()

# Optimize scaling - bigger scale = faster but less accurate
# 4 is good balance for 640x480
cv_scaler = 4  # Process at 160x120 (very fast)

# Process every N frames, not every frame
PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame
frame_counter = 0

# Cache for face data
face_locations = []
face_encodings = []
face_names = []

# Performance tracking
frame_count = 0
start_time = time.time()
fps = 0
process_time = 0

# Confidence threshold - ignore low confidence matches
CONFIDENCE_THRESHOLD = 0.6  # 60% confidence minimum

def process_frame(frame):
    global face_locations, face_encodings, face_names, process_time
    
    process_start = time.time()
    
    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Use HOG (faster) instead of CNN for real-time
    face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
    
    # Use 'small' model for faster encoding
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame, face_locations, model='small')
    
    face_names = []
    face_confidences = []  # Track confidence levels
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        confidence = 0
        
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding)
        
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            
            # Calculate confidence percentage
            confidence = 1 - face_distances[best_match_index]
            
            # Only accept if confidence is high enough
            if matches[best_match_index] and confidence > CONFIDENCE_THRESHOLD:
                name = known_face_names[best_match_index]
            else:
                name = "Unknown"
        
        face_names.append(name)
        face_confidences.append(confidence)
    
    process_time = (time.time() - process_start) * 1000  # Convert to ms
    return face_confidences

def draw_results(frame, confidences):
    for idx, ((top, right, bottom, left), name) in enumerate(zip(face_locations, face_names)):
        # Scale back to original size
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        
        # Color code by confidence
        confidence = confidences[idx] if idx < len(confidences) else 0
        if name == "Unknown":
            color = (0, 0, 255)  # Red for unknown
        elif confidence > 0.8:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence > 0.6:
            color = (0, 165, 255)  # Orange for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence
        
        # Draw rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw label background
        cv2.rectangle(frame, (left, top - 35), (right, top), color, cv2.FILLED)
        
        # Show confidence percentage
        label = f"{name} ({confidence*100:.1f}%)"
        cv2.putText(frame, label, (left + 6, top - 6), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

# Add performance info display
def draw_performance_info(frame, fps, process_time):
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Processing time
    cv2.putText(frame, f"Process: {process_time:.1f}ms", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Detection status
    status = f"Faces: {len(face_locations)}"
    cv2.putText(frame, status, (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

print("\n" + "=" * 60)
print("FACE RECOGNITION ACTIVE")
print("=" * 60)
print(f"Resolution: 640x480")
print(f"Processing scale: 1/{cv_scaler} ({640//cv_scaler}x{480//cv_scaler})")
print(f"Process every: {PROCESS_EVERY_N_FRAMES} frames")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD*100}%")
print("Press 'q' to quit")
print("=" * 60 + "\n")

# Main loop
while True:
    frame = picam2.capture_array()
    
    # Only process every Nth frame
    if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
        confidences = process_frame(frame)
    frame_counter += 1
    
    # Always draw (uses cached data if not processing)
    display_frame = draw_results(frame, confidences if 'confidences' in locals() else [])
    display_frame = draw_performance_info(display_frame, calculate_fps(), process_time)
    
    cv2.imshow('Face Recognition', display_frame)
    
    if cv2.waitKey(1) == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
print("\n[INFO] Face recognition stopped")