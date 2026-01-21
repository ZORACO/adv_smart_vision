import cv2
import os
from datetime import datetime
from picamera2 import Picamera2
import time
import face_recognition  # ADD THIS for face validation

PERSON_NAME = "Xyz"
MIN_PHOTOS = 30  # Minimum photos needed for good training
TARGET_PHOTOS = 50  # Target number of photos

def create_folder(name):
    dataset_folder = "dataset"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    person_folder = os.path.join(dataset_folder, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder

def capture_photos(name):
    folder = create_folder(name)
    
    picam2 = Picamera2()
    # CHANGE 1: Lower resolution for faster capture but still good quality
    picam2.configure(picam2.create_preview_configuration(
        main={"format": 'XRGB8888', "size": (640, 480)}))
    picam2.start()
    
    time.sleep(2)
    
    photo_count = 0
    rejected_count = 0  # Track rejected photos
    
    print(f"Taking photos for {name}.")
    print(f"Target: {TARGET_PHOTOS} photos (minimum: {MIN_PHOTOS})")
    print("Press SPACE to capture, 'q' to quit.")
    print("-" * 50)
    
    while photo_count < TARGET_PHOTOS:
        frame = picam2.capture_array()
        
        # CHANGE 2: Convert to RGB for face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # CHANGE 3: Detect faces in real-time
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # CHANGE 4: Draw rectangles around detected faces
        display_frame = frame.copy()
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # CHANGE 5: Show status on screen
        status_color = (0, 255, 0) if len(face_locations) == 1 else (0, 0, 255)
        status_text = f"Faces detected: {len(face_locations)} | Photos: {photo_count}/{TARGET_PHOTOS}"
        cv2.putText(display_frame, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        if len(face_locations) == 1:
            cv2.putText(display_frame, "READY - Press SPACE", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif len(face_locations) == 0:
            cv2.putText(display_frame, "NO FACE DETECTED", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "MULTIPLE FACES - Move others away", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Capture', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            # CHANGE 6: Only save if exactly ONE face is detected
            if len(face_locations) == 1:
                photo_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{timestamp}.jpg"
                filepath = os.path.join(folder, filename)
                cv2.imwrite(filepath, frame)
                print(f"✓ Photo {photo_count}/{TARGET_PHOTOS} saved: {filename}")
            else:
                rejected_count += 1
                if len(face_locations) == 0:
                    print(f"✗ Rejected: No face detected (Total rejected: {rejected_count})")
                else:
                    print(f"✗ Rejected: {len(face_locations)} faces detected, need exactly 1 (Total rejected: {rejected_count})")
        
        elif key == ord('q'):
            if photo_count >= MIN_PHOTOS:
                print(f"\nEarly exit: {photo_count} photos captured (minimum reached)")
                break
            else:
                print(f"\nWarning: Only {photo_count} photos. Need at least {MIN_PHOTOS} for good accuracy.")
                confirm = input("Exit anyway? (y/n): ")
                if confirm.lower() == 'y':
                    break
    
    cv2.destroyAllWindows()
    picam2.stop()
    
    print("\n" + "=" * 50)
    print(f"Capture Complete!")
    print(f"✓ Photos saved: {photo_count}")
    print(f"✗ Photos rejected: {rejected_count}")
    print(f"Location: {folder}")
    print("=" * 50)
    
    # CHANGE 7: Give tips for better training
    if photo_count < MIN_PHOTOS:
        print("\n⚠ WARNING: Low photo count may result in poor recognition!")
    
    print("\nTips for next person:")
    print("- Capture from different angles (front, left, right)")
    print("- Try different expressions (neutral, smiling)")
    print("- Vary lighting conditions slightly")
    print("- Keep distance consistent (2-4 feet from camera)")

if __name__ == "__main__":
    capture_photos(PERSON_NAME)