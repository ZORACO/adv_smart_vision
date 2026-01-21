import os
from imutils import paths
import face_recognition
import pickle
import cv2

print("[INFO] Starting face encoding process...")

imagePaths = list(paths.list_images("dataset"))

if len(imagePaths) == 0:
    print("[ERROR] No images found in 'dataset' folder!")
    print("Please run capturing_face.py first to collect training data.")
    exit(1)

knownEncodings = []
knownNames = []

# Track statistics
total_images = len(imagePaths)
successful = 0
failed = 0
multiple_faces = 0

print(f"[INFO] Found {total_images} images to process")
print("-" * 60)

for (i, imagePath) in enumerate(imagePaths):
    print(f"[INFO] Processing image {i + 1}/{total_images}: {os.path.basename(imagePath)}")
    
    name = imagePath.split(os.path.sep)[-2]
    
    # Add error handling for corrupted images
    try:
        image = cv2.imread(imagePath)
        if image is None:
            print(f"  ✗ Failed to load image, skipping...")
            failed += 1
            continue
            
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use CNN model for better accuracy during training
        # Training is one-time, so we can afford to be slower but more accurate
        boxes = face_recognition.face_locations(rgb, model="cnn")
        
        # Validate face detection quality
        if len(boxes) == 0:
            print(f"  ✗ No face detected, skipping...")
            failed += 1
            continue
        elif len(boxes) > 1:
            print(f"  ⚠ Multiple faces detected ({len(boxes)}), using largest...")
            # Use the largest face (closest to camera)
            boxes = [max(boxes, key=lambda box: (box[2]-box[0]) * (box[1]-box[3]))]
            multiple_faces += 1
        
        # Use 'large' model for encoding - better accuracy
        encodings = face_recognition.face_encodings(rgb, boxes, model='large')
        
        if len(encodings) == 0:
            print(f"  ✗ Could not encode face, skipping...")
            failed += 1
            continue
        
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
            successful += 1
            print(f"  ✓ Successfully encoded")
    
    except Exception as e:
        print(f"  ✗ Error processing image: {str(e)}")
        failed += 1
        continue

print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)
print(f"Total images processed: {total_images}")
print(f"✓ Successfully encoded: {successful}")
print(f"✗ Failed to process: {failed}")
print(f"⚠ Multiple faces found: {multiple_faces}")

# Validate minimum encodings per person
if successful == 0:
    print("\n[ERROR] No faces were successfully encoded!")
    print("Please check your dataset and try again.")
    exit(1)

# Count encodings per person
from collections import Counter
person_counts = Counter(knownNames)

print("\nEncodings per person:")
for person, count in person_counts.items():
    status = "✓" if count >= 20 else "⚠"
    print(f"  {status} {person}: {count} encodings")
    if count < 20:
        print(f"     Warning: Less than 20 encodings may reduce accuracy")

print("\n" + "=" * 60)
print("[INFO] Serializing encodings to 'encodings.pickle'...")

data = {"encodings": knownEncodings, "names": knownNames}

try:
    with open("encodings.pickle", "wb") as f:
        f.write(pickle.dumps(data))
    print("[SUCCESS] Training complete!")
    print(f"[INFO] Saved {successful} face encodings to 'encodings.pickle'")
    print("\nYou can now run face_recognition_code.py for detection")
except Exception as e:
    print(f"[ERROR] Failed to save encodings: {str(e)}")
    exit(1)