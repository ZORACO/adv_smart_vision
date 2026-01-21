# Face Recognition System using Raspberry Pi

A real-time face recognition system built on **Raspberry Pi** using **Picamera2**, **OpenCV**, and **face_recognition**.  
The system follows a complete pipeline:
1. High-quality face data collection
2. Offline training using deep face encodings
3. Optimized real-time face recognition with confidence scoring

---

##  Features

- Real-time face detection and recognition
- Dataset validation (exactly one face per image)
- Optimized for Raspberry Pi performance
- Confidence-based recognition filtering
- FPS and processing-time monitoring
- Robust error handling and statistics reporting

---

## 🗂 Project Structure

- ├── dataset/
- │ └── person_name/
- │ ├── image_1.jpg
- │ ├── image_2.jpg
- │ └── ...
- ├── capturing_face.py
- ├── train_model.py
- ├── recognition_code.py
- ├── encodings.pickle
- ├── requirements.txt
- └── README.md
