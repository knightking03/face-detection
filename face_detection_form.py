from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2

app = Flask(__name__)


def detect_faces_with_features_cv2(image_np):
    # Load the pre-trained Haar Cascade classifiers for face, eyes, nose, and mouth detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
    mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))

    valid_faces = []
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for the face
        face_roi = gray_image[y:y+h, x:x+w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(10, 10))
        
        # Detect nose within the face ROI
        nose = nose_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(10, 10))
        
        # Detect mouth within the face ROI (considering the lower half of the face)
        mouth_roi = face_roi[h//2:, :]
        mouth = mouth_cascade.detectMultiScale(mouth_roi, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10))

        # Verify if the detected face has at least two eyes, one nose, and one mouth
        if len(eyes) >= 2 or len(nose) >= 1 or len(mouth) >= 1:
            valid_faces.append((x, y, w, h))

    # Draw rectangles around the valid faces
    for (x, y, w, h) in valid_faces:
        cv2.rectangle(image_np, (x, y), (x + w, y + h), (255, 0, 0), 4)

    # Return the number of valid faces and the image with rectangles drawn
    return len(valid_faces), image_np


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Perform face detection
        image_np = cv2.imread(file_path)
        num_faces, image_with_faces = detect_faces_with_features_cv2(image_np)

        # Save the image with detected faces
        cv2.imwrite(file_path, image_with_faces)

        return render_template('result.html', filename=filename, num_faces=num_faces)
    else:
        return redirect(request.url)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)



