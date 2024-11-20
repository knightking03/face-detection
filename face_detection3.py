import cv2
import argparse

def detect_faces_with_features(image_path):
    # Load the pre-trained Haar Cascade classifiers for face, eyes, nose, and mouth detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
    mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from path {image_path}")
        return

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    valid_faces = []
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for the face
        face_roi = gray_image[y:y+h, x:x+w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
        
        # Detect nose within the face ROI
        nose = nose_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
        
        # Detect mouth within the face ROI (considering the lower half of the face)
        mouth_roi = face_roi[h//2:, :]
        mouth = mouth_cascade.detectMultiScale(mouth_roi, scaleFactor=1.1, minNeighbors=15, minSize=(20, 20))

        # Verify if the detected face has at least two eyes, one nose, and one mouth
        if len(eyes) >= 1 and len(nose) >= 1 and len(mouth) >= 1:
            valid_faces.append((x, y, w, h))

    # Print the number of valid faces detected
    print(f"Number of valid faces detected: {len(valid_faces)}")

    # Draw rectangles around the valid faces
    for (x, y, w, h) in valid_faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
      
        # Display the output image with detected faces
        cv2.imshow('Valid Faces detected', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



# Create the parser
parser = argparse.ArgumentParser(description="enter name of image")

# Add arguments
parser.add_argument('name', type=str, help="Image Name")

# Parse the arguments
args = parser.parse_args()


# Example usage
image_path = args.name
detect_faces_with_features(image_path)
