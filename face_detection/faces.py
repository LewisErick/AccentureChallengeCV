import os
import cv2

def get_faces(img_path):
    # Read the image.
    img_path = os.path.join(os.path.dirname(__file__), img_path)
    image = cv2.imread(img_path)

    # Resize the image
    aspect_ratio = image.shape[0] / image.shape[1]
    new_size = 500
    img = cv2.resize(image, (int(new_size * (1/aspect_ratio)), int(new_size * (1*aspect_ratio))))

    # Preprocess the image to adapt to lighting
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)

    # Get features of image
    haar_cascade_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

    # Get the rectangles of the faces you find in the image
    faces_rects = haar_cascade_face.detectMultiScale(
        img_gray,
        scaleFactor = 1.2,
        minNeighbors = 5)

    # Extract image
    contour_mask = np.zeros(img_gray.shape,np.uint8)
    faces = []
    for (x,y,w,h) in faces_rects:
        faces.append(img[y:y+h, x:x+w])
    return faces

def prepare_face_recognizer(faces):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    labels = []
    training_data = []
    for i, face in enumerate(faces):
        img = cv2.resize(face, (300, 300))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.equalizeHist(img_gray)
        training_data.append(img_gray)
        labels.append(i)
    
    face_recognizer.train(training_data, np.array(labels))
    return face_recognizer

def recognize_face(recognizer, face):
    label = recognizer.predict(face)
    return label
