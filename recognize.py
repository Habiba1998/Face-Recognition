import face_recognition
import cv2
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input


def get_encodings(images):
    encodes = []
    for img in images:
        #Get the 128-dimension face encoding , it returns a list of all faces in the image but as we have only
        #one face we use [0]
        encoded_face = face_recognition.face_encodings(img)
        if (len(encoded_face) == 1):
            encodes.append(encoded_face[0])
    return encodes



def recognize_face(captured_image, faces):
    encoded_face_train = get_encodings(faces)
    model = tf.keras.models.load_model("my_model3")
    # Get face location --> it returns the location as a tuple(top, rigth, bottom, left)
    faces_location = face_recognition.face_locations(captured_image)
    # Get a list of the encodings
    encoded_faces = face_recognition.face_encodings(captured_image, faces_location)

    # Prepare image for live detection model
    # Resize the image shape to the VGG input shape 224x224x3
    # interpolation=cv2.INTER_AREA is used if the size is to be reduced
    liveimg = cv2.resize(captured_image, (224, 224), 3, interpolation=cv2.INTER_AREA)
    liveimg = cv2.cvtColor(liveimg, cv2.COLOR_BGR2RGB)

    # Is required to be done before preprocess_input step to convert 224x224x3 to 1x224x224x3
    np_final = np.expand_dims(liveimg, axis=0)
    # a vgg16 built-in function that process the input before being fed to the vgg model
    processedimage = preprocess_input(np_final)


    # Apply the image to the liveness model
    prediction = model.predict([processedimage])

    # Get the index of the maximum output "softmax layer is used at the output"
    # fake :0 real:1
    max = np.argmax(prediction, axis=-1)

    if max == 0:
        return {'status' : 0, 'identity':'None'}

    max_area = 0
    index = None
    # We expect only one face but we can use a for loop two
    for encoded_face, faceloc in zip(encoded_faces, faces_location):
        # Get a list of true and false which indicate whether a match occur between the two encodings or not
        compare_list = face_recognition.compare_faces(encoded_face_train, encoded_face)
        # Get a list of euclidean distance between the two encodings
        faceDist = face_recognition.face_distance(encoded_face_train, encoded_face)

        # Get the index of the minimum distance
        matchIndex = np.argmin(faceDist)
        if faceDist[matchIndex] < 0.5:
            y1, x2, y2, x1 = faceloc
            area = abs((y2 - y1) * (x2 - x1))
            if area > max_area:
                max_area = area
                index = matchIndex

    return {'status': 1, 'identity': index}
