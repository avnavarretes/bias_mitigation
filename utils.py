import cv2
import math
import sys
from deepface import DeepFace
import face_recognition

folder_drive = "/content/drive/MyDrive/MasterDissertation/"

# Defined the model files
FACE_PROTO = f"{folder_drive}cv_models/weights/opencv_face_detector.pbtxt"
FACE_MODEL = f"{folder_drive}cv_models/weights/opencv_face_detector_uint8.pb"

AGE_PROTO = f"{folder_drive}cv_models/weights/age_deploy.prototxt"
AGE_MODEL = f"{folder_drive}cv_models/weights/age_net.caffemodel"

GENDER_PROTO = f"{folder_drive}cv_models/weights/gender_deploy.prototxt"
GENDER_MODEL = f"{folder_drive}cv_models/weights/gender_net.caffemodel"

# Load network
FACE_NET = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
AGE_NET = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
GENDER_NET = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_LIST = ["Male", "Female"]

box_padding = 20

def get_face_box (net, frame, conf_threshold = 0.7):

    frame_copy = frame.copy()
    frame_height = frame_copy.shape[0]
    frame_width = frame_copy.shape[1]
    blob = cv2.dnn.blobFromImage(frame_copy, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)
    return frame_copy, boxes

def age_gender_detector (input_path):
    image = cv2.imread(input_path)
    resized_image = cv2.resize(image, (640, 480))
    frame = resized_image.copy()
    frame_face, boxes = get_face_box(FACE_NET, frame)
    for box in boxes:
        face = frame[max(0, box[1] - box_padding):min(box[3] + box_padding, frame.shape[0] - 1), \
            max(0, box[0] - box_padding):min(box[2] + box_padding, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB = False)
        GENDER_NET.setInput(blob)
        gender_predictions = GENDER_NET.forward()
        gender = GENDER_LIST[gender_predictions[0].argmax()]
        print("Gender: {}, conf: {:.3f}".format(gender, gender_predictions[0].max()))
        AGE_NET.setInput(blob)
        age_predictions = AGE_NET.forward()
        age = AGE_LIST[age_predictions[0].argmax()]
        print("Age: {}, conf: {:.3f}".format(age, age_predictions[0].max()))
        label = "{},{}".format(gender, age)
        cv2.putText(frame_face, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    return frame_face, str(gender), str(age)

def detect_ethnicity(image_path):
    # Load the image
    image = face_recognition.load_image_file(image_path)

    # Detect faces in the image
    face_locations = face_recognition.face_locations(image)

    result_faces = []
    # Iterate through face locations and analyze each face
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]

        # Convert the image to the format required by DeepFace
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Analyze the face for ethnicity
        result = DeepFace.analyze(face_image, actions=['race'], enforce_detection=False)
        print("Ethnicity prediction:", result)
        result_faces.append(result[0])

    return result_faces
    
#https://dev.to/ethand91/simple-age-and-gender-detection-using-python-and-opencv-319h
