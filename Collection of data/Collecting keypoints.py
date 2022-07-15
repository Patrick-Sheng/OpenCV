import numpy as np
import mediapipe as mp
import cv2
import os

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def mediapipe_detection(image, holistic):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))


def extract_keypoints(results):
    # Extracting landmarks from pose coordiantes, if none is found then create an empty array to prevent error. Note: 132=33*4.
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    return pose

def setting_folders(data_path, actions, data_number):
    for action in actions:
        for sequence in range(data_number):
            try:
                os.makedirs(os.path.join(data_path, action, str(sequence)))
            except:
                pass

## Main Code
data_path = os.path.join('.dataCollection')
# actions = np.array(['cSitting', 'cStanding'])
actions = np.array(['cStanding', 'cSitting'])
data_number = 30
data_length = 10

setting_folders(data_path, actions, data_number)

## Camera settings
wCam, hCam = 1200, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # while cap.isOpened():
    for action in actions:
        for sequence in range(data_number):
            for no_frame in range(data_length):

                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                result_test = extract_keypoints(results)

                # Starting from frame 0
                if no_frame == 0:
                    cv2.putText(image, 'Starting capture', (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3,
                                cv2.LINE_AA)
                    cv2.imshow('Enhanced holistic detection', image)
                    cv2.waitKey(1000)
                    cv2.putText(image, 'Capturing frames for {} Video Number {} Frame Number {}'.format(action, sequence,
                                no_frame), (50, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Enhanced holistic detection', image)
                    cv2.waitKey(200)
                else:
                    cv2.putText(image, 'Capturing frames for {} Video Number {} Frame Number {}'.format(action, sequence,
                                no_frame), (50, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Enhanced holistic detection', image)
                    cv2.waitKey(200)

                cv2.imshow('Enhanced holistic detection', image)

                result_test = extract_keypoints(results)
                file_path = os.path.join(data_path, action, str(sequence), str(no_frame))
                np.save(file_path, result_test)

                # stop machine running when key 'z' is pressed
                if cv2.waitKey(10) & 0xFF == ord('z'):
                    break

    cap.release()
    cv2.destroyAllWindows()
