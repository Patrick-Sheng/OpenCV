import numpy as np
import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def mediapipe_detection(image, holistic):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def calculate_angle(landmarks, point):
    if point == "Head":
        landmark_one = [landmarks[mp_holistic.PoseLandmark.NOSE].x, landmarks[mp_holistic.PoseLandmark.NOSE.value].y]
        landmark_two = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
        landmark_three = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y]
    elif point == "Shoulder":
        landmark_one = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
        landmark_two = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
        landmark_three = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]
    elif point == "Hip":
        landmark_one = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
        landmark_two = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y]
        landmark_three = [landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].y]
    elif point == "Knee":
        landmark_one = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP].x, landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y]
        landmark_two = [landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].y]
        landmark_three = [landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].y]

    arr_landmark_one = np.array(landmark_one)
    arr_landmark_two = np.array(landmark_two)
    arr_landmark_three = np.array(landmark_three)

    a = calc_cd_distance(arr_landmark_one, arr_landmark_two)
    b = calc_cd_distance(arr_landmark_two, arr_landmark_three)
    c = calc_cd_distance(arr_landmark_one, arr_landmark_three)

    rad_angle = np.arccos((a**2 + b**2 - c**2)/(2*a*b))
    deg_angle = (180/np.pi) * rad_angle

    return deg_angle


def calc_cd_distance(arr_one, arr_two):
    dist = ((arr_one[0] - arr_two[0])**2 + (arr_one[1] - arr_two[1])**2)**0.5
    return dist


def determine_posture(head_angle, shoulder_angle, hip_angle, knee_angle):
    if 140 < head_angle < 180:
        # image, "Window name", "org, co-ordinate", font, fontScale, colour, thickness, cv2.LINE_AA
        cv2.putText(image, 'Head: Good angle', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, 'Head: Bad angle', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if 90 < shoulder_angle < 135:
        # image, "Window name", "org, co-ordinate", font, fontScale, colour, thickness, cv2.LINE_AA
        cv2.putText(image, 'Shoulder: Good angle', (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, 'Shoulder: Bad angle', (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if 90 < hip_angle < 125:
        # image, "Window name", "org, co-ordinate", font, fontScale, colour, thickness, cv2.LINE_AA
        cv2.putText(image, 'Hip: Good angle', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, 'Hip: Bad angle', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if 90 < knee_angle < 125:
        # image, "Window name", "org, co-ordinate", font, fontScale, colour, thickness, cv2.LINE_AA
        cv2.putText(image, 'Knee: Good angle', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, 'Knee: Bad angle', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


wCam, hCam = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=3))

        # Use try and except to prevent machine breaking when there's coordinate that cannot be captured/seen
        try:
            landmarks = results.pose_landmarks.landmark
        except:
            pass

        head_angle = calculate_angle(landmarks, "Head")
        shoulder_angle = calculate_angle(landmarks, "Shoulder")
        hip_angle = calculate_angle(landmarks, "Hip")
        knee_angle = calculate_angle(landmarks, "Knee")
        # print(shoulder_angle)

        determine_posture(head_angle, shoulder_angle, hip_angle, knee_angle)
        cv2.imshow('Pose estimation', image)

        if cv2.waitKey(10) & 0xFF == ord('z'):
            break

    cap.release()
    cv2.destroyAllWindows()
