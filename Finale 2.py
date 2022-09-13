import cv2
import mediapipe as mp
import numpy as np

from time import sleep

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)


def mediapipe_detection(image, pose):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image)
    return image, result


def calculate_angle(landmarks, point):
    if point == "Head":
        landmark_one = [landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
        landmark_two = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        landmark_three = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    elif point == "Shoulder":
        landmark_one = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        landmark_two = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        landmark_three = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    elif point == "Hip":
        landmark_one = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        landmark_two = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        landmark_three = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    elif point == "Knee":
        landmark_one = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        landmark_two = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        landmark_three = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    arr_landmark_one = np.array(landmark_one)
    arr_landmark_two = np.array(landmark_two)
    arr_landmark_three = np.array(landmark_three)

    a = calc_cd_distance(arr_landmark_one, arr_landmark_two)
    b = calc_cd_distance(arr_landmark_two, arr_landmark_three)
    c = calc_cd_distance(arr_landmark_one, arr_landmark_three)

    rad_angle = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
    deg_angle = (180 / np.pi) * rad_angle

    return deg_angle


def calc_cd_distance(arr_one, arr_two):
    dist = ((arr_one[0] - arr_two[0]) ** 2 + (arr_one[1] - arr_two[1]) ** 2) ** 0.5
    return dist

def determine_posture(head_angle, shoulder_angle, hip_angle, knee_angle, org_image):
    if 140 < head_angle < 180:
        # image, "Window name", "org, co-ordinate", font, fontScale, colour, thickness, cv2.LINE_AA
        cv2.putText(org_image, 'Head: Good angle', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(org_image, 'Head: Bad angle', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if 90 < shoulder_angle < 135:
        # image, "Window name", "org, co-ordinate", font, fontScale, colour, thickness, cv2.LINE_AA
        cv2.putText(org_image, 'Shoulder: Good angle', (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(org_image, 'Shoulder: Bad angle', (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if 90 < hip_angle < 125:
        # image, "Window name", "org, co-ordinate", font, fontScale, colour, thickness, cv2.LINE_AA
        cv2.putText(org_image, 'Hip: Good angle', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(org_image, 'Hip: Bad angle', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if 90 < knee_angle < 125:
        # image, "Window name", "org, co-ordinate", font, fontScale, colour, thickness, cv2.LINE_AA
        cv2.putText(org_image, 'Knee: Good angle', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(org_image, 'Knee: Bad angle', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


def detectPose(image, org_image, pose, draw=False, display=False):

    image, result = mediapipe_detection(image, pose)

    try:
        landmarks = result.pose_landmarks.landmark
    except:
        pass

    head_angle = calculate_angle(landmarks, "Head")
    shoulder_angle = calculate_angle(landmarks, "Shoulder")
    hip_angle = calculate_angle(landmarks, "Hip")
    knee_angle = calculate_angle(landmarks, "Knee")

    sleep(1)
    determine_posture(head_angle, shoulder_angle, hip_angle, knee_angle, org_image)

    # show lines and points for display purpose
    if result.pose_landmarks and draw:
        mp_drawing.draw_landmarks(org_image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=3))

    cv2.putText(org_image, 'Press "z" key again to exit', (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Image', org_image)

wCam, hCam = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

while cap.isOpened():

    ret, frame = cap.read()
    cv2.putText(frame, 'Press "z" key to capture image', (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Pose estimation', frame)

    if cv2.waitKey(10) & 0xFF == ord('z'):
        # cv2.destroyWindow('Pose estimation')

        # for i in range(10, 50, 1):
        #     ret_s, frame_s = cap.read()
        #     cv2.putText(frame_s, 'Press "z" key', (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,cv2.LINE_AA)
        #     cv2.imshow('Pose', frame_s)
        #     sleep(0.1)
        # break

        sleep(3)

        ret_copy, frame_copy = cap.read()
        cv2.imwrite("Real_image.png", frame_copy)
        cv2.destroyWindow('Pose estimation')
        cap.release()


file_name = 'Real_image.png'
output = cv2.imread(file_name)
detectPose(output, output, pose_image, draw=True, display=True)

cv2.waitKey(0)
cv2.destroyAllWindows()