import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
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


def determine_posture(head_angle, shoulder_angle, hip_angle, knee_angle):
    if 140 < head_angle < 180:
        print("Head angle: Good " + str(head_angle))
    else:
        print("Head angle: Bad " + str(head_angle))

    if 90 < shoulder_angle < 135:
        print("Shoulder angle: Good " + str(shoulder_angle))
    else:
        print("Shoulder angle: Bad " + str(shoulder_angle))

    if 90 < hip_angle < 125:
        print("Hip angle: Good " + str(hip_angle))
    else:
        print("Hip angle: Bad " + str(hip_angle))

    if 90 < knee_angle < 125:
        print("Knee angle: Good " + str(knee_angle))
    else:
        print("Knee angle: Bad " + str(knee_angle))


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

    determine_posture(head_angle, shoulder_angle, hip_angle, knee_angle)

    # show lines and points for display purpose
    if result.pose_landmarks and draw:
        mp_drawing.draw_landmarks(org_image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=3))

    cv2.imshow('Image', org_image)

file_name = 'Image1.png'
output = cv2.imread(file_name)
detectPose(output, output, pose_image, draw=True, display=True)

cv2.waitKey(0)
cv2.destroyAllWindows()