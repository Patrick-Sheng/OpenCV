import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def detectPose(image_pose, pose, draw=False):
    org_image = image_pose.copy()
    # rgb_image = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)
    # resultant = pose.process(rgb_image)
    resultant = pose.process(org_image)

    if resultant.pose_landmarks and draw:
        # mp_drawing.draw_landmarks(image=org_image, landmark_list=resultant.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS,
        #                           landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=3),
        #                           connection_drawing_spec=mp_drawing.DrawingSpec(color=(49, 125, 237), thickness=2, circle_radius=2))
        mp_drawing.draw_landmarks(org_image, resultant.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=3))


image = cv2.imread('Image.png', 0)
detectPose(image, pose_image, draw=True)

# cv2.imshow('Image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
