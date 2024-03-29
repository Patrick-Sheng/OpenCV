import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose

pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils


def detectPose(image_pose, pose, draw=False, display=False):
    original_image = image_pose.copy()

    image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)

    resultant = pose.process(image_in_RGB)

    if resultant.pose_landmarks and draw:
        mp_drawing.draw_landmarks(image=original_image, landmark_list=resultant.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                               thickness=3, circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(49, 125, 237),
                                                                                 thickness=2, circle_radius=2))

    if display:
        plt.figure(figsize=[22, 22])
        plt.subplot(121);
        plt.imshow(image_pose[:, :, ::-1]);
        plt.title("Input Image");
        plt.axis('off');
        plt.subplot(122);
        plt.imshow(original_image[:, :, ::-1]);
        plt.title("Pose detected Image");
        plt.axis('off');

    else:
        return original_image, results

image_path = 'Image.png'
output = cv2.imread(image_path)
detectPose(output, pose_image, draw=True, display=True)

cv2.imshow('Image', output)

cv2.waitKey(0)
cv2.destroyAllWindows()