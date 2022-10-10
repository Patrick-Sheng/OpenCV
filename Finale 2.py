# Import python packages
import cv2
import mediapipe as mp
import numpy as np

from time import sleep

# Set up pose library and drawing util from mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Set detection confidence to 0.5 for precise estimation (Note: Lower value for better precision)
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)


# Convert image from colour to grey-scale
def mediapipe_detection(image, pose):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image)
    return image, result


# Calculate angle from three different coordinates of body parts
def calculate_angle(landmarks, point):

    # Switch statement to calculate angle for different body parts
    # Note: landmark one, two, three for three different body part coordinates
    match point:
        case "Head":
            landmark_one = [landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            landmark_two = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            landmark_three = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        case "Shoulder":
            landmark_one = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            landmark_two = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            landmark_three = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        case "Hip":
            landmark_one = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            landmark_two = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            landmark_three = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        case "Knee":
            landmark_one = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            landmark_two = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            landmark_three = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    # Convert landmarks into numpy arrays. (Note: [0] is x-coordinate and [1] is y-coordinate)
    arr_landmark_one = np.array(landmark_one)
    arr_landmark_two = np.array(landmark_two)
    arr_landmark_three = np.array(landmark_three)

    # Calculate distance between each coordinate
    a = calc_cd_distance(arr_landmark_one, arr_landmark_two)
    b = calc_cd_distance(arr_landmark_two, arr_landmark_three)
    c = calc_cd_distance(arr_landmark_one, arr_landmark_three)

    # Calculate angle from distance values, using Cosine rule.
    rad_angle = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
    # Convert angle from radians to degrees
    deg_angle = (180 / np.pi) * rad_angle

    return deg_angle


# Calculate distance between two coordinates using distance formula
def calc_cd_distance(arr_one, arr_two):
    # Note: [0] is x-coordinate and [1] is y-coordinate
    dist = ((arr_one[0] - arr_two[0]) ** 2 + (arr_one[1] - arr_two[1]) ** 2) ** 0.5
    return dist


# Determine posture from angles, good body postures will output angle values at a certain range
def determine_posture(head_angle, shoulder_angle, hip_angle, knee_angle, org_image):
    # Ouputing Good/Bad results for different body parts, for user to adjust to.
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


# Main class
# Detecting pose through camera input
def detectPose(image, org_image, pose, draw=False, display=False):

    # Convert image from colour to grey-scale
    image, result = mediapipe_detection(image, pose)

    # Try to collect landmarks from pose detection, passing landmarks if they can't be seen (i.e. Only half of the body is observed)
    try:
        landmarks = result.pose_landmarks.landmark
    except:
        pass

    # Collecting angle for each body mark, using landmarks (Require processing)
    head_angle = calculate_angle(landmarks, "Head")
    shoulder_angle = calculate_angle(landmarks, "Shoulder")
    hip_angle = calculate_angle(landmarks, "Hip")
    knee_angle = calculate_angle(landmarks, "Knee")

    sleep(1)

    # Determine posture from angles, good body postures will output angle values at a certain range
    determine_posture(head_angle, shoulder_angle, hip_angle, knee_angle, org_image)

    # Show lines of points and joints on user body for display purposes
    if result.pose_landmarks and draw:
        mp_drawing.draw_landmarks(org_image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=3))

    cv2.putText(org_image, 'Press any key to exit', (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # Show processed image with text on window popup
    cv2.imshow('Image', org_image)


# Main code

# Choose camera output device, (0) for first device
cap = cv2.VideoCapture(0)
# Set up dimensions for camera output device
wCam, hCam = 1280, 720
cap.set(3, wCam)
cap.set(4, hCam)

while cap.isOpened():

    # Collecting images and videos through camera capture
    ret, frame = cap.read()
    cv2.putText(frame, 'Press "z" key to capture image', (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # Show image with text on window popup
    cv2.imshow('Pose estimation', frame)

    # Collect data after user presses "z" key
    if cv2.waitKey(10) & 0xFF == ord('z'):

        # Wait for 3 seconds for user to prepare
        sleep(3)

        # Make copy of images and videos through camera capture, and write image on image file
        ret_copy, frame_copy = cap.read()
        cv2.imwrite("Real_image.png", frame_copy)
        # Close windows
        cv2.destroyWindow('Pose estimation')
        cap.release()

# Open image file and process with pose detection
file_name = 'Real_image.png'
output = cv2.imread(file_name)
detectPose(output, output, pose_image, draw=True, display=True)

# Close window if user presses any key (after results)
cv2.waitKey(0)
cv2.destroyAllWindows()