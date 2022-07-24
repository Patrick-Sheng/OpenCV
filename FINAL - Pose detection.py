# import python packages
import numpy as np
import mediapipe as mp
import cv2

# setting up mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# converting image to black and white to save resources while processing
def mediapipe_detection(image, holistic):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    # processing image with mediapipe holistic
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# calcating angle for different joints
def calculate_angle(landmarks, point):
    # collecting x-coordinates and y-coordinates
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

    # converting landmarks to numpy arrays
    arr_landmark_one = np.array(landmark_one)
    arr_landmark_two = np.array(landmark_two)
    arr_landmark_three = np.array(landmark_three)

    # use function to calculate distance between points
    a = calc_cd_distance(arr_landmark_one, arr_landmark_two)
    b = calc_cd_distance(arr_landmark_two, arr_landmark_three)
    c = calc_cd_distance(arr_landmark_one, arr_landmark_three)

    # calculation to find angle
    rad_angle = np.arccos((a**2 + b**2 - c**2)/(2*a*b))
    deg_angle = (180/np.pi) * rad_angle

    return deg_angle

# function to find distance between two points
def calc_cd_distance(arr_one, arr_two):
    dist = ((arr_one[0] - arr_two[0])**2 + (arr_one[1] - arr_two[1])**2)**0.5
    return dist


def determine_posture(head_angle, shoulder_angle, hip_angle, knee_angle):
    # determine whether if the user is sitting in the right posture (angles formed from body parts will fit in a specific range)
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

# setting webcam properties
wCam, hCam = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# tracking image with lower confidence level (slightly under-fitting)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # opening webcam to capture feed
        ret, frame = cap.read()
        # preprocess image for calculation purposes
        image, results = mediapipe_detection(frame, holistic)

        # display landmarks with connections on screen for testing/user viewing purposes
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=3))

        # Use try and except to prevent machine breaking when there's coordinate that cannot be captured/seen
        try:
            landmarks = results.pose_landmarks.landmark
        except:
            pass

        # calculating angles for different body parts
        head_angle = calculate_angle(landmarks, "Head")
        shoulder_angle = calculate_angle(landmarks, "Shoulder")
        hip_angle = calculate_angle(landmarks, "Hip")
        knee_angle = calculate_angle(landmarks, "Knee")
        # print(shoulder_angle)

        # determine posture correction with body parts (no return)
        determine_posture(head_angle, shoulder_angle, hip_angle, knee_angle)

        # show webcam feed with processed calculation/connections on screen
        cv2.imshow('Pose estimation', image)

        # stop when 'z' key is pressed on keyboard
        if cv2.waitKey(10) & 0xFF == ord('z'):
            break
            
    # close windows
    cap.release()
    cv2.destroyAllWindows()
