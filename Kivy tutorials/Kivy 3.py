import cv2
import mediapipe as mp
import numpy as np

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

class CameraCaptureApp(App):

    def build(self):
        self.image = Image(size_hint=(1, .8))
        self.capture_button = Button(text="Take picture", pos_hint={'center_x':.7, 'center_y':.7}, size_hint=(None, None))
        self.capture_button.bind(on_press=self.take_picture)

        self.check_button = Button(text="Check posture", pos_hint={'center_x':.2, 'center_y':.2}, size_hint=(None, None))
        self.check_button.bind(on_press=self.check_posture)

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.image)
        layout.add_widget(self.capture_button)
        layout.add_widget(self.check_button)

        # cv2.namedWindow("Kivy OpenCV Feed")
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/30.0)

        return layout

    def update(self, *args):
        ret, frame = self.capture.read()

        self.image_frame = frame

        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    def take_picture(self, *args):
        image_name = "Image1.png"
        cv2.imwrite(image_name, self.image_frame)

    def check_posture(self, *args):
        file_path = "Image.png"

        image = cv2.imread(file_path, 0)
        # results = mp_pose.Pose(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        cv2.imshow('Image', image)


if __name__ == '__main__':
    CameraCaptureApp().run()
    cv2.destroyAllWindows()