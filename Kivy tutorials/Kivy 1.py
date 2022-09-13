import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

class CameraCaptureApp(App):

    def build(self):
        self.image = Image(size_hint=(1, .8))
        self.button = Button(text="Click here", pos_hint={'center_x':.5, 'center_y':.5}, size_hint=(None, None))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.image)
        layout.add_widget(self.button)

        # cv2.namedWindow("Kivy OpenCV Feed")
        Clock.schedule_interval(self.update, 1.0/30.0)
        self.capture = cv2.VideoCapture(0)

        return layout

    def update(self, *args):
        ret, frame = self.capture.read()
        # cv2.imshow("Kivy OpenCV Feed", frame)

        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture


if __name__ == '__main__':
    CameraCaptureApp().run()
    cv2.destroyAllWindows()