import mediapipe
import numpy

mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles
mp_pose = mediapipe.solutions.pose


class Body:

    def __init__(self, results):

        self._results = results

        self._seen = results.pose_landmarks is not None

        if not self._seen:
            self.landmarks = numpy.zeros((33, 2), numpy.float64)
            return

        self.landmarks = numpy.array(
            [
                (landmark.x, landmark.y)
                for landmark in results.pose_landmarks.landmark
            ]
        )

        # self.angle_rad = 0.0
        # self.gesture = "Unknown"

    def is_seen(self):
        return self._seen

    def __str__(self):

        if not self.is_seen():
            return "No Body Seen."

        return "Body"

    def draw(self, frame: numpy.ndarray):

        drawing_frame = numpy.copy(frame)

        if not self.is_seen():
            return drawing_frame

        # joint_style = mp_drawing.DrawingSpec(
        #     color=(255, 0, 0), thickness=4, circle_radius=3
        # )
        line_style = mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=3)

        mp_drawing.draw_landmarks(
            drawing_frame,
            self._results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            connection_drawing_spec=line_style,
        )

        return drawing_frame
