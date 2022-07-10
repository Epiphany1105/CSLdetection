import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def mediapipe_draw(_method, _frame):
    _frame.flags.writeable = False
    _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
    results = _method.process(_frame)

    _frame.flags.writeable = True
    _frame = cv2.cvtColor(_frame, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        _frame,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing_styles.get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        _frame,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style())
    mp_drawing.draw_landmarks(
        _frame,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style())

    return _frame
