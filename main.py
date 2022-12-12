import copy
import cv2
import dataclasses
import mediapipe as mp
import mouse

from numpy import float_
from numpy.typing import NDArray
from typing import Iterable, Literal, NamedTuple, Sequence, TypeAlias


class LandmarkListType(NamedTuple):
    """mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList"""

    landmark: Sequence


class SolutionOutputs(NamedTuple):
    """mediapipe.python.solution_base.SolutionOutputs"""

    pose_landmarks: LandmarkListType
    pose_world_landmarks: LandmarkListType
    right_hand_landmarks: LandmarkListType
    left_hand_landmarks: LandmarkListType


class Coordinates(NamedTuple):
    x: float
    y: float
    z: float


@dataclasses.dataclass
class Finger(Iterable):
    is_up: bool
    pip: Coordinates | None = None
    dip: Coordinates | None = None
    tip: Coordinates | None = None

    def __iter__(self):
        return iter(self.__dict__.values())


@dataclasses.dataclass
class Hand(Iterable):
    thumb: Finger
    index: Finger
    middle: Finger
    ring: Finger
    pinky: Finger
    total_raised: list[int] = dataclasses.field(default_factory=list)

    def __iter__(self):
        return iter(self.__dict__.values())


NumpyNDArray: TypeAlias = NDArray[float_]
Side: TypeAlias = Literal["right", "left"]


class GestureControl:
    LANDMARK_IDS = mp.solutions.holistic.HandLandmark
    HAND_LANDMARK_IDS = [
        [LANDMARK_IDS.THUMB_IP, LANDMARK_IDS.THUMB_TIP],
        [LANDMARK_IDS.INDEX_FINGER_PIP, LANDMARK_IDS.INDEX_FINGER_DIP, LANDMARK_IDS.INDEX_FINGER_TIP],
        [LANDMARK_IDS.MIDDLE_FINGER_PIP, LANDMARK_IDS.MIDDLE_FINGER_DIP, LANDMARK_IDS.MIDDLE_FINGER_TIP],
        [LANDMARK_IDS.RING_FINGER_PIP, LANDMARK_IDS.RING_FINGER_DIP, LANDMARK_IDS.RING_FINGER_TIP],
        [LANDMARK_IDS.PINKY_PIP, LANDMARK_IDS.PINKY_DIP, LANDMARK_IDS.PINKY_TIP]
    ]

    def __init__(self, win_w: int, win_h: int) -> None:
        self.width, self.height = win_w, win_h

        self.right_hand = Hand(
            Finger(is_up=False),
            Finger(is_up=False),
            Finger(is_up=False),
            Finger(is_up=False),
            Finger(is_up=False)
        )
        self.left_hand = Hand(
            Finger(is_up=False),
            Finger(is_up=False),
            Finger(is_up=False),
            Finger(is_up=False),
            Finger(is_up=False)
        )

    def update_mouse(self, image_processing_result: SolutionOutputs) -> None:
        if image_processing_result.right_hand_landmarks:
            self._update_hand("right", image_processing_result.right_hand_landmarks)
        if image_processing_result.left_hand_landmarks:
            self._update_hand('left', image_processing_result.left_hand_landmarks)

        if all((self.right_hand.index.is_up, self.right_hand.middle.is_up)):
            index_tip = self.right_hand.index.tip
            middle_tip = self.right_hand.middle.tip

            mouse.move(
                (index_tip.x + middle_tip.x) / 2,
                (index_tip.y + middle_tip.y) / 2,
                duration=1/24
            )

            if abs(index_tip.x - middle_tip.x) / self.width > 0.07:
                print(abs(index_tip.x - middle_tip.x) / self.width)
                mouse.press()
            else:
                mouse.release()

    def _update_hand(self, side: Side, landmarks: LandmarkListType) -> None:
        hand = self.right_hand if side == "right" else self.left_hand

        for finger, landmark_ids in zip(hand, self.HAND_LANDMARK_IDS):
            if len(landmark_ids) == 3:
                pip = copy.copy(landmarks.landmark[landmark_ids[0]])
                pip.x = self.width * (1 - pip.x)
                pip.y *= self.height

                dip = copy.copy(landmarks.landmark[landmark_ids[1]])
                dip.x = self.width * (1 - dip.x)
                dip.y *= self.height

                tip = copy.copy(landmarks.landmark[landmark_ids[2]])
                tip.x = self.width * (1 - tip.x)
                tip.y *= self.height

                self._update_finger(finger, pip=pip, dip=dip, tip=tip)
            else:
                pip = copy.copy(landmarks.landmark[landmark_ids[0]])
                pip.x *= self.width
                pip.y *= self.height

                tip = copy.copy(landmarks.landmark[landmark_ids[1]])
                tip.x *= self.width
                tip.y *= self.height

                self._update_finger(finger, pip=pip, tip=tip)

    @staticmethod
    def _update_finger(finger: Finger, *, pip: Coordinates, tip: Coordinates,
                       dip: Coordinates = None) -> None:
        finger.pip = pip
        finger.dip = dip
        finger.tip = tip

        if finger.pip and finger.tip:
            finger.is_up = tip.y < pip.y


def _get_test_frame(self) -> NumpyNDArray | None:
    if not self._cap.isOpened():
        raise IOError("Cannot open webcam")

    success = False
    while not success:
        success, frame = self._cap.read()
        return frame


if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    analyzer = GestureControl(1280, 720)
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    ) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            analyzer.update_mouse(results)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
            )

            cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
