from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int


class HaarFaceDetector:
    def __init__(self) -> None:
        self._detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect(self, gray_frame) -> list[FaceBox]:
        faces = self._detector.detectMultiScale(gray_frame, 1.3, 5)
        return [FaceBox(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

