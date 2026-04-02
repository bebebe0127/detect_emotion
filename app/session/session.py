from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from app.capture.window_capture import WindowGrabber, WindowInfo
from app.vision.emotion_model import DEFAULT_LABELS, EmotionModel
from app.vision.face_detector import HaarFaceDetector


@dataclass(frozen=True)
class SessionSummary:
    duration_s: float
    frames_total: int
    frames_with_faces: int
    frames_no_faces: int
    emotion_counts: dict[str, int]


class ResearchWorker(QThread):
    progress = Signal(int, int)  # frames_total, frames_with_faces
    finished_summary = Signal(object)  # SessionSummary
    failed = Signal(str)

    def __init__(
        self,
        window: WindowInfo,
        model_path: str | Path,
        target_fps: float = 8.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._window = window
        self._model_path = Path(model_path)
        self._target_dt = 1.0 / max(1e-3, float(target_fps))
        self._stop = False

        self._frames_total = 0
        self._frames_with_faces = 0
        self._emotion_counts = {k: 0 for k in DEFAULT_LABELS}

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            if not self._model_path.exists():
                self.failed.emit(
                    f"Не найдена модель: {self._model_path}. Сначала обучите модель (src/train.py) "
                    "или положите .h5 по указанному пути."
                )
                return

            grabber = WindowGrabber()
            face_detector = HaarFaceDetector()
            model = EmotionModel(self._model_path)
        except Exception as e:
            self.failed.emit(f"Ошибка инициализации: {e}")
            return

        started = time.perf_counter()
        last_tick = started

        while not self._stop:
            now = time.perf_counter()
            dt = now - last_tick
            if dt < self._target_dt:
                time.sleep(max(0.0, self._target_dt - dt))
                continue
            last_tick = time.perf_counter()

            try:
                frame_bgr = grabber.grab_bgr(self._window)
            except Exception:
                # Window moved/closed/coordinates invalid: just skip.
                continue

            self._frames_total += 1
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detect(gray)
            if faces:
                self._frames_with_faces += 1

                probs_sum = None
                for f in faces:
                    roi = gray[f.y : f.y + f.h, f.x : f.x + f.w]
                    if roi.size == 0:
                        continue
                    roi48 = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
                    pred = model.predict_probs_from_gray48(roi48)
                    probs_sum = pred.probs if probs_sum is None else (probs_sum + pred.probs)

                if probs_sum is not None:
                    probs_avg = probs_sum / float(np.sum(probs_sum) + 1e-8)
                    label = DEFAULT_LABELS[int(np.argmax(probs_avg))]
                    self._emotion_counts[label] = int(self._emotion_counts.get(label, 0) + 1)

            self.progress.emit(self._frames_total, self._frames_with_faces)

        duration_s = max(0.0, time.perf_counter() - started)
        frames_no_faces = self._frames_total - self._frames_with_faces
        summary = SessionSummary(
            duration_s=duration_s,
            frames_total=self._frames_total,
            frames_with_faces=self._frames_with_faces,
            frames_no_faces=frames_no_faces,
            emotion_counts=dict(self._emotion_counts),
        )
        self.finished_summary.emit(summary)

