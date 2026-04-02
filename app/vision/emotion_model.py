from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


@dataclass(frozen=True)
class EmotionPrediction:
    probs: np.ndarray  # shape: (7,)

    @property
    def index(self) -> int:
        return int(np.argmax(self.probs))

    def label(self, labels: list[str] | tuple[str, ...] = DEFAULT_LABELS) -> str:
        return str(labels[self.index])


class EmotionModel:
    def __init__(self, model_path: str | Path) -> None:
        from tensorflow.keras.models import load_model  # type: ignore

        self.model_path = Path(model_path)
        self.model = load_model(str(self.model_path))

    def predict_probs_from_gray48(self, gray48: np.ndarray) -> EmotionPrediction:
        """
        Input: grayscale image 48x48 (uint8 or float), output: probabilities.
        """
        from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore

        roi = gray48.astype("float32") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        probs = self.model.predict(roi, verbose=0)[0]
        return EmotionPrediction(probs=np.asarray(probs, dtype=np.float32))

