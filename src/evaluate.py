"""
Оценка модели на data/processed/test: classification report, матрица ошибок,
опционально — сетка примеров из train (иллюстрация для отчёта).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from data.dataloaders import get_test_generator
from utils.metrics import classification_report_metrics
from utils.seed import DEFAULT_SEED, set_seed
from utils.visualization import plot_confusion_matrix, plot_dataset_class_samples

DATA_DIR = "data/processed"
MODEL_PATH = "results/models/emotion_cnn.h5"
FIGURES_DIR = Path("results/figures")


def _class_names_in_label_order(test_gen) -> list[str]:
    return [
        name
        for name, _ in sorted(
            test_gen.class_indices.items(),
            key=lambda kv: kv[1],
        )
    ]


def evaluate(
    data_dir: str = DATA_DIR,
    model_path: str = MODEL_PATH,
) -> None:
    set_seed(DEFAULT_SEED)

    test_dir = Path(data_dir) / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Нет каталога {test_dir}. Сначала выполните: python src/data/prepare_dataset.py"
        )
    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Нет файла модели {model_path}. Сначала выполните: python src/train.py"
        )

    test_gen = get_test_generator(data_dir)
    class_names = _class_names_in_label_order(test_gen)
    y_true = test_gen.classes

    model = load_model(model_path)
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Classification report (test) ===")
    classification_report_metrics(y_true, y_pred, class_names)

    cm_path = FIGURES_DIR / "confusion_matrix.png"
    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        save_path=cm_path,
    )
    print(f"Матрица ошибок сохранена: {cm_path}")

    train_dir = Path(data_dir) / "train"
    if train_dir.is_dir():
        samples_path = FIGURES_DIR / "dataset_samples_train.png"
        plot_dataset_class_samples(
            data_dir,
            split="train",
            n_per_class=3,
            save_path=samples_path,
        )
        print(f"Примеры изображений (train): {samples_path}")


if __name__ == "__main__":
    evaluate()
