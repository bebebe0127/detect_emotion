from pathlib import Path

import tensorflow as tf
from models.cnn_baseline import create_model
from data.dataloaders import get_generators
from utils.seed import DEFAULT_SEED, set_seed
from utils.visualization import plot_training_history

MODEL_PATH = "results/models/emotion_cnn.h5"
TRAINING_PLOT_PATH = "results/figures/training_history.png"


def train():
    set_seed(DEFAULT_SEED)
    train_gen, val_gen = get_generators("data/processed")

    model = create_model()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
    )

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    plot_training_history(history, save_path=TRAINING_PLOT_PATH)
    return history


if __name__ == "__main__":
    train()
