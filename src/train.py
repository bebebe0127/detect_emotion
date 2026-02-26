import tensorflow as tf
from models.cnn_baseline import create_model
from data.dataloaders import get_generators

MODEL_PATH = "results/models/emotion_cnn.h5"

def train():
    train_gen, val_gen = get_generators("data/processed")

    model = create_model()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30
    )

    model.save(MODEL_PATH)

if __name__ == "__main__":
    train()
