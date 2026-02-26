from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_generators(data_dir, batch_size=64):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=(48, 48),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical"
    )

    val_gen = datagen.flow_from_directory(
        f"{data_dir}/val",
        target_size=(48, 48),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical"
    )

    return train_gen, val_gen
