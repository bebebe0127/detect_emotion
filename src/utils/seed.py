"""
Воспроизводимость экспериментов: единое значение seed для random / NumPy / TensorFlow.
"""

from __future__ import annotations

import os
import random

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Задаёт детерминизм насколько это поддерживают используемые библиотеки.
    Вызывать в начале prepare_dataset, train и скриптов оценки модели.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass
