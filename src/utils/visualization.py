"""
Графики для отчёта и анализа: кривые обучения, матрица ошибок, примеры из датасета.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def _ensure_parent_dir(path: str | Path | None) -> None:
    if path is None:
        return
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)


def _show_or_save(save_path: str | Path | None) -> None:
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    classes: Sequence[str],
    *,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (8.0, 6.0),
) -> None:
    """Тепловая карта матрицы ошибок (подписи классов — по списку classes)."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
    )
    plt.xlabel("Предсказано")
    plt.ylabel("Истина")
    plt.title("Матрица ошибок")
    plt.tight_layout()
    _ensure_parent_dir(save_path)
    _show_or_save(save_path)


def plot_training_history(
    history: Any,
    *,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10.0, 4.0),
) -> None:
    """
    Графики loss и accuracy по эпохам (объект History Keras или словарь history.history).
    """
    h = history.history if hasattr(history, "history") else history
    if not isinstance(h, dict):
        raise TypeError("Ожидается History Keras или dict с ключами метрик.")

    loss_key = "loss" if "loss" in h else None
    val_loss_key = "val_loss" if "val_loss" in h else None
    acc_key = "accuracy" if "accuracy" in h else ("acc" if "acc" in h else None)
    val_acc_key = "val_accuracy" if "val_accuracy" in h else (
        "val_acc" if "val_acc" in h else None
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if loss_key:
        epochs = range(1, len(h[loss_key]) + 1)
        axes[0].plot(epochs, h[loss_key], label="train")
        if val_loss_key:
            axes[0].plot(epochs, h[val_loss_key], label="val")
        axes[0].set_xlabel("Эпоха")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].set_title("Функция потерь")
    else:
        axes[0].set_visible(False)

    if acc_key:
        epochs = range(1, len(h[acc_key]) + 1)
        axes[1].plot(epochs, h[acc_key], label="train")
        if val_acc_key:
            axes[1].plot(epochs, h[val_acc_key], label="val")
        axes[1].set_xlabel("Эпоха")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].set_title("Точность")
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    _ensure_parent_dir(save_path)
    _show_or_save(save_path)


def plot_dataset_class_samples(
    processed_dir: str | Path,
    *,
    split: str = "train",
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    n_per_class: int = 4,
    classes: Iterable[str] | None = None,
    save_path: str | Path | None = None,
    figsize_per_cell: tuple[float, float] = (1.6, 1.6),
) -> None:
    """
    Сетка примеров изображений по классам (папки внутри processed_dir/split/).
    Удобно для иллюстрации раздела о данных в ВКР.
    """
    from matplotlib import image as mpimg

    root = Path(processed_dir) / split
    if not root.is_dir():
        raise FileNotFoundError(f"Нет каталога: {root}")

    class_names = (
        sorted(p.name for p in root.iterdir() if p.is_dir())
        if classes is None
        else list(classes)
    )
    if not class_names:
        raise ValueError(f"В {root} нет подпапок классов.")

    n_classes = len(class_names)
    fig, axes = plt.subplots(
        n_classes,
        n_per_class,
        figsize=(n_per_class * figsize_per_cell[0], n_classes * figsize_per_cell[1]),
        squeeze=False,
    )

    for i, cls in enumerate(class_names):
        cdir = root / cls
        files = sorted(
            f
            for f in cdir.iterdir()
            if f.suffix.lower() in extensions and f.is_file()
        )[:n_per_class]
        for j in range(n_per_class):
            ax = axes[i][j]
            if j < len(files):
                img = mpimg.imread(files[j])
                if img.ndim == 2:
                    ax.imshow(img, cmap="gray")
                else:
                    ax.imshow(img)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(cls, rotation=90, size=9, labelpad=8)

    plt.suptitle(f"Примеры изображений ({split})", y=1.02)
    plt.tight_layout()
    _ensure_parent_dir(save_path)
    _show_or_save(save_path)
