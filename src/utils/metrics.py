from sklearn.metrics import classification_report, confusion_matrix

# Графики вынесены в utils.visualization; для совместимости реэкспорт:
from utils.visualization import plot_confusion_matrix


def classification_report_metrics(y_true, y_pred, classes):
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)


__all__ = [
    "classification_report",
    "confusion_matrix",
    "classification_report_metrics",
    "plot_confusion_matrix",
]
