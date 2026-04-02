from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.capture.window_capture import WindowInfo, list_visible_windows
from app.report.summary import build_popup_summary
from app.session.session import ResearchWorker, SessionSummary


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Emotion Research")
        self.setMinimumWidth(520)

        self._timer = QElapsedTimer()
        self._ui_tick = QTimer(self)
        self._ui_tick.setInterval(250)
        self._ui_tick.timeout.connect(self._on_tick)
        self._worker: ResearchWorker | None = None

        root = QWidget(self)
        self.setCentralWidget(root)

        self.window_combo = QComboBox()
        self.refresh_btn = QPushButton("Обновить список окон")
        self.start_btn = QPushButton("Начать исследование")
        self.stop_btn = QPushButton("Остановить")
        self.status_lbl = QLabel("Готово. Выберите окно и нажмите «Начать исследование».")
        self.time_lbl = QLabel("00:00")

        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(False)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Окно:"))
        top_row.addWidget(self.window_combo, 1)
        top_row.addWidget(self.refresh_btn)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(QLabel("Время:"))
        btn_row.addWidget(self.time_lbl)

        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addLayout(btn_row)
        layout.addWidget(self.status_lbl)
        root.setLayout(layout)

        self.refresh_btn.clicked.connect(self._refresh_windows)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.window_combo.currentIndexChanged.connect(self._on_window_selected)

        self._refresh_windows()

    def _refresh_windows(self) -> None:
        self.window_combo.clear()
        try:
            windows = list_visible_windows()
        except Exception as e:
            self.window_combo.addItem("— ошибка перечисления окон —", userData=None)
            self.start_btn.setEnabled(False)
            self.status_lbl.setText(f"Не удалось получить список окон: {e}")
            return

        if not windows:
            self.window_combo.addItem("— окна не найдены —", userData=None)
            self.start_btn.setEnabled(False)
            self.status_lbl.setText("Не найдено ни одного подходящего окна. Откройте приложение видеозвонка и обновите список.")
            return

        for w in windows:
            self.window_combo.addItem(w.title, userData=w)

        self.status_lbl.setText("Выберите окно видеозвонка и нажмите «Начать исследование».")
        self._on_window_selected()

    def _on_window_selected(self) -> None:
        choice = self.window_combo.currentData()
        self.start_btn.setEnabled(isinstance(choice, WindowInfo) and not self._timer.isValid())

    def _start(self) -> None:
        if self._timer.isValid():
            return
        win: WindowInfo | None = self.window_combo.currentData()
        if win is None:
            QMessageBox.warning(self, "Не выбрано окно", "Выберите окно видеозвонка для анализа.")
            return

        model_path = Path("results/models/emotion_cnn.h5")
        self._worker = ResearchWorker(window=win, model_path=model_path, target_fps=8.0, parent=self)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished_summary.connect(self._on_worker_finished)
        self._worker.start()

        self._timer.start()
        self._ui_tick.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.refresh_btn.setEnabled(False)
        self.window_combo.setEnabled(False)
        self.status_lbl.setText("Исследование запущено. Идёт анализ…")

    def _stop(self) -> None:
        if not self._timer.isValid():
            return
        if self._worker is not None:
            self._worker.request_stop()

        self.stop_btn.setEnabled(False)
        self.status_lbl.setText("Остановка…")

    def _on_worker_progress(self, frames_total: int, frames_with_faces: int) -> None:
        self.status_lbl.setText(
            f"Идёт анализ… кадров: {frames_total}, с лицами: {frames_with_faces}."
        )

    def _on_worker_failed(self, message: str) -> None:
        self._cleanup_after_stop()
        QMessageBox.critical(self, "Ошибка", message)

    def _on_worker_finished(self, summary_obj: object) -> None:
        self._cleanup_after_stop()
        summary = summary_obj if isinstance(summary_obj, SessionSummary) else None
        if summary is None:
            QMessageBox.information(self, "Итоги разговора", "Сессия завершена.")
            return
        popup = build_popup_summary(summary)
        QMessageBox.information(self, popup.title, popup.text)

    def _cleanup_after_stop(self) -> None:
        if self._timer.isValid():
            self._timer.invalidate()
        self._ui_tick.stop()
        self.start_btn.setEnabled(self.window_combo.currentData() is not None)
        self.stop_btn.setEnabled(False)
        self.refresh_btn.setEnabled(True)
        self.window_combo.setEnabled(True)
        self.status_lbl.setText("Готово. Выберите окно и нажмите «Начать исследование».")
        self._worker = None

    def _on_tick(self) -> None:
        if not self._timer.isValid():
            self.time_lbl.setText("00:00")
            return

        elapsed_ms = self._timer.elapsed()
        mins = elapsed_ms // 60000
        secs = (elapsed_ms % 60000) // 1000
        self.time_lbl.setText(f"{mins:02d}:{secs:02d}")
        self.status_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

