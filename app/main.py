from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication

def _ensure_project_root_on_syspath() -> None:
    """
    Allow running either:
      - python -m app.main   (recommended)
      - python app/main.py   (supported)
    """
    if __package__:
        return
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_ensure_project_root_on_syspath()

from app.gui.main_window import MainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

