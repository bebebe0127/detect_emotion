from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowInfo:
    handle: int
    title: str
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)


def list_visible_windows() -> list[WindowInfo]:
    """
    Returns top-level visible windows with non-empty titles.

    Uses pywin32 (`win32gui`). Kept isolated here to make non-Windows
    testing easier.
    """
    import win32gui  # type: ignore

    wins: list[WindowInfo] = []

    def enum_cb(hwnd: int, _lparam: int) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd) or ""
        title = title.strip()
        if not title:
            return
        try:
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        except Exception:
            return
        if right - left <= 0 or bottom - top <= 0:
            return
        wins.append(
            WindowInfo(
                handle=int(hwnd),
                title=title,
                left=int(left),
                top=int(top),
                right=int(right),
                bottom=int(bottom),
            )
        )

    win32gui.EnumWindows(enum_cb, 0)
    wins.sort(key=lambda w: w.title.lower())
    return wins


class WindowGrabber:
    def __init__(self) -> None:
        import mss  # type: ignore

        self._mss = mss.mss()

    def grab_bgr(self, win: WindowInfo) -> np.ndarray:
        """
        Grab window rectangle and return BGR image (OpenCV-friendly).

        Note: This captures screen pixels; if the window is occluded,
        the image may include occluding content.
        """
        monitor = {
            "left": int(win.left),
            "top": int(win.top),
            "width": int(win.width),
            "height": int(win.height),
        }
        raw = self._mss.grab(monitor)  # BGRA
        img = np.asarray(raw, dtype=np.uint8)
        # BGRA -> BGR
        return img[:, :, :3]

